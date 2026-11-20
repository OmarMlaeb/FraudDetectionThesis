"""Audit purchaser/recipient email domains using fixed splits and verified predictions.

Run with the project's portable Python. Writes only into this analysis directory.
No model training, inference, threshold tuning, or existing-result changes.
"""
import hashlib
import html
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
AUDIT = ROOT / 'analysis/supervisor_2026-10-13'
FIELDS = ['P_emaildomain', 'R_emaildomain']
SEEDS = list(range(42, 52))
MODELS = {'GraphSAGE-email': 'GraphSAGE-email_domain', 'GraphSAGE-main': 'GraphSAGE-main', 'MLP': 'MLP'}
MISSING = '<MISSING>'
REASONS = ['missing', 'singleton', 'over_1000', 'eligible']
PROTOCOL = {
    'date': '2026-11-20',
    'training_min_transactions': 100,
    'training_min_fraud_rate_ratio': 2.0,
    'training_wilson_95_lower_must_exceed_overall_training_rate': True,
    'screening_rule_fixed_before_validation_test_rates_or_model_subgroups': True,
    'screening_is_exploratory_not_a_multiple_testing_corrected_discovery_test': True,
    'graph_max_group_size': 1000,
    'graph_group_size_scope': 'All 590540 nodes, separately per raw email field, as in saved graph.',
    'splits': 'Audited temporal masks, sorted by TransactionDT then TransactionID; 80/10/10.',
    'outside_definition': 'Every split transaction not in the selected domain/domain set, including missing values.',
    'model_selection': 'Accepted historical runs, seeds 42-51, saved validation thresholds, verified prediction hashes.',
    'metrics': 'Average precision (PR-AUC), F1, recall, precision, confusion counts; sample SD ddof=1.',
    'uncertainty': 'Seed SD describes training variation on fixed IDs; Wilson bounds are marginal binomial descriptive bounds.',
    'tests': 'No new significance tests. Paired model differences are descriptive. Existing Holm families unchanged.',
}
INPUTS = {}


def check(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path, expected=None):
    path = Path(path)
    key = path.relative_to(ROOT).as_posix()
    digest = sha(path)
    check(expected is None or digest == expected, 'Source hash mismatch: '+key)
    INPUTS[key] = digest
    return path


def wilson(k, n):
    if n == 0:
        return float('nan'), float('nan')
    p, z = k/n, 1.959963984540054
    mid = (p+z*z/(2*n))/(1+z*z/n)
    width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return mid-width, mid+width


def population(y):
    n, f = len(y), int(np.sum(y))
    lo, hi = wilson(f, n)
    return dict(transactions=n, frauds=f, fraud_rate=f/n if n else np.nan,
                wilson95_lower=lo, wilson95_upper=hi)


def metrics(y, p, threshold):
    n, f = len(y), int(np.sum(y))
    pred = p >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = f-tp, n-f-fp
    return dict(transactions=n, frauds=f, true_positive=tp, false_positive=fp,
                false_negative=fn, true_negative=tn,
                pr_auc=float(average_precision_score(y, p)) if f else np.nan,
                f1_score=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                recall=tp/f if f else np.nan, precision=tp/(tp+fp) if tp+fp else np.nan)


def save_csv(name, data):
    data.to_csv(OUT / name, index=False, float_format='%.17g', na_rep='')


def display_table(frame, rename=None, rates=(), decimals=4):
    frame = frame.copy()
    for col in rates:
        if col in frame:
            frame[col] = frame[col].map(lambda x: f'{x:.2%}' if pd.notna(x) else 'undefined')
    if rename:
        frame = frame.rename(columns=rename)
    return '<div class="table-wrap">'+frame.to_html(index=False, escape=True, border=0,
                float_format=lambda x: f'{x:.{decimals}f}', na_rep='undefined')+'</div>'


def write_report(domains, risks, exclusions, joint, baselines, groups, summary, paired, verification):
    def metric_cell(mean, sd, count):
        if pd.isna(mean):
            return 'undefined (0/10 seeds)'
        uncertainty = f'{sd:.4f}' if pd.notna(sd) else 'undefined SD'
        return f'{mean:.4f} ± {uncertainty}' + (f' ({count}/10 defined)' if count!=10 else '')

    def model_table(frame, detail=False):
        cols = ['field','domain','side','model','transactions','frauds','fraud_rate'] if detail else [
            'field','side','model','transactions','frauds','fraud_rate']
        shown = frame[cols].copy()
        for metric in ['pr_auc','f1_score','recall','precision','false_positive']:
            shown[metric+' mean +/- sample SD'] = [metric_cell(a,b,int(n)) for a,b,n in zip(
                frame[metric+'_mean'],frame[metric+'_sample_sd'],frame[metric+'_defined_seeds'])]
        return display_table(shown,rates=['fraud_rate'])

    persistence_rows = []
    for (field,domain),frame in risks.groupby(['field','domain'],sort=False):
        row = dict(field=field,domain=domain)
        for split in ['train','validation','test']:
            r = frame[frame.split==split].iloc[0]
            row[split+' fraud rate (frauds / transactions)'] = f'{r.fraud_rate:.2%} ({r.frauds:,} / {r.transactions:,})'
        r = frame[frame.split=='test'].iloc[0]
        row['test share of all fraud'] = f'{r.share_of_all_fraud:.2%}'
        row['domain link eligibility'] = r.graph_group_reason
        row['test linked via either email'] = r.linked_via_either_email
        persistence_rows.append(row)
    persistence = pd.DataFrame(persistence_rows)
    rpool = groups[(groups.split=='test')&(groups.field=='R_emaildomain')&(groups.scope=='selected_union')&(groups.side=='inside')].iloc[0]
    epool = summary[(summary.split=='test')&(summary.scope=='selected_union')]
    email = epool[epool.model=='GraphSAGE-email'].set_index(['field','side'])
    mlp = epool[epool.model=='MLP'].set_index(['field','side'])
    mlp_wins = int(((mlp.pr_auc_mean>email.pr_auc_mean)&(mlp.f1_score_mean>email.f1_score_mean)).sum())
    pieces = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Email-domain analysis</title>',
        '<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#183039}'
        'h1,h2,h3{line-height:1.2}h2{margin-top:38px}table{border-collapse:collapse;font-size:13px;width:100%}'
        'th,td{padding:7px 9px;border-bottom:1px solid #d8e1e3;text-align:right;white-space:nowrap}'
        'th:first-child,td:first-child{text-align:left}th{background:#eaf1f2}tr:nth-child(even){background:#f8fafb}'
        '.table-wrap{overflow-x:auto}summary{cursor:pointer;font-weight:600;margin:18px 0}a{color:#006d77}'
        'code{font-size:13px}p{max-width:1000px}.note{border-left:3px solid #006d77;padding-left:14px}</style>',
        '<h1>Purchaser and recipient email domains</h1>',
        '<p>IEEE-CIS analysis, 20 November 2026. Fields are analyzed separately with their original spelling. '
        'Missing values remain a separate category. All counts refer to transactions; fraud is the supplied isFraud label. '
        'A domain label identifies an observed transaction attribute, not wrongdoing by the email provider.</p>',
        '<h2>Findings</h2>',
        '<p>Risk depends strongly on whether a domain is the purchaser or recipient field. Some training-selected '
        'domains retain elevated fraud rates in validation and test, while others have very small held-out groups '
        'and no observed test fraud. The full counts below distinguish high fraud volume from high fraud rate.</p>',
        f'<p class="note">Among the {int(rpool.transactions):,} test transactions in the training-selected recipient domains, '
        f'{int(rpool.frauds):,} are fraudulent. Only {int(rpool.connected_transactions):,} transactions have an email-graph neighbour, '
        f'and {int(rpool.connected_frauds):,} of those are frauds. Thus {int(rpool.frauds-rpool.connected_frauds):,} '
        'frauds in this group are isolated in the email graph. Large selected domains are excluded by the group-size limit.</p>',
        f'<p>MLP has higher mean test PR-AUC and F1 than email-GraphSAGE in {mlp_wins} of the four pooled '
        'inside/outside populations. Email-GraphSAGE versus main-GraphSAGE varies by field and domain. '
        'These observations do not support a general explanation that linking high-risk email domains makes '
        'email-GraphSAGE superior. The individual-domain tables preserve exceptions and threshold-dependent differences.</p>',
        '<h2>Scope and interpretation</h2>',
        '<p>The risk screen was fixed before examining held-out rates: at least 100 training transactions, '
        'training fraud rate at least twice the overall training rate, and its marginal 95% Wilson lower bound above that rate. '
        'This is an exploratory screen, not a multiplicity-corrected significance test. Selected domains are frozen for '
        'validation/test; no domains are added using future labels. Wilson intervals assume binomial observations and '
        'do not adjust for repeated-entity dependence. Persistence below is descriptive relative to each split baseline.</p>',
        display_table(baselines, rates=['fraud_rate','wilson95_lower','wilson95_upper']),
        '<p>The graph admits domain groups of 2–1,000 transactions inclusive, counted across the full saved graph '
        '(train + validation + test), and links consecutive transactions within each field. It does not connect purchaser '
        'values directly to recipient values. A transaction excluded through one field may still receive links through the other. '
        'The graph is the historical transductive graph; these counts do not describe an online-only graph.</p>',
        '<h2>Link eligibility and exclusions</h2>',
        display_table(exclusions, rates=['transaction_share','fraud_rate','share_of_all_fraud']),
        '<details><summary>Joint purchaser/recipient reasons (no double counting)</summary>',
        display_table(joint, rates=['fraud_rate']), '</details>',
        '<h2>Training-selected domains and persistence</h2>',
        '<p>Share of all fraud divides domain frauds by all frauds in the same split, including transactions with missing '
        'email values. Lift divides the domain fraud rate by that split’s overall fraud rate. '
        'An undefined held-out rate means no transactions, not zero risk.</p>',
        display_table(persistence),
        '<p>Zero observed fraud in a small group does not establish that the underlying risk disappeared. '
        'The complete per-split counts, fraud shares, lifts and Wilson bounds are in risk_persistence.csv.</p>',
        '<h2>Same-ID model comparisons</h2>',
        '<p>For every group, all three models use exactly the same transaction IDs and labels for every seed, 42–51. '
        'AP is recomputed from saved probabilities. F1 uses each original saved validation threshold; thresholds are not '
        'retuned for a domain. Mean ± sample SD (ddof=1) describes variation across seeds on a fixed population. '
        'AP should be compared across models within a row population; different inside/outside fraud prevalence makes '
        'their absolute AP values unsuitable as a direct comparison of difficulty or topology benefit. '
        'Validation metrics are descriptive because validation was used to select the original models and thresholds. '
        'AP and recall are undefined with no positive labels; precision is undefined with no positive predictions; '
        'F1 is undefined when its denominator is zero. Available-seed counts are shown when fewer than ten values are defined.</p>']
    for split in ['test','validation']:
        chosen = summary[(summary.split == split) & (summary.scope == 'selected_union')]
        pieces += [f'<h3>{split.capitalize()}: union of training-selected domains in each field</h3>',
                   model_table(chosen)]
        for scope,title in [('individual_domain','Each selected domain and its complement'),
                            ('missingness','Missing email and known nonselected domains'),
                            ('selected_union_connected','Connected subset of the selected-domain groups'),
                            ('selected_union_isolated','Isolated subset of the selected-domain groups')]:
            pieces += [f'<details><summary>{split.capitalize()}: {title}</summary>',
                       model_table(summary[(summary.split==split)&(summary.scope==scope)],detail=True), '</details>']
    pieces += [
               '<details><summary>Paired differences between models on identical IDs</summary>',
               '<p>A minus B, paired by seed. These are descriptive effects; no new significance tests were run and '
               'no existing Holm correction family was changed. Median and mean differences in AP do not add across subgroups.</p>',
               display_table(paired), '</details>']
    for field in FIELDS:
        pieces += [f'<h2>Every domain: {field}</h2>']
        for split in ['all','train','validation','test']:
            frame = domains[(domains.field == field) & (domains.split == split)].sort_values(
                ['frauds','transactions'], ascending=False)
            pieces += [f'<details{" open" if split == "all" else ""}><summary>{split.capitalize()}: all domains and missing values</summary>',
                       display_table(frame[['domain','transactions','frauds','fraud_rate','share_of_all_fraud',
                                           'transaction_share','rate_ratio_vs_split','selected_on_training',
                                           'graph_group_reason','linked_via_either_email']],
                                     rates=['fraud_rate','share_of_all_fraud','transaction_share']), '</details>']
    files = ['domain_counts.csv','training_risk_selection.csv','risk_persistence.csv','link_exclusions.csv',
             'joint_link_exclusions.csv','model_group_counts.csv','model_summary.csv','model_per_seed.csv',
             'paired_model_differences.csv','split_baselines.csv','protocol.json','provenance.json','verification.json']
    pieces += ['<h2>Data files and verification</h2><ul>']
    pieces += [f'<li><a href="{f}">{f}</a></li>' for f in files]
    pieces += ['</ul><pre>'+html.escape(json.dumps(verification,indent=2))+'</pre>',
               '<p>Primary inputs: data/ieee-cis/train_transaction.csv; '
               'analysis/supervisor_2026-10-13/node_mappings/IEEE-CIS.npz; selected_per_seed_results.csv; '
               'accepted checks and predictions for MLP, GraphSAGE-main and GraphSAGE-email_domain; '
               'results/ieee_cis_graph_edges_email_domain_max1000.csv. Exact paths and SHA-256 hashes are in provenance.json.</p>',
               '<p>These comparisons assess whether the proposed explanation is consistent with the observed domain risks '
               'and performance. They do not isolate a causal effect of email edges: full-graph and email models have '
               'different trained parameters, and even isolated nodes can be affected by shared parameters.</p></html>']
    (OUT/'report.html').write_text('\n'.join(pieces),encoding='utf-8')


def main():
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    print('Loading raw email fields and checking audited split IDs...',flush=True)
    archived = {r['path']:r['sha256'] for r in json.loads((AUDIT/'artifact_hashes_before.json').read_text())}
    raw_path = ROOT/'data/ieee-cis/train_transaction.csv'
    record(raw_path, archived[raw_path.relative_to(ROOT).as_posix()])
    d = pd.read_csv(raw_path,usecols=['TransactionID','TransactionDT','isFraud',*FIELDS])
    d = d.sort_values(['TransactionDT','TransactionID'],kind='mergesort').reset_index(drop=True)
    record(ROOT/'src/ieee_cis/graph_data.py')
    record(ROOT/'src/ieee_cis/preprocessing.py')
    record(Path(__file__))
    with np.load(record(AUDIT/'node_mappings/IEEE-CIS.npz'),allow_pickle=False) as z:
        check(np.array_equal(d.TransactionID,z['node_ids']), 'Raw transaction ID order differs from audited mapping.')
        check(np.array_equal(d.isFraud,z['labels']), 'Raw fraud labels differ from audit.')
        masks = {'all':np.ones(len(d),bool),'train':z['train_mask'].copy(),
                 'validation':z['val_mask'].copy(),'test':z['test_mask'].copy()}
    check(len(d)==590540 and d.TransactionID.is_unique, 'Unexpected raw population.')
    check([int(masks[s].sum()) for s in ['train','validation','test']]==[472432,59054,59054], 'Wrong split sizes.')
    baselines = pd.DataFrame([dict(split=s,**population(d.loc[mask,'isFraud'])) for s,mask in masks.items()])
    train_rate = float(d.loc[masks['train'],'isFraud'].mean())
    selections, train_rows, reason = {}, [], {}
    edge_sets = {}
    for field in FIELDS:
        # Select exclusively from the training partition, before evaluating held-out rates.
        tr = d.loc[masks['train']].groupby(field).isFraud.agg(['size','sum','mean'])
        chosen = []
        for domain,row in tr.iterrows():
            lo,hi = wilson(int(row['sum']),int(row['size']))
            flag = row['size']>=100 and row['mean']>=2*train_rate and lo>train_rate
            train_rows.append(dict(field=field,domain=domain,transactions=int(row['size']),frauds=int(row['sum']),
                fraud_rate=row['mean'],wilson95_lower=lo,wilson95_upper=hi,training_baseline=train_rate,
                training_rate_ratio=row['mean']/train_rate,selected=bool(flag)))
            if flag:
                chosen.append(domain)
        selections[field] = sorted(chosen)
        sizes = d[field].value_counts(dropna=True)
        n = d[field].map(sizes)
        reason[field] = np.select([d[field].isna(),n.eq(1),n.gt(1000)],REASONS[:3],default='eligible')
        edges = set()
        for domain,g in d[d[field].notna()].groupby(field,sort=False):
            if 2 <= len(g) <= 1000:
                ids = g.sort_values('TransactionDT').index.to_numpy()
                edges.update((int(a),int(b)) for a,b in zip(ids[:-1],ids[1:]))
                edges.update((int(b),int(a)) for a,b in zip(ids[:-1],ids[1:]))
        edge_sets[field] = edges
        print(field,'training-selected:',', '.join(chosen),flush=True)
    # Frozen list is saved before held-out domain-rate calculations.
    (OUT/'selected_domains.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    save_csv('training_risk_selection.csv',pd.DataFrame(train_rows))
    edges_path = record(ROOT/'results/ieee_cis_graph_edges_email_domain_max1000.csv')
    e = pd.read_csv(edges_path).to_numpy(dtype=np.int64)
    actual_edges = set(map(tuple,e))
    rebuilt = edge_sets[FIELDS[0]] | edge_sets[FIELDS[1]]
    check(rebuilt == actual_edges, 'Reconstructed per-field chain union differs from saved edge CSV.')
    connected = np.zeros(len(d),bool)
    connected[e.ravel()] = True
    check(np.array_equal(connected,(reason[FIELDS[0]]=='eligible')|(reason[FIELDS[1]]=='eligible')),
          'Field eligibility does not match saved graph connectivity.')
    with np.load(record(AUDIT/'subgroup_ids/IEEE-CIS__email_domain__connected.npz')) as z:
        check(np.array_equal(d.loc[masks['test']&connected,'TransactionID'],z['transaction_ids']), 'Audited connected IDs differ.')
    domain_rows, exclusion_rows, joint_rows = [], [], []
    for split,mask in masks.items():
        base = population(d.loc[mask,'isFraud'])
        sub = d.loc[mask]
        for field in FIELDS:
            for domain in sorted(d[field].dropna().unique().tolist())+[MISSING]:
                member = d[field].isna().to_numpy() if domain == MISSING else d[field].eq(domain).to_numpy()
                take = mask & member
                v = population(d.loc[take,'isFraud'])
                full_n = int(member.sum())
                why = 'missing' if domain == MISSING else 'singleton' if full_n==1 else 'over_1000' if full_n>1000 else 'eligible'
                domain_rows.append(dict(field=field,domain=domain,split=split,**v,
                    transaction_share=v['transactions']/base['transactions'],
                    share_of_all_fraud=v['frauds']/base['frauds'],
                    rate_ratio_vs_split=v['fraud_rate']/base['fraud_rate'],
                    selected_on_training=domain in selections[field],full_graph_group_size=full_n,
                    graph_group_reason=why,linked_via_this_field=int(np.sum(take&(reason[field]=='eligible'))),
                    linked_via_either_email=int(np.sum(take&connected)),
                    above_split_baseline=bool(v['fraud_rate']>base['fraud_rate']) if v['transactions'] else None,
                    at_least_2x_split_baseline=bool(v['fraud_rate']>=2*base['fraud_rate']) if v['transactions'] else None))
            for why in REASONS:
                take = mask & (reason[field]==why)
                v = population(d.loc[take,'isFraud'])
                exclusion_rows.append(dict(field=field,split=split,reason=why,**v,
                    transaction_share=v['transactions']/base['transactions'],share_of_all_fraud=v['frauds']/base['frauds'],
                    linked_via_other_field_despite_exclusion=int(np.sum(take&connected)) if why!='eligible' else 0))
        for a,b in itertools.product(REASONS,repeat=2):
            take = mask & (reason[FIELDS[0]]==a) & (reason[FIELDS[1]]==b)
            joint_rows.append(dict(split=split,purchaser_reason=a,recipient_reason=b,
                                   no_email_link=a!='eligible' and b!='eligible',**population(d.loc[take,'isFraud'])))
    domains = pd.DataFrame(domain_rows)
    exclusions, joint = pd.DataFrame(exclusion_rows), pd.DataFrame(joint_rows)
    risks = domains[domains.selected_on_training & domains.split.ne('all')].copy()
    check(domains.groupby(['field','split']).transactions.sum().tolist()==[int(masks[s].sum()) for f,s in domains.groupby(['field','split']).groups], 'Domain transaction totals do not reconcile.')
    for field in FIELDS:
        for split,mask in masks.items():
            frame = domains[(domains.field==field)&(domains.split==split)]
            check(int(frame.frauds.sum())==int(d.loc[mask,'isFraud'].sum()), 'Domain fraud totals do not reconcile.')
    for name,frame in [('domain_counts.csv',domains),('risk_persistence.csv',risks),('link_exclusions.csv',exclusions),
                       ('joint_link_exclusions.csv',joint),('split_baselines.csv',baselines)]:
        save_csv(name,frame)
    print('Verified saved edge union:',len(actual_edges),'directed edges;',int(connected.sum()),'connected nodes.',flush=True)

    # Identical groups reused for all models and all seeds. Include each domain versus its complement.
    defs = []
    for field in FIELDS:
        scopes = [('selected_union','ALL_SELECTED',d[field].isin(selections[field]).to_numpy())]
        scopes += [('individual_domain',domain,d[field].eq(domain).to_numpy()) for domain in selections[field]]
        for scope,domain,member in scopes:
            for side,take in [('inside',member),('outside',~member)]:
                defs.append((field,scope,domain,side,take))
        member = d[field].isin(selections[field]).to_numpy()
        defs += [(field,'missingness','MISSING','missing',d[field].isna().to_numpy()),
                 (field,'missingness','KNOWN_NONSELECTED','outside_known',~member&d[field].notna().to_numpy())]
        for side,take in [('inside',member),('outside',~member)]:
            for status,linkmask in [('connected',connected),('isolated',~connected)]:
                defs.append((field,'selected_union_'+status,'ALL_SELECTED',side,take&linkmask))
    defs.append(('all','whole_split','ALL','all',np.ones(len(d),bool)))
    group_rows, group_masks = [], {}
    memberships = {}
    for split in ['validation','test']:
        for field,scope,domain,side,take in defs:
            take = take & masks[split]
            key = (split,field,scope,domain,side)
            ids = d.loc[take,'TransactionID'].to_numpy(dtype=np.int64)
            group_masks[key] = take[masks[split]]
            memberships['__'.join(key)] = ids
            group_rows.append(dict(zip(['split','field','scope','domain','side'],key),
                **population(d.loc[take,'isFraud']),
                share_of_split_fraud=int(d.loc[take,'isFraud'].sum())/int(d.loc[masks[split],'isFraud'].sum()),
                connected_transactions=int(np.sum(take&connected)), connected_frauds=int(d.loc[take&connected,'isFraud'].sum()),
                isolated_transactions=int(np.sum(take&~connected)),ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
    groups = pd.DataFrame(group_rows)
    np.savez_compressed(OUT/'group_transaction_ids.npz',**memberships)
    save_csv('model_group_counts.csv',groups)
    results = pd.read_csv(record(AUDIT/'selected_per_seed_results.csv'),keep_default_na=False)
    rows = []
    verified = 0
    for model,source_model in MODELS.items():
        for seed in SEEDS:
            run_id = f'IEEE-CIS__{source_model}__seed{seed}'
            entry = results[results.run_id.eq(run_id)]
            check(len(entry)==1, 'Missing selected run '+run_id)
            entry = entry.iloc[0]
            audit = json.loads(record(AUDIT/'checks'/f'{run_id}.json').read_text())
            check(audit['accepted'] and audit['checkpoint_sha256']==entry.checkpoint_sha256, 'Checkpoint not accepted.')
            t = float(entry.threshold)
            check(t==float(audit['saved_threshold']), 'Saved thresholds differ.')
            with np.load(record(AUDIT/audit['prediction_file'],audit['prediction_sha256']),allow_pickle=False) as z:
                check(float(z['threshold'])==t, 'Prediction threshold differs.')
                for split,prefix in [('validation','val'),('test','test')]:
                    y,p,ids = z[prefix+'_y'],z[prefix+'_probs'],z[prefix+'_ids']
                    check(np.array_equal(ids,d.loc[masks[split],'TransactionID']), 'Prediction IDs differ from group universe.')
                    check(np.array_equal(y,d.loc[masks[split],'isFraud']), 'Prediction labels differ from raw labels.')
                    whole = metrics(y,p,t)
                    if split=='test':
                        for metric in ['true_positive','true_negative','false_positive','false_negative']:
                            check(whole[metric]==int(entry[metric])==audit['whole_test_metrics'][metric], 'Whole-test counts differ.')
                        check(abs(whole['pr_auc']-float(entry.pr_auc))<=2e-5, 'Whole-test AP differs.')
                    for key,take in group_masks.items():
                        if key[0]!=split:
                            continue
                        rows.append(dict(zip(['split','field','scope','domain','side'],key),
                            model=model,seed=seed,run_id=run_id,saved_threshold=t,**metrics(y[take],p[take],t)))
            verified += 1
        print('Verified and evaluated all ten seeds:',model,flush=True)
    per_seed = pd.DataFrame(rows)
    keys = ['split','field','scope','domain','side']
    stats_rows, pair_rows = [], []
    measures = ['pr_auc','f1_score','recall','precision','false_positive','true_positive','false_negative','true_negative']
    for key,frame in per_seed.groupby(keys+['model'],sort=False):
        check(len(frame)==10 and set(frame.seed)==set(SEEDS), 'Incomplete seeds in subgroup.')
        row = dict(zip(keys+['model'],key),seeds=10)
        counts = groups
        for name,value in zip(keys,key[:-1]):
            counts = counts[counts[name].eq(value)]
        check(len(counts)==1,'Missing group counts.')
        for name in ['transactions','frauds','fraud_rate','connected_transactions','connected_frauds','ids_sha256']:
            row[name] = counts.iloc[0][name]
        for metric in measures:
            row[metric+'_mean'] = frame[metric].mean()
            row[metric+'_sample_sd'] = frame[metric].std(ddof=1)
            row[metric+'_defined_seeds'] = int(frame[metric].notna().sum())
        stats_rows.append(row)
    for key,frame in per_seed.groupby(keys,sort=False):
        for a,b in itertools.combinations(MODELS,2):
            fa = frame[frame.model.eq(a)].set_index('seed').sort_index()
            fb = frame[frame.model.eq(b)].set_index('seed').sort_index()
            check(fa.index.equals(fb.index), 'Unpaired seeds.')
            for metric in ['pr_auc','f1_score','recall','precision','false_positive']:
                delta = fa[metric]-fb[metric]
                valid = delta.dropna()
                pair_rows.append(dict(zip(keys,key),model_a=a,model_b=b,metric=metric,n_pairs=10,
                    defined_pairs=len(valid),mean_a_minus_b=valid.mean() if len(valid) else np.nan,
                    sample_sd_a_minus_b=valid.std(ddof=1) if len(valid)>1 else np.nan,
                    median_a_minus_b=valid.median() if len(valid) else np.nan,
                    seeds_a_greater=int(delta.gt(0).sum()),seeds_b_greater=int(delta.lt(0).sum()),
                    seeds_equal=int(delta.eq(0).sum())))
    summary, paired = pd.DataFrame(stats_rows), pd.DataFrame(pair_rows)
    # Every individual-domain and selected-union inside/outside pair partitions the whole split exactly.
    for key,frame in per_seed[per_seed.scope.isin(['individual_domain','selected_union'])].groupby(
            ['split','field','scope','domain','model','seed'],sort=False):
        whole = per_seed[(per_seed.split==key[0])&(per_seed.scope=='whole_split')&(per_seed.model==key[4])&(per_seed.seed==key[5])].iloc[0]
        check(len(frame)==2,'Missing complementary group.')
        for metric in ['transactions','frauds','true_positive','true_negative','false_positive','false_negative']:
            check(int(frame[metric].sum())==int(whole[metric]), 'Complementary subgroup count reconciliation failed.')
    save_csv('model_per_seed.csv',per_seed)
    save_csv('model_summary.csv',summary)
    save_csv('paired_model_differences.csv',paired)
    verification = dict(raw_nodes=len(d),raw_frauds=int(d.isFraud.sum()),raw_hash_matches_prior_audit=True,
        node_ids_and_labels_match_audit=True,split_sizes_match_audit=True,
        reconstructed_email_edges_match_saved_csv=True,directed_edges=len(actual_edges),
        undirected_edges=len(actual_edges)//2,connected_nodes=int(connected.sum()),isolated_nodes=int((~connected).sum()),
        test_connected_ids_match_prior_audit=True,verified_prediction_files=verified,
        identical_model_ids_and_labels=True,all_domain_counts_reconcile=True,
        all_inside_outside_confusion_counts_reconcile=True,per_seed_metric_rows=len(per_seed),
        group_definitions=len(groups),domain_summary_rows=len(domains),selected_domains=selections)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    (OUT/'provenance.json').write_text(json.dumps({'protocol':PROTOCOL,'input_sha256':INPUTS},indent=2),encoding='utf-8')
    write_report(domains,risks,exclusions,joint,baselines,groups,summary,paired,verification)
    print(json.dumps(verification,indent=2),flush=True)
    print('Report:',OUT/'report.html')


if __name__=='__main__':
    main()
