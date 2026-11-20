"""Preserve thesis tables as printed and compare to audited saved runs."""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
raw = json.loads((OUT / 'raw_tables.json').read_text(encoding='utf-8'))
runs = json.loads((OUT.parent / 'run_manifest.json').read_text())
titles = [
    'Graph Statistics for Original and Sampled Complement Graphs',
    'Structural Properties of Alternative IEEE-CIS Graph Constructions',
    'Isolated Transactions in IEEE-CIS Graph Constructions',
    'Class-Specific Connectivity of Alternative IEEE-CIS Graphs',
    'IEEE-CIS GNN Performance Across Graph Constructions: Mean ± Standard Deviation Across 10 Runs',
    'IEEE-CIS Test-Set False Positives Across Graph Constructions: Mean ± Sample Standard Deviation Across 10 Runs',
    'Mean Performance Across 10 Runs on IEEE-CIS',
    'Mean Performance Across 10 Runs on Elliptic',
    'Final-Run Results and Confusion Matrices on IEEE-CIS',
    'Final-Run Results and Confusion Matrices on Elliptic',
    'Sampled Complement Graph Ablation Results',
    'Main Wilcoxon Findings on IEEE-CIS',
    'Main Wilcoxon Findings on Elliptic',
]
replacements = {
    'GraphVariant': 'Graph Variant', 'AverageDegree': 'Average Degree',
    'ConnectedComponents': 'Connected Components', 'SampledComplement': 'Sampled Complement',
    'FraudNodeswith\nAnyNeighbor(%)': 'Fraud Nodes with Any Neighbor (%)',
    'FraudNodeswith\naFraudNeighbor(%)': 'Fraud Nodes with a Fraud Neighbor (%)',
    'Emaildomain': 'Email domain', 'Identifier,24hours': 'Identifier, 24 hours',
    '1.75×10−6': '1.75 × 10^-6', '4.26×10−8': '4.26 × 10^-8',
    '9.55×10−8': '9.55 × 10^-8', '1.98×10−6': '1.98 × 10^-6',
}
notes = {
    3: 'Isolated means no non-self neighbors. The denominator is all 590,540 transactions.',
    4: 'Unique undirected edges, excluding self-loops. Both percentages use all fraudulent transactions as denominator. The thesis describes this as retrospective label-based structural analysis, not label-based graph construction or optimization.',
    5: 'The preceding thesis text specifies ten runs using seeds 42–51. The saved results reproduce these standard deviations with ddof=1 (sample standard deviation).',
    6: 'Ten runs. False positives are measured at each run’s validation-F1-selected threshold; standard deviations are sample standard deviations.',
    7: 'Ten-run means. Printed GNN names omit the -main suffix used by the matching saved results.',
    8: 'Ten-run means. Reproducing the August GAT mean requires the seed-51 row recovered from Git.',
    9: 'The thesis calls these final-run results and ranks them by F1-score. Matching current baseline rows have seed 51.',
    10: 'The PDF footnote states LSTM and Transformer use 8,832 test transactions because nine sequence warm-up targets are excluded. Other models use 8,841 transactions.',
    11: 'One sampled-complement ablation row per dataset/model. The surrounding text claims average graph degree 20. The saved result rows do not record degree, so this claim does not resolve the IEEE-CIS degree-5 versus degree-20 artifact ambiguity.',
    12: 'Significance threshold α = 0.05. These are selected pairwise findings, not a complete matrix of comparisons.',
    13: 'Significance threshold α = 0.05. These are selected pairwise findings, not a complete matrix of comparisons.',
}
tables = []
for page, found in raw.items():
    for t in found:
        n = len(tables) + 1
        cells = [[replacements.get(c, c or '') for c in row] for row in t['cells']]
        for row in cells:
            for j, cell in enumerate(row):
                row[j] = cell.replace('sampledcomplement', 'sampled complement')
                if n == 13 and j == 0:
                    row[j] = row[j].replace('vs', ' vs ')
        if n == 11:
            dataset = ''
            for row in cells[1:]:
                dataset = row[0] or dataset
                row[0] = dataset
        tables.append({'number': f'4.{n}', 'title': titles[n-1], 'pdf_page': int(page),
                       'printed_page': int(page)-15, 'headers': cells[0], 'rows': cells[1:],
                       'notes': notes.get(n, ''), 'preview': f'previews/table_4_{n}.png'})
assert len(tables) == 13
source = Path('C:/Users/user/Desktop/AAI_Thesis.pdf')
(OUT / 'final_tables.json').write_text(json.dumps({'source_pdf': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'tables': tables}, ensure_ascii=False, indent=2), encoding='utf-8')
text = '# Final results tables extracted from AAI_Thesis.pdf\n\n'
text += 'All 13 empirical-results tables in Chapter 4, transcribed from the supplied final PDF and visually checked against rendered table images. Table 2.1 is a literature comparison and is outside this results extraction. Values and row order are preserved as printed, including discrepancies. Only spacing, exponent notation and merged dataset labels are normalized. Source: `C:/Users/user/Desktop/AAI_Thesis.pdf`.\n\n'
text += 'PDF page numbers count from the first file page. Printed thesis page numbers are also provided. [Experiment-source matching and discrepancies](TABLE_PROVENANCE.md) is separate from this faithful extraction.\n\n'
for t in tables:
    text += f"## Table {t['number']}: {t['title']}\n\nPrinted page **{t['printed_page']}**; PDF page **{t['pdf_page']}**.\n\n"
    text += '| ' + ' | '.join(t['headers']) + ' |\n|' + '---|'*len(t['headers']) + '\n'
    text += ''.join('| ' + ' | '.join(row) + ' |\n' for row in t['rows'])
    if t['notes']:
        text += '\n**Context / audit note:** ' + t['notes'] + '\n'
    text += '\n'
(OUT / 'FINAL_TABLES.md').write_text(text, encoding='utf-8')

fields = ['recall', 'precision', 'f1_score', 'pr_auc', 'roc_auc']
cm = ['true_negative', 'false_positive', 'false_negative', 'true_positive']
checks = []

def refs(rs):
    return [{'source': r['source'], 'line': r['source_line'], 'model': r['model'],
             'seed': r.get('seed'), 'timestamp': r['timestamp'], 'threshold': r['threshold'],
             'generation': r['generation'], 'checkpoint_path_from_code': r['checkpoint_path_from_code'],
             'graph_path': r['graph_path'], 'test_ids_reconstructed_path': r['test_ids_reconstructed_path']} for r in rs]

for n in [5, 6]:
    for row in tables[n-1]['rows']:
        tag = {'Main':'main', 'Card':'card', 'Address':'address', 'Email domain':'email_domain', 'Identifier, 24 h':'identifier_time_window-24h'}[row[1]]
        rs = [r for r in runs if r['generation']=='august_graph_constructions' and r['model']==row[0]+'-'+tag]
        metrics = ['pr_auc', 'f1_score', 'recall'] if n==5 else ['false_positive']
        values = []
        for metric in metrics:
            a=np.array([float(r[metric]) for r in rs])
            values.append(f'{a.mean():.4f}±{a.std(ddof=1):.4f}' if n==5 else f'{a.mean():,.1f}±{a.std(ddof=1):,.1f}')
        assert values == row[2:], (n, row, values)
        checks.append({'table':f'4.{n}', 'row':row[0]+' '+row[1], 'match':True, 'sources':refs(rs)})
for n in [7,8]:
    for row in tables[n-1]['rows']:
        groups=collections.defaultdict(list)
        for r in runs:
            candidates=[row[0],row[0]+'-main'] if n==7 else ['Elliptic-'+row[0]]
            if r['model'] in candidates:
                groups[(r['generation'],r['model'])].append(r)
        matches=[]
        for (gen,model),rs in groups.items():
            values=[f"{np.mean([float(r[f]) for r in rs]):.4f}" for f in fields]
            if len(rs)==10 and values==row[1:]:
                matches.append({'generation':gen,'model':model,'sources':refs(rs)})
        assert matches, (n,row)
        checks.append({'table':f'4.{n}','row':row[0],'matching_batches':matches})
for n in [9,10,11]:
    for row in tables[n-1]['rows']:
        if n==11:
            names=[('Elliptic-' if row[0]=='Elliptic' else '')+row[1].replace('-sampled complement','-sampled_complement')]
            values=row[2:];cols=fields
        else:
            names=[row[0],row[0]+'-main'] if n==9 else ['Elliptic-'+row[0]]
            values=row[1:];cols=fields+cm
        matches=[r for r in runs if r['model'] in names and [f'{float(r[f]):.4f}' if f in fields else f'{int(r[f]):,}' for f in cols]==values]
        assert matches, (n,row)
        checks.append({'table':f'4.{n}','row':row[0] if n!=11 else row[0]+' '+row[1], 'matching_rows':refs(matches)})

def exact_wilcoxon(x,y):
    d=np.array(x)-np.array(y)
    assert np.all(d!=0) and len(set(abs(d)))==len(d), 'Exact no-tie calculation not applicable'
    ranks=np.argsort(np.argsort(abs(d)))+1
    w=min(ranks[d>0].sum(),ranks[d<0].sum())
    # Enumerate all 2^10 sign assignments under the paired null.
    sums=[sum(i+1 for i,b in enumerate(bits) if b) for bits in itertools.product([0,1],repeat=len(d))]
    return min(1,2*sum(v<=w for v in sums)/len(sums))

for n in [12,13]:
    for row in tables[n-1]['rows']:
        a,b=row[0].replace(' ','').split('vs');metric={'PR-AUC':'pr_auc','F1-score':'f1_score'}[row[1]]
        alternatives=[]
        for gen in (['june_original_ten_seed','august_v2','august_main'] if n==12 else ['june_original_ten_seed','august_v2']):
            rs=[]
            for model in [a,b]:
                g='august_graph_constructions' if gen=='august_main' and model in ['GCN','GAT','GraphSAGE'] else 'august_v2' if gen=='august_main' else gen
                name=('Elliptic-' if n==13 else '')+model+('-main' if g=='august_graph_constructions' else '')
                rs.append(sorted([r for r in runs if r['generation']==g and r['model']==name],key=lambda r:int(r['seed'])))
            assert all(len(z)==10 for z in rs)
            x,y=[[float(r[metric]) for r in group] for group in rs]
            p=exact_wilcoxon(x,y)
            higher=a if np.mean(x)>np.mean(y) else b
            alternatives.append({'generation':gen,'p_value':p,'higher_mean_model':higher,'matches':f'{p:.4f}'==row[3] and higher==row[2] and ('Yes' if p<.05 else 'No')==row[4], 'sources':refs(rs[0]+rs[1])})
        assert any(a['matches'] for a in alternatives),(n,row)
        checks.append({'table':f'4.{n}','row':row[0]+' '+row[1],'recomputed_alternatives':alternatives})
(OUT/'table_matches.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(f'Extracted {len(tables)} tables; matched {len(checks)} performance/statistical rows. All assertions passed.')
