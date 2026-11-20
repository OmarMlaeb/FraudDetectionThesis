import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
import argparse
import itertools
from scipy.stats import rankdata, wilcoxon

def signed_rank(d):
    d=np.asarray(d,dtype=float); nonzero=d[d!=0]; n=len(nonzero)
    if n==0:
        return {'statistic':0.,'p_raw':1.,'n_nonzero':0,'n_zero':len(d),'tied_absolute_differences':False}
    ranks=rankdata(np.abs(nonzero),method='average')
    w=float(min(ranks[nonzero>0].sum(),ranks[nonzero<0].sum()))
    distribution=np.array([0.])
    for r in ranks:
        distribution=np.concatenate([distribution,distribution+r])
    p=min(1.,2*float(np.mean(distribution<=w)))
    ties=len(np.unique(np.abs(nonzero)))!=n
    if not ties and n==len(d):
        assert abs(p-wilcoxon(d,method='exact').pvalue)<1e-14
    return {'statistic':w,'p_raw':p,'n_nonzero':n,'n_zero':len(d)-n,'tied_absolute_differences':ties}

def specifications():
    families=[]
    for dataset in ['IEEE-CIS','Elliptic']:
        models=['MLP','LSTM','Transformer']+[m+('-main' if dataset=='IEEE-CIS' else '') for m in ['GCN','GraphSAGE','GAT']]
        families.append((dataset,'core',list(itertools.combinations(models,2))))
    comparisons=[]
    for m in ['GCN','GraphSAGE','GAT']:
        comparisons.extend(itertools.combinations([m+'-'+c for c in CONSTRUCTIONS],2))
    families.append(('IEEE-CIS','construction',comparisons))
    families.append(('IEEE-CIS','alternative_gnn_vs_mlp',[(m+'-'+c,'MLP') for m in ['GCN','GraphSAGE','GAT'] for c in CONSTRUCTIONS[1:]]))
    return families

def main(mode):
    selected=read('selected_runs.json')
    accepted={r['run_id'] for r in selected}
    if mode=='inference_verified':
        accepted=set()
        for p in (OUT/'checks').glob('*.json'):
            c=json.loads(p.read_text());
            if c.get('accepted'):accepted.add(c['run_id'])
    lookup={(r['dataset'],r['model'],r['seed']):r for r in selected}
    rows=[];definitions=[];paired=[]
    for dataset,family,comparisons in specifications():
        for metric in METRICS:
            fid=f'{dataset}__{family}__{metric}'
            definitions.append({'family_id':fid,'dataset':dataset,'metric':metric,'planned_hypotheses':len(comparisons),'comparisons':[list(x) for x in comparisons],'alpha':.05,'adjustment':'Holm step-down','alternative':'two-sided','zero_method':'wilcox','p_value_method':'exhaustive sign permutation of average absolute ranks','effect':'median(A-B)','higher_is_better':metric!='false_positive','required_seeds':SEEDS})
            family_rows=[]
            for a,b in comparisons:
                valid=[s for s in SEEDS if lookup[(dataset,a,s)]['run_id'] in accepted and lookup[(dataset,b,s)]['run_id'] in accepted]
                x=np.array([float(lookup[(dataset,a,s)][metric]) for s in valid]);y=np.array([float(lookup[(dataset,b,s)][metric]) for s in valid]);d=x-y
                complete=len(valid)==10
                result=signed_rank(d) if complete else {'statistic':None,'p_raw':None,'n_nonzero':int(np.count_nonzero(d)),'n_zero':int((d==0).sum()),'tied_absolute_differences':None}
                r={'analysis_mode':mode,'family_id':fid,'dataset':dataset,'metric':metric,'model_a':a,'model_b':b,'planned_family_size':len(comparisons),'n_pairs':len(valid),'paired_seeds':';'.join(map(str,valid)),'excluded_seeds':';'.join(map(str,sorted(set(SEEDS)-set(valid)))),'status':'complete' if complete else 'blocked_incomplete_verified_pairs','median_paired_difference_a_minus_b':float(np.median(d)) if len(d) else None,'mean_a':float(x.mean()) if len(d) else None,'mean_b':float(y.mean()) if len(d) else None,'mean_difference_a_minus_b':float(d.mean()) if len(d) else None,'higher_is_better':metric!='false_positive',**result}
                family_rows.append(r)
                paired.append({'family_id':fid,'a':a,'b':b,'pairs':[{'seed':s,'a':float(av),'b':float(bv),'difference_a_minus_b':float(dv)} for s,av,bv,dv in zip(valid,x,y,d)]})
            order=sorted(range(len(family_rows)),key=lambda i:family_rows[i]['p_raw'] if family_rows[i]['p_raw'] is not None else 1.)
            running=0.
            for rank,i in enumerate(order):
                p=family_rows[i]['p_raw'];running=max(running,min(1.,(len(order)-rank)*(p if p is not None else 1.)))
                family_rows[i]['p_holm']=running if p is not None else None
                family_rows[i]['significant_holm_0_05']=running<=.05 if p is not None else None
            rows.extend(family_rows)
    assert len(rows)==432 and len(definitions)==24
    write_csv(f'wilcoxon_holm_{mode}.csv',rows)
    dump(f'wilcoxon_holm_{mode}.json',rows)
    dump(f'paired_differences_{mode}.json',paired)
    dump('comparison_families.json',definitions)
    print(mode,len(rows),'comparisons;',sum(r['status']=='complete' for r in rows),'complete;',sum(bool(r['significant_holm_0_05']) for r in rows),'significant after Holm.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['source_verified','inference_verified'],default='source_verified');main(p.parse_args().mode)
