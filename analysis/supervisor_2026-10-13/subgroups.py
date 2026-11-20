import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
import torch
from sklearn.metrics import average_precision_score,roc_auc_score

def calculate(y,p,t):
    n=len(y);pos=int(np.sum(y));neg=n-pos;pred=p>=t
    tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=pos-tp;tn=neg-fp
    undefined=[]
    def ratio(a,b,name):
        if b==0:undefined.append(name);return None
        return a/b
    result={'n':n,'positive':pos,'negative':neg,'fraud_rate':ratio(pos,n,'fraud_rate'),
            'true_negative':tn,'false_positive':fp,'false_negative':fn,'true_positive':tp,
            'recall':ratio(tp,pos,'recall'),'precision':ratio(tp,tp+fp,'precision'),
            'f1_score':ratio(2*tp,2*tp+fp+fn,'f1_score')}
    if pos:
        result['pr_auc']=float(average_precision_score(y,p))
    else:
        result['pr_auc']=None;undefined.append('pr_auc_no_positive_labels')
    if pos and neg:
        result['roc_auc']=float(roc_auc_score(y,p))
    else:
        result['roc_auc']=None;undefined.append('roc_auc_single_class_or_empty')
    result['group_status']='empty' if not n else 'no_positive_labels' if not pos else 'no_negative_labels' if not neg else 'two_classes'
    result['undefined_metrics']=';'.join(undefined)
    return result

def main():
    selected=read('selected_runs.json'); lookup={(r['dataset'],r['model'],r['seed']):r for r in selected}
    checks={p.stem:json.loads(p.read_text()) for p in (OUT/'checks').glob('*.json')}
    counts=[];per_seed=[];reconciliation=[]
    (OUT/'subgroup_ids').mkdir(exist_ok=True)
    hashes={r['path']:r['sha256'] for r in read('artifact_hashes_before.json')}
    for dataset,constructions in [('IEEE-CIS',CONSTRUCTIONS),('Elliptic',['native'])]:
        mapping=np.load(OUT/'node_mappings'/f'{dataset}.npz');test_mask=mapping['test_mask'];ids=mapping['node_ids'][test_mask];y=mapping['labels'][test_mask]
        for construction in constructions:
            path=graph_path(dataset,construction)
            graph=torch.load(ROOT/path,map_location='cpu',weights_only=False)
            e=graph.edge_index.numpy();e=e[:,e[0]!=e[1]]
            connected=np.zeros(len(graph.y),dtype=bool);connected[e.ravel()]=True;connected=connected[test_mask]
            for name,mask in [('connected',connected),('isolated',~connected)]:
                np.savez_compressed(OUT/'subgroup_ids'/f'{dataset}__{construction}__{name}.npz',transaction_ids=ids[mask],labels=y[mask])
                counts.append({'dataset':dataset,'construction':construction,'group':name,'graph_path':path,'graph_sha256':hashes[path],'n':int(mask.sum()),'positive':int(y[mask].sum()),'negative':int(mask.sum()-y[mask].sum()),'fraud_rate':int(y[mask].sum())/int(mask.sum()) if mask.any() else None,'id_sha256':hashlib.sha256(ids[mask].tobytes()).hexdigest(),'definition':'degree in full inference graph ignoring self loops'})
            del graph,e
            for architecture in ['GCN','GraphSAGE','GAT','MLP']:
                model=architecture+('-'+construction if dataset=='IEEE-CIS' and architecture!='MLP' else '')
                for seed in SEEDS:
                    r=lookup[(dataset,model,seed)];check=checks.get(r['run_id'],{})
                    base={'dataset':dataset,'construction':construction,'architecture':architecture,'source_model':model,'seed':seed,'run_id':r['run_id'],'saved_threshold':r['threshold'],'checkpoint_accepted':check.get('accepted',False)}
                    if not check.get('accepted'):
                        for name,mask in [('connected',connected),('isolated',~connected)]:
                            per_seed.append(dict(base,group=name,status='blocked_checkpoint_not_verified',n=int(mask.sum()),positive=int(y[mask].sum()),negative=int(mask.sum()-y[mask].sum()),fraud_rate=int(y[mask].sum())/int(mask.sum()) if mask.any() else None))
                        continue
                    pred=np.load(OUT/check['prediction_file'])
                    assert np.array_equal(ids,pred['test_ids']) and np.array_equal(y,pred['test_y'])
                    group_results=[]
                    for name,mask in [('connected',connected),('isolated',~connected)]:
                        vals=calculate(y[mask],pred['test_probs'][mask],float(r['threshold']))
                        group_results.append(vals)
                        per_seed.append(dict(base,group=name,status='verified',**vals))
                    sums={k:sum(v[k] for v in group_results) for k in CM}
                    whole=check['whole_test_metrics']
                    exact=all(sums[k]==whole[k]==int(r[k]) for k in CM)
                    assert exact,(r['run_id'],sums,whole)
                    reconciliation.append(dict(base,counts_sum_exact=exact,**sums))
    summaries=[]
    numeric=['n','positive','negative','fraud_rate','pr_auc','f1_score','recall','precision','roc_auc',*CM]
    for key in sorted({(r['dataset'],r['construction'],r['architecture'],r['group']) for r in per_seed}):
        rs=[r for r in per_seed if (r['dataset'],r['construction'],r['architecture'],r['group'])==key]
        verified=[r for r in rs if r['status']=='verified']
        summary=dict(zip(['dataset','construction','architecture','group'],key),expected_seeds=10,verified_seeds=len(verified),missing_seeds=';'.join(str(r['seed']) for r in rs if r['status']!='verified'))
        for m in numeric:
            values=[float(r[m]) for r in verified if r.get(m) is not None]
            summary[m+'_n_defined']=len(values)
            summary[m+'_mean']=float(np.mean(values)) if values else None
            summary[m+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summaries.append(summary)
    write_csv('subgroup_counts.csv',counts);write_csv('subgroup_per_seed.csv',per_seed);write_csv('subgroup_summary.csv',summaries);write_csv('subgroup_confusion_reconciliation.csv',reconciliation)
    dump('subgroup_counts.json',counts);dump('subgroup_per_seed.json',per_seed);dump('subgroup_summary.json',summaries)
    print('Groups',len(counts),'per-seed rows',len(per_seed),'verified',sum(r['status']=='verified' for r in per_seed),'reconciliations',len(reconciliation),flush=True)

if __name__=='__main__':main()
