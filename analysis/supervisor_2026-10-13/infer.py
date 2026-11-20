import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
sys.path.insert(0,str(ROOT/'src'))
import argparse
import gc
import time
import traceback
import torch
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,confusion_matrix
from common.metrics import find_best_threshold

def metrics(y,p,t):
    pred=p>=t
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(zip(CM,map(int,[tn,fp,fn,tp])),recall=float(recall_score(y,pred,zero_division=0)),precision=float(precision_score(y,pred,zero_division=0)),f1_score=float(f1_score(y,pred,zero_division=0)),pr_auc=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)))

def infer_split(model,x,sequence=False):
    if not sequence:
        return torch.sigmoid(model(x.to('cuda'))).cpu().numpy()
    # Exact corrected sequence: split-local windows ending at each target, repeat
    # the first split observation on the left when fewer than 10 are available.
    output=[]
    for start in range(0,len(x),512):
        end=min(start+512,len(x))
        idx=(torch.arange(start,end)[:,None]-torch.arange(9,-1,-1)[None,:]).clamp_min(0)
        output.append(torch.sigmoid(model(x[idx].to('cuda'))).cpu().numpy())
    return np.concatenate(output)

def main(dataset,kind):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[r for r in read('selected_runs.json') if r['dataset']==dataset and ((r['architecture'] in ['LSTM','Transformer'])==(kind=='sequence'))]
    for folder in ['checks','predictions']:(OUT/folder).mkdir(exist_ok=True)
    feature_checks={x['graph_path']:x for x in read(f'feature_checks_{dataset}.json')}
    mapping=np.load(OUT/'node_mappings'/f'{dataset}.npz')
    ids=mapping['node_ids']; y=mapping['labels']; val_mask=mapping['val_mask'];test_mask=mapping['test_mask']
    graph=None;loaded_path=None
    for r in rows:
        rid=r['run_id'];check_file=OUT/'checks'/f'{rid}.json'
        if check_file.exists():
            print('Already checked',rid,flush=True);continue
        started=time.time()
        check={'run_id':rid,'checkpoint_path':r['checkpoint_path'],'checkpoint_sha256':r['checkpoint_sha256'],'selected_result_source':r['source'],'source_line':r['source_line'],'seed':r['seed'],'saved_threshold':r['threshold'],'accepted':False}
        try:
            torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
            path=r['graph_path'] or graph_path(dataset,'main' if dataset=='IEEE-CIS' else 'native')
            check['feature_graph_path']=path
            assert feature_checks[path]['accepted'],'Reconstructed features/IDs/masks do not match cache'
            assert sha(ROOT/r['checkpoint_path'])==r['checkpoint_sha256'],'Checkpoint hash changed'
            if loaded_path!=path:
                del graph;gc.collect();torch.cuda.empty_cache()
                graph=torch.load(ROOT/path,map_location='cpu',weights_only=False)
                loaded_path=path
            input_dim=graph.x.shape[1]
            gnn=r['architecture'] in ['GCN','GraphSAGE','GAT']
            if gnn:
                from ieee_cis.gnn_models import create_gnn_model
                model=create_gnn_model({'GraphSAGE':'sage'}.get(r['architecture'],r['architecture'].lower()),input_dim).to('cuda')
            else:
                from ieee_cis.models import MLPFraudDetector,LSTMFraudDetector,TransformerFraudDetector
                model={'MLP':lambda:MLPFraudDetector(input_dim),'LSTM':lambda:LSTMFraudDetector(input_dim),'Transformer':lambda:TransformerFraudDetector(input_dim,10)}[r['architecture']]().to('cuda')
            state=torch.load(ROOT/r['checkpoint_path'],map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True);del state
            model.eval()
            with torch.no_grad():
                if gnn:
                    gx=graph.x.to('cuda');ge=graph.edge_index.to('cuda')
                    logits=model(gx,ge)
                    val_probs=torch.sigmoid(logits[torch.tensor(val_mask,device='cuda')]).cpu().numpy()
                    test_probs=torch.sigmoid(logits[torch.tensor(test_mask,device='cuda')]).cpu().numpy()
                    del gx,ge,logits
                else:
                    val_probs=infer_split(model,graph.x[val_mask],kind=='sequence')
                    test_probs=infer_split(model,graph.x[test_mask],kind=='sequence')
            assert np.isfinite(val_probs).all() and np.isfinite(test_probs).all()
            t=float(r['threshold']);observed=metrics(y[test_mask],test_probs,t)
            val_f1=float(f1_score(y[val_mask],val_probs>=t,zero_division=0))
            if r['threshold_strategy']=='f1_max_alert_rate_0.20':
                from ieee_cis.train_transformer import build_threshold_candidates
                selected_t,selected_vf1=find_best_threshold(y[val_mask],val_probs,thresholds=build_threshold_candidates(val_probs),max_positive_rate=.2)
            else:
                selected_t,selected_vf1=find_best_threshold(y[val_mask],val_probs,strategy=r['threshold_strategy'])
            deltas={k:observed[k]-float(r[k]) for k in observed}
            count_match=all(deltas[k]==0 for k in CM)
            discrete_match=all(abs(deltas[k])<=1e-10 for k in ['recall','precision','f1_score'])
            rank_match=all(abs(deltas[k])<=2e-5 for k in ['pr_auc','roc_auc'])
            val_match=abs(val_f1-float(r['validation_f1_at_threshold']))<=1e-10
            threshold_match=abs(selected_t-t)<=1e-6
            np.savez_compressed(OUT/'predictions'/f'{rid}.npz',test_ids=ids[test_mask],test_y=y[test_mask],test_probs=test_probs,val_ids=ids[val_mask],val_y=y[val_mask],val_probs=val_probs,threshold=np.array(t))
            check.update(whole_test_metrics=observed,differences_from_saved=deltas,counts_exact=count_match,classification_metrics_match=discrete_match,ranking_metrics_within_tolerance=rank_match,validation_f1_at_saved_threshold=val_f1,validation_f1_matches=val_match,reselected_validation_threshold=float(selected_t),threshold_matches=threshold_match,test_count=int(test_mask.sum()),val_count=int(val_mask.sum()),test_ids_sha256=hashlib.sha256(ids[test_mask].tobytes()).hexdigest(),accepted=bool(count_match and discrete_match and rank_match and val_match and threshold_match),prediction_file=f'predictions/{rid}.npz')
            check['prediction_sha256']=sha(OUT/check['prediction_file'])
            del model,val_probs,test_probs;gc.collect();torch.cuda.empty_cache()
        except Exception as e:
            check['error']=str(e);check['traceback']=traceback.format_exc()
            gc.collect();torch.cuda.empty_cache()
        check['elapsed_seconds']=round(time.time()-started,3)
        dump(f'checks/{rid}.json',check)
        print(rid,'PASS' if check['accepted'] else 'MISMATCH',check.get('differences_from_saved',check.get('error')),f"{check['elapsed_seconds']}s",flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['IEEE-CIS','Elliptic']);p.add_argument('--kind',choices=['required','sequence'],default='required');a=p.parse_args();main(a.dataset,a.kind)
