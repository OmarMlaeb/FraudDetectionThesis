import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
import torch
import pandas as pd

def canon(e):
    a=np.minimum(e[0],e[1]);b=np.maximum(e[0],e[1]);keep=a!=b
    return np.unique(a[keep]*590540+b[keep])

def main():
    folder=OUT/'graph_diagnostics';folder.mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'data/ieee-cis/train_transaction.csv',usecols=['TransactionID','TransactionDT','isFraud']).sort_values(['TransactionDT','TransactionID'],kind='mergesort')
    ids=df.TransactionID.to_numpy(dtype=np.int64);times=df.TransactionDT.to_numpy();labels=df.isFraud.to_numpy()
    candidates={};details=[]
    paths=['results/ieee_cis_graph.pt','results/ieee_cis_graph_trainfit.pt','results/ieee_cis_graph_trainfit_main_max1000.pt']
    for path in paths:
        g=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        e=g.edge_index.numpy();codes=canon(e);candidates[path]=codes
        pairs=np.column_stack([codes//len(ids),codes%len(ids)]);txpairs=ids[pairs]
        a,b=pairs.T
        details.append({'version':path,'kind':'saved_cache','edges':len(codes),'self_loop_columns':int((e[0]==e[1]).sum()),'file_sha256':sha(ROOT/path),'transaction_edge_sha256':hashlib.sha256(txpairs.tobytes()).hexdigest(),'fraud_fraud_edges':int(((labels[a]==1)&(labels[b]==1)).sum()),'fraud_legitimate_edges':int((labels[a]!=labels[b]).sum()),'legitimate_legitimate_edges':int(((labels[a]==0)&(labels[b]==0)).sum())})
        del g,e
    for path in sorted(folder.glob('diagnostic_*.npz')):
        pairs=np.load(path)['node_pairs'];codes=canon(pairs.T);candidates[path.name]=codes
        a,b=pairs.T
        details.append({'version':path.name,'kind':'diagnostic_only','edges':len(codes),'file_sha256':sha(path),'transaction_edge_sha256':hashlib.sha256(ids[pairs].tobytes()).hexdigest(),'fraud_fraud_edges':int(((labels[a]==1)&(labels[b]==1)).sum()),'fraud_legitimate_edges':int((labels[a]!=labels[b]).sum()),'legitimate_legitimate_edges':int(((labels[a]==0)&(labels[b]==0)).sum())})
    reference=candidates[paths[-1]];comparisons=[];diffs=[]
    for name,codes in candidates.items():
        removed=np.setdiff1d(reference,codes);added=np.setdiff1d(codes,reference)
        comparisons.append({'reference':paths[-1],'candidate':name,'only_in_saved_inference_graph':len(removed),'only_in_candidate':len(added),'same_edge_set':np.array_equal(reference,codes)})
        for role,values in [('only_in_saved_inference_graph',removed),('only_in_candidate',added)]:
            for code in values:
                a,b=divmod(int(code),len(ids))
                diffs.append({'candidate':name,'difference':role,'source_node':a,'target_node':b,'source_TransactionID':int(ids[a]),'target_TransactionID':int(ids[b]),'source_TransactionDT':int(times[a]),'target_TransactionDT':int(times[b]),'source_isFraud':int(labels[a]),'target_isFraud':int(labels[b]),'endpoints_same_time':bool(times[a]==times[b])})
    dump('graph_diagnostics/graph_versions.json',details);dump('graph_diagnostics/edge_set_comparisons.json',comparisons)
    write_csv('graph_diagnostics/transaction_id_edge_differences.csv',diffs)
    # Retain the selected cache's canonical transaction-ID edges as a small external
    # provenance artifact; this never replaces the full tensor inference cache.
    pairs=np.column_stack([reference//len(ids),reference%len(ids)])
    np.savez_compressed(folder/'saved_main_transaction_edges.npz',transaction_id_pairs=ids[pairs])
    print(json.dumps(comparisons,indent=2),flush=True)

if __name__=='__main__':main()
