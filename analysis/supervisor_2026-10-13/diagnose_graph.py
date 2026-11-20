"""Diagnostic reconstruction only; never changes or supplies inference graphs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
sys.path.insert(0,str(ROOT/'src'))
import argparse
import os
import pandas as pd
from ieee_cis.analyze_connectivity import load_aligned_transactions,GRAPH_EDGE_COLUMNS

def main(label):
    print('DIAGNOSTIC ONLY: reconstructing main edges with',label,pd.__version__,np.__version__,flush=True)
    df=load_aligned_transactions(ROOT/'data/ieee-cis/train_transaction.csv',ROOT/'data/ieee-cis/train_identity.csv')
    all_edges={'quicksort':set(),'mergesort':set()}; by_column=[]
    for column in GRAPH_EDGE_COLUMNS:
        edge_sets={'quicksort':set(),'mergesort':set()}
        for _,g in df.loc[df[column].notna()].groupby(column,sort=False):
            if not 2<=len(g)<=1000:continue
            for kind,edges in edge_sets.items():
                ids=g.sort_values('TransactionDT',kind=kind).index.to_numpy()
                edges.update((int(min(a,b)),int(max(a,b))) for a,b in zip(ids[:-1],ids[1:]) if a!=b)
        for kind,edges in edge_sets.items():all_edges[kind].update(edges)
        by_column.append({'column':column,'quicksort_edges':len(edge_sets['quicksort']),'mergesort_edges':len(edge_sets['mergesort']),'symmetric_difference':len(edge_sets['quicksort']^edge_sets['mergesort'])})
    folder=OUT/'graph_diagnostics';folder.mkdir(exist_ok=True)
    summary={'label':label,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'NPY_DISABLE_CPU_FEATURES':os.environ.get('NPY_DISABLE_CPU_FEATURES'),'by_column':by_column,'not_an_experiment_graph':True}
    for kind,edges in all_edges.items():
        pairs=np.array(sorted(edges),dtype=np.int64)
        txpairs=df.TransactionID.to_numpy(dtype=np.int64)[pairs]
        f=folder/f'diagnostic_{label}_{kind}.npz'
        np.savez_compressed(f,node_pairs=pairs,transaction_id_pairs=txpairs)
        summary[kind]={'edges':len(edges),'transaction_edge_sha256':hashlib.sha256(txpairs.tobytes()).hexdigest(),'artifact_sha256':sha(f),'path':str(f.relative_to(OUT))}
    dump(f'graph_diagnostics/reconstruction_{label}.json',summary)
    print(label,{k:len(v) for k,v in all_edges.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');main(p.parse_args().label)
