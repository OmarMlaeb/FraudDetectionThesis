import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
sys.path.insert(0,str(ROOT/'src'))
import gc
import torch

def fingerprint(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a))).hexdigest()

def check_graph(path,X,y,ids,masks):
    g=torch.load(ROOT/path,map_location='cpu',weights_only=False)
    return {'graph_path':path,'features_equal':np.array_equal(g.x.numpy(),X),'labels_equal':np.array_equal(g.y.numpy(),y),
            'shape':list(g.x.shape),'feature_tensor_sha256':fingerprint(g.x.numpy()),'label_tensor_sha256':fingerprint(g.y.numpy()),
            'node_ids_sha256':fingerprint(ids),'test_ids_sha256':fingerprint(ids[masks['test_mask']]),
            **{k+'_equal':np.array_equal(g[k].numpy(),v) for k,v in masks.items()}}

def main(dataset):
    (OUT/'node_mappings').mkdir(exist_ok=True)
    if dataset=='IEEE-CIS':
        from ieee_cis.preprocessing import load_ieee_cis,sort_ieee_cis_temporally,fit_ieee_cis_preprocessor,transform_ieee_cis_features
        print('Loading IEEE-CIS raw data and reconstructing TRAIN-FIT features only (no graph rebuild).',flush=True)
        df=sort_ieee_cis_temporally(load_ieee_cis(str(ROOT/'data/ieee-cis/train_transaction.csv'),str(ROOT/'data/ieee-cis/train_identity.csv'))).reset_index(drop=True)
        n=len(df);train_end=int(n*.8);val_end=int(n*.9)
        p=fit_ieee_cis_preprocessor(df.iloc[:train_end])
        saved={'feature_columns':p['feature_columns'],'categorical_columns':p['categorical_columns'],'numerical_columns':p['numerical_columns'],'category_maps':p['category_maps'],'medians':p['medians'].to_dict(),'scaler_mean':p['scaler'].mean_.tolist(),'scaler_scale':p['scaler'].scale_.tolist(),'scaler_var':p['scaler'].var_.tolist(),'fit_rows':train_end,'status':'analysis reconstruction; verified against saved graph features, not an original serialized preprocessor'}
        dump('reconstructed_ieee_preprocessor.json',saved)
        ids=df.TransactionID.to_numpy(dtype=np.int64);X,y=transform_ieee_cis_features(df,p)
        del df,p;gc.collect()
        masks={'train_mask':np.arange(n)<train_end,'val_mask':(np.arange(n)>=train_end)&(np.arange(n)<val_end),'test_mask':np.arange(n)>=val_end}
        checks=[check_graph(graph_path(dataset,c),X,y,ids,masks) for c in CONSTRUCTIONS]
    else:
        from elliptic.data import load_elliptic_tabular_data,build_elliptic_edge_index
        # Source paths are relative to repository cwd.
        tab=load_elliptic_tabular_data();X,y,ids=tab['X'],tab['y'],tab['df'].txId.to_numpy(dtype=np.int64)
        masks={k:tab[k] for k in ['train_mask','val_mask','test_mask']}
        checks=[check_graph(graph_path(dataset,'native'),X,y,ids,masks)]
        original=torch.load(ROOT/graph_path(dataset,'native'),map_location='cpu',weights_only=False)
        rebuilt_edges=build_elliptic_edge_index(tab['df'])
        checks[0]['raw_edgelist_matches_saved_ordered_edges']=torch.equal(original.edge_index,rebuilt_edges)
        del tab,original,rebuilt_edges
    np.savez_compressed(OUT/'node_mappings'/f'{dataset}.npz',node_ids=ids,labels=y,**masks)
    checks=[dict(c,accepted=all(v for k,v in c.items() if k.endswith('_equal') or k=='raw_edgelist_matches_saved_ordered_edges')) for c in checks]
    dump(f'feature_checks_{dataset}.json',checks)
    print(dataset,[(c['graph_path'],c['accepted']) for c in checks],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['IEEE-CIS','Elliptic']);main(p.parse_args().dataset)
