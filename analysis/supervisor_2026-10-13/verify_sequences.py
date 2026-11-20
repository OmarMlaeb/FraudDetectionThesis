import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
sys.path.insert(0,str(ROOT/'src'))
import torch

def main():
    from ieee_cis.train_lstm import FraudSequenceDataset as LSTMDataset
    from ieee_cis.train_transformer import FraudSequenceDataset as TransformerDataset
    from elliptic.sequence import EllipticSequenceDataset
    out=[]
    for dataset in ['IEEE-CIS','Elliptic']:
        m=np.load(OUT/'node_mappings'/f'{dataset}.npz')
        g=torch.load(ROOT/graph_path(dataset,'main' if dataset=='IEEE-CIS' else 'native'),map_location='cpu',weights_only=False)
        for split in ['val_mask','test_mask']:
            mask=m[split];ids=m['node_ids'][mask];x=g.x[mask];y=g.y[mask]
            classes=[LSTMDataset(x.numpy(),y.numpy(),10),TransformerDataset(x.numpy(),y.numpy(),10)] if dataset=='IEEE-CIS' else [EllipticSequenceDataset(g.x.numpy(),g.y.numpy(),mask,10)]
            examples=[]
            for i in [0,1,8,9,10,len(ids)-1]:
                idx=np.maximum(np.arange(i-9,i+1),0)
                assert all(torch.equal(c[i][0],x[idx]) and c[i][1]==y[i] and len(c)==len(ids) for c in classes)
                examples.append({'target_index_in_split':i,'target_id':int(ids[i]),'input_ids':ids[idx].tolist(),'target_label':int(y[i])})
            out.append({'dataset':dataset,'split':split,'targets':len(ids),'sequence_length':10,'features':g.x.shape[1],'source_dataset_equivalence':True,'examples':examples})
        del g
    dump('sequence_input_checks.json',out)
    print('Selected August padding/windows verified against source Dataset classes.',flush=True)

if __name__=='__main__':main()
