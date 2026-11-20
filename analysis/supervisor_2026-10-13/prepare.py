import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
import io
import subprocess
import platform

def main():
    selected=[]
    for r in source_rows('results/model_results.csv'):
        if not r['timestamp'].startswith('2026-08-30'):
            continue
        if r['model'].startswith('Elliptic-') or r['model'] in ('MLP','LSTM','Transformer'):
            selected.append(r)
    historical=subprocess.check_output(['git','show','2be96ee:results/model_results.csv'],cwd=ROOT,text=True)
    recovered=next(dict(r,source='git:2be96ee:results/model_results.csv',source_line=i) for i,r in enumerate(csv.DictReader(io.StringIO(historical)),2) if r['timestamp'].startswith('2026-08-30') and r['model']=='Elliptic-GAT' and r['seed']=='51')
    matches=[r for r in selected if r['model']=='Elliptic-GAT' and r['seed']=='51']
    if not matches:
        selected.append(recovered)
    else:
        assert all(matches[0][k]==v for k,v in recovered.items() if k not in ['source','source_line'])
    dump('recovered_result_provenance.json',{'row':recovered,'git_commit':subprocess.check_output(['git','rev-parse','2be96ee'],cwd=ROOT,text=True).strip(),'removed_by':'5333c55','historical_csv_sha256':hashlib.sha256(historical.encode()).hexdigest(),'git_diff':subprocess.check_output(['git','diff','2be96ee','5333c55','--','results/model_results.csv'],cwd=ROOT,text=True)})
    selected.extend(source_rows('results/ieee_cis_graph_model_results.csv'))
    inventory=[]
    for r in selected:
        r['dataset']='Elliptic' if r['model'].startswith('Elliptic-') else 'IEEE-CIS'
        r['original_model_name']=r['model']
        r['model']=r['model'].removeprefix('Elliptic-')
        r['seed']=int(r['seed']); assert r['seed'] in SEEDS
        r['architecture']=r['model'].split('-')[0]
        r['construction']=r['model'].split('-',1)[1] if '-' in r['model'] else ('native' if r['dataset']=='Elliptic' else 'none')
        r['run_id']=run_id(r)
        key={'GraphSAGE':'sage'}.get(r['architecture'],r['architecture'].lower())
        prefix='elliptic_' if r['dataset']=='Elliptic' else ''
        tag='_'+r['construction'].replace('-24h','_24h') if r['dataset']=='IEEE-CIS' and r['construction']!='none' else ''
        r['checkpoint_path']=f"results/best_{prefix}{key}{tag}_seed{r['seed']}_model.pt"
        r['graph_path']=graph_path(r['dataset'],r['construction']) if r['architecture'] in ['GCN','GraphSAGE','GAT'] else None
        r['source_snapshot']='5333c55' if tag else '2be96ee'
        r['checkpoint_exists']=(ROOT/r['checkpoint_path']).exists()
        r['checkpoint_sha256']=sha(ROOT/r['checkpoint_path']) if r['checkpoint_exists'] else None
    assert len(selected)==240 and len({r['run_id'] for r in selected})==240
    for key in {(r['dataset'],r['model']) for r in selected}:
        assert sorted(r['seed'] for r in selected if (r['dataset'],r['model'])==key)==SEEDS
    dump('selected_runs.json',selected)
    write_csv('selected_per_seed_results.csv',selected)
    paths={r['checkpoint_path'] for r in selected}|{r['graph_path'] for r in selected if r['graph_path']}
    paths|={'data/ieee-cis/train_transaction.csv','data/ieee-cis/train_identity.csv','data/elliptic_bitcoin_dataset/elliptic_txs_features.csv','data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv','data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv','results/model_results.csv','results/ieee_cis_graph_model_results.csv','results/ieee_cis_graph.pt','results/ieee_cis_graph_trainfit.pt'}
    paths|={str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src').rglob('*.py')}
    paths|={str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'results').glob('ieee_cis_graph_edges_*.csv')}
    for path in sorted(paths):
        p=ROOT/path
        inventory.append({'path':path,'absolute_path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None,'sha256':sha(p) if p.exists() else None})
    dump('artifact_hashes_before.json',inventory)
    snapshots=OUT/'source_snapshots'
    files=['src/common/metrics.py','src/common/early_stopping.py','src/ieee_cis/models.py','src/ieee_cis/gnn_models.py','src/ieee_cis/preprocessing.py','src/ieee_cis/graph_data.py','src/ieee_cis/train_mlp.py','src/ieee_cis/train_gnn.py','src/ieee_cis/train_lstm.py','src/ieee_cis/train_transformer.py','src/elliptic/data.py','src/elliptic/sequence.py','src/elliptic/models.py','src/elliptic/gnn_models.py','src/elliptic/train_gnn.py','src/elliptic/train_mlp.py','src/elliptic/train_lstm.py','src/elliptic/train_transformer.py']
    for commit in ['2be96ee','5333c55']:
        for file in files:
            p=snapshots/commit/file;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(subprocess.check_output(['git','show',commit+':'+file],cwd=ROOT))
    import torch,torch_geometric,pandas,scipy,sklearn
    dump('environment.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,'numpy':np.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'portable_runtime_source':'https://www.python.org/ftp/python/3.10.0/python-3.10.0-embed-amd64.zip','portable_runtime_sha256':sha(OUT/'runtime/python-3.10.0-embed-amd64.zip')})
    print('Selected 240 complete August runs, recovered GAT seed 51, hashed input artifacts.',flush=True)

if __name__=='__main__':main()
