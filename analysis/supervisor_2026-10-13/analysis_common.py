import csv
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SEEDS = list(range(42, 52))
METRICS = ['pr_auc', 'f1_score', 'recall', 'precision', 'roc_auc', 'false_positive']
CM = ['true_negative', 'false_positive', 'false_negative', 'true_positive']
CONSTRUCTIONS = ['main', 'card', 'address', 'email_domain', 'identifier_time_window-24h']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(name, obj):
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))

def write_csv(name, rows, fields=None):
    rows=list(rows)
    if fields is None:
        fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def source_rows(path):
    with (ROOT/path).open(newline='', encoding='utf-8-sig') as f:
        return [dict(r, source=path, source_line=i) for i,r in enumerate(csv.DictReader(f),2)]

def graph_path(dataset, construction):
    if dataset=='Elliptic':
        return 'results/elliptic_graph.pt'
    tag=construction.replace('-24h','_24h')
    return 'results/ieee_cis_graph_trainfit_'+tag+('' if tag.endswith('24h') else '_max1000')+'.pt'

def run_id(row):
    return row['dataset']+'__'+row['model']+'__seed'+str(row['seed'])
