"""Read-only provenance inspection; writes derived audit files only."""
import collections
import csv
import hashlib
import io
import json
import pathlib
import pickle
import subprocess
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit'


class SavedObject:
    def __setstate__(self, state):
        self.__dict__.update(state)


class TensorRef:
    def __init__(self, storage, offset, shape, stride, *args):
        self.storage, self.offset, self.shape, self.stride = storage, offset, shape, stride


class MetadataUnpickler(pickle.Unpickler):
    """Allow only inert stand-ins, never import or execute pickle globals."""
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return TensorRef
        if module == 'torch' and name.endswith('Storage'):
            return name
        if module.startswith('torch_geometric.') and name in ('Data', 'GlobalStorage', 'DataEdgeAttr', 'DataTensorAttr'):
            return SavedObject
        raise ValueError(f'Unsupported pickle global: {module}.{name}')

    def persistent_load(self, pid):
        return pid


def inspect_pt(path):
    with zipfile.ZipFile(path) as z:
        entry = next(n for n in z.namelist() if n.endswith('/data.pkl'))
        prefix = entry[:-8]
        obj = MetadataUnpickler(io.BytesIO(z.read(entry))).load()
        if isinstance(obj, dict):
            return {'kind': 'state_dict', 'tensor_count': len(obj),
                    'shapes': {k: list(v.shape) for k, v in obj.items()}}
        mapping = obj._store._mapping
        info = {'kind': 'graph', 'attributes': {}, 'tensors': {}}
        for k, v in mapping.items():
            if isinstance(v, TensorRef):
                info['tensors'][k] = list(v.shape)
                if k.endswith('_mask'):
                    arr = np.frombuffer(z.read(prefix + 'data/' + v.storage[2]), dtype=np.bool_)
                    indices = np.flatnonzero(arr[v.offset:v.offset + v.shape[0]])
                    info[k] = {'count': len(indices), 'first': int(indices[0]), 'last': int(indices[-1]),
                               'index_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()}
            elif isinstance(v, (str, int, float, bool)):
                info['attributes'][k] = v
        return info


def main():
    graphs = {}
    checkpoints = {}
    for p in sorted((ROOT / 'results').glob('*.pt')):
        info = inspect_pt(p)
        info['bytes'] = p.stat().st_size
        info['mtime_local'] = pd.Timestamp(p.stat().st_mtime, unit='s', tz='UTC').tz_convert('Asia/Beirut').isoformat()
        if info['kind'] == 'state_dict':
            info['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            checkpoints[p.name] = info
        else:
            graphs[p.name] = info
    (OUT / 'artifact_inventory.json').write_text(json.dumps({'graphs': graphs, 'checkpoints': checkpoints}, indent=2))
    print('GRAPHS', json.dumps(graphs, indent=2))
    ieee = pd.read_csv(ROOT / 'data/ieee-cis/train_transaction.csv', usecols=['TransactionID', 'TransactionDT', 'isFraud'])
    sorted_ieee = ieee.sort_values(['TransactionDT', 'TransactionID'], kind='mergesort')
    start = int(len(ieee) * .9)
    splits = {}
    for name, frame in [('ieee_original_full', ieee.iloc[start:]), ('ieee_original_sequence', ieee.iloc[start+9:]), ('ieee_august_all', sorted_ieee.iloc[start:])]:
        ids = frame.TransactionID.astype(int).tolist()
        (OUT / (name + '_test_ids.json')).write_text(json.dumps(ids))
        splits[name] = {'count': len(ids), 'first_ids': ids[:12], 'last_ids': ids[-3:], 'fraud_count': int(frame.isFraud.sum())}
    feats = pd.read_csv(ROOT / 'data/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None, usecols=[0, 1], names=['txId', 'time_step'])
    classes = pd.read_csv(ROOT / 'data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    elliptic = feats.merge(classes, on='txId', how='left').sort_values(['time_step', 'txId']).reset_index(drop=True)
    known = elliptic['class'].isin(['1', '2'])
    steps = sorted(elliptic.loc[known, 'time_step'].unique())
    test = elliptic[known & elliptic.time_step.isin(steps[int(len(steps) * (.70 + .15)):])]
    for name, frame in [('elliptic_full', test), ('elliptic_original_sequence', test.iloc[9:])]:
        ids = frame.txId.astype(int).tolist()
        (OUT / (name + '_test_ids.json')).write_text(json.dumps(ids))
        splits[name] = {'count': len(ids), 'first_ids': ids[:12], 'last_ids': ids[-3:], 'fraud_count': int((frame['class'] == '1').sum()), 'time_steps': sorted(map(int, frame.time_step.unique()))}
    splits['ieee_raw_order_equals_august_sort'] = bool(np.array_equal(ieee.TransactionID, sorted_ieee.TransactionID))
    indices = test.index.to_numpy(dtype='<i8')
    splits['elliptic_reconstructed_mask_sha256'] = hashlib.sha256(indices.tobytes()).hexdigest()
    (OUT / 'split_summary.json').write_text(json.dumps(splits, indent=2))
    print('SPLITS', json.dumps(splits, indent=2))
    all_rows = []
    for file in ['model_results_old.csv', 'model_results.csv', 'ieee_cis_graph_model_results.csv']:
        rows = list(csv.DictReader((ROOT / 'results' / file).open()))
        print(file, 'rows', len(rows))
        for line, row in enumerate(rows, 2):
            row = dict(row, source='results/' + file, source_line=line)
            all_rows.append(row)
    previous = subprocess.check_output(['git', 'show', '2be96ee:results/model_results.csv'], cwd=ROOT, text=True)
    recovered = next(r for r in csv.DictReader(io.StringIO(previous)) if r['model'] == 'Elliptic-GAT' and r['seed'] == '51' and r['timestamp'].startswith('2026-08'))
    (OUT / 'recovered_elliptic_gat_seed51.json').write_text(json.dumps(dict(recovered, source='2be96ee:results/model_results.csv', source_line=247, removed_by='5333c55'), indent=2))
    (OUT / 'saved_result_rows.json').write_text(json.dumps(all_rows, indent=2))


if __name__ == '__main__':
    main()
