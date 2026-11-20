"""Explicit training entry point. No work occurs on import."""
import gc
import os
import random
import traceback
from datetime import datetime, timezone
import numpy as np
from common import *
from plan import training_config
from graphs import read_graph_manifest, load_inputs


def model_factory(model, dataset, dimension):
    if model in GNN:
        return frozen(dataset+'_gnn').create_gnn_model(model, dimension)
    classes = frozen('elliptic_models')
    return {'mlp': lambda: classes.MLPFraudDetector(dimension),
            'lstm': lambda: classes.LSTMFraudDetector(dimension),
            'transformer': lambda: classes.TransformerFraudDetector(dimension, 10)}[model]()


def metric_values(y, p, threshold):
    m = frozen('metrics').evaluate_binary_classification(y, p, threshold)
    result = {k: float(m[v]) for k, v in dict(pr_auc='PR-AUC', roc_auc='ROC-AUC', f1_score='F1-score', recall='Recall', precision='Precision').items()}
    result.update(dict(zip(COUNTS, map(int, m['Confusion Matrix'].ravel()))))
    return result


def compatibility(row, graph, env, ref):
    # Git status is recorded but is not a compatibility key: new outputs change it.
    return dict(row=row, training=training_config(row['model']), graph_config=graph['config'],
                topology=graph['topology'], source_graph=graph['source_graph'],
                effective_graph_sha256=graph['effective_graph_sha256'],
                reference=ref, reference_lock_sha256=sha(HERE / 'reference_lock.json'),
                code_hashes=env['code_hashes'], packages=env['packages'], python=env['python'],
                gpu=env['gpu'], cuda=env['cuda'], cudnn=env['cudnn'])


def check_completed(status, expected=None):
    require(status.get('status') == 'complete', 'Run is incomplete')
    config_path = verify_artifact(status['config'])
    config = read(config_path)
    require(digest(config['compatibility']) == status['compatibility_sha256'], 'Stored configuration mismatch')
    if expected is not None:
        require(status['compatibility_sha256'] == digest(expected), 'Completed run configuration differs from this request; refusing reuse/overwrite.')
    for a in status['artifacts'].values():
        verify_artifact(a)
    return config


def train_one(row, settings, retry_failed=False):
    import torch
    graph = read_graph_manifest(row['graph_spec'])  # Never auto-generate.
    env = environment()
    require(torch.cuda.is_available(), 'Pinned experiment expects a CUDA GPU; no silent CPU fallback.')
    ref = reference(row)
    compat = compatibility(row, graph, env, ref)
    directory = OUT / 'runs' / row['family'] / row['run_id']
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / 'status.json'
    if status_path.exists():
        status = read(status_path)
        if status['status'] == 'complete':
            check_completed(status, compat)
            print('Verified complete; skipping:', row['run_id'], flush=True)
            return 'skipped'
        require(retry_failed, f'Incomplete run. Restart in a NEW attempt with: {rerun_command(row, settings)}')
    elif list(directory.glob('attempt_*')):
        require(retry_failed, f'Interrupted setup. Use: {rerun_command(row, settings)}')
    attempt = directory / f'attempt_{len(list(directory.glob("attempt_*")))+1:03d}'
    attempt.mkdir(exist_ok=False)
    config = dict(compatibility=compat, environment=env, created_utc=datetime.now(timezone.utc).isoformat(),
                  rerun_command=rerun_command(row, settings))
    atomic_json(attempt / 'config.json', config)
    status = dict(status='running', run_id=row['run_id'], attempt=str(attempt),
                  compatibility_sha256=digest(compat), config=artifact(attempt / 'config.json'),
                  rerun_command=config['rerun_command'])
    atomic_json(status_path, status)
    try:
        result, checkpoint = train_loop(row, graph, attempt)
        atomic_json(attempt / 'metrics.json', result)
        status.update(status='complete', artifacts={
            'checkpoint': artifact(checkpoint), 'predictions': artifact(attempt / 'predictions.npz'),
            'metrics': artifact(attempt / 'metrics.json'), 'log': artifact(attempt / 'training.jsonl')})
        atomic_json(attempt / 'completion.json', status)
        atomic_json(status_path, status)
        return 'complete'
    except BaseException as exc:
        trace = traceback.format_exc()
        atomic_bytes(attempt / 'failure.log', trace.encode())
        status.update(status='failed', error=repr(exc), failure_log=artifact(attempt / 'failure.log'))
        atomic_json(attempt / 'failure.json', status)
        atomic_json(status_path, status)
        print('FAILED:', row['run_id'], '\n', config['rerun_command'], flush=True)
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def train_loop(row, graph, attempt):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    cfg = training_config(row['model'])
    g, mapping = load_inputs(row['graph_spec'], graph)
    x, y = g.x.numpy(), g.y.numpy()
    masks = {s: getattr(g, s+'_mask').numpy() for s in ['train', 'val', 'test']}
    ids = mapping['node_ids']
    seed = row['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    model_name = row['model']
    is_gnn = model_name in GNN
    is_sequence = model_name in ['lstm', 'transformer']
    seq = frozen('elliptic_sequence') if is_sequence else None
    loaders = {}
    if is_sequence:
        for s in masks:
            ds = seq.EllipticSequenceDataset(x, y, masks[s], cfg['sequence_length'])
            require(len(ds) == int(masks[s].sum()), 'Sequence target population changed')
            loaders[s] = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
        train_targets = loaders['train'].dataset.y.numpy()
    elif not is_gnn:
        loaders['train'] = DataLoader(TensorDataset(torch.tensor(x[masks['train']]), torch.tensor(y[masks['train']])),
                                     batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
        # Preserve original MLP whole-split validation/test forward calls.
        val_tensor = torch.tensor(x[masks['val']]).to(device)
        test_tensor = torch.tensor(x[masks['test']]).to(device)
        train_targets = y[masks['train']]
    else:
        g = g.to(device)
        train_targets = g.y[g.train_mask]
    model = model_factory(model_name, row['dataset'], x.shape[1]).to(device)
    positive = train_targets.sum()
    negative = len(train_targets)-positive
    require(float(positive) > 0, 'No train positives')
    pos_weight = torch.tensor([negative/positive], dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    metric_module = frozen('metrics')

    def predict(split):
        model.eval()
        with torch.no_grad():
            if is_gnn:
                mask = getattr(g, split+'_mask')
                return g.y[mask].cpu().numpy(), torch.sigmoid(model(g.x, g.edge_index)[mask]).cpu().numpy()
            if is_sequence:
                return seq.predict_sequence_model(model, loaders[split], device)
            tensor = val_tensor if split == 'val' else test_tensor
            return y[masks[split]], torch.sigmoid(model(tensor)).cpu().numpy()

    best_score, best_epoch, stale, checkpoint = None, 0, 0, None
    stop_reason = 'epoch_limit'
    with (attempt / 'training.jsonl').open('x', encoding='utf-8') as log:
        def record(event):
            log.write(json.dumps(event, allow_nan=False)+'\n'); log.flush()
            if event['event'] == 'start':
                print(f"Using CUDA | Features: {event['dimension']} | Positive weight: {event['pos_weight']:.4f}", flush=True)
            elif event['event'] == 'epoch':
                print(f"Epoch [{event['epoch']}/{cfg['epochs']}] Loss: {event['loss']:.4f} "
                      f"Val PR-AUC: {event['validation_average_precision']:.4f} "
                      f"Best epoch: {event['best_epoch']}", flush=True)
            elif event['event'] == 'complete':
                metrics = event['metrics']
                print(f"\nFinal test results | Threshold: {event['threshold']:.2f}", flush=True)
                print(f"Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f} | "
                      f"F1: {metrics['f1_score']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}", flush=True)
                print(f"TN: {metrics['true_negative']} | FP: {metrics['false_positive']} | "
                      f"FN: {metrics['false_negative']} | TP: {metrics['true_positive']}", flush=True)
        record(dict(event='start', run_id=row['run_id'], dimension=x.shape[1], pos_weight=float(pos_weight.item())))
        for epoch in range(1, cfg['epochs']+1):
            model.train()
            total_loss = 0.0
            nonfinite = False
            if is_gnn:
                optimizer.zero_grad()
                logits = model(g.x, g.edge_index)
                loss = criterion(logits[g.train_mask], g.y[g.train_mask])
                require(bool(torch.isfinite(loss)), 'Nonfinite GNN training loss')
                loss.backward(); optimizer.step()
                total_loss = loss.item()
            else:
                for batch_x, batch_y in loaders['train']:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(batch_x), batch_y)
                    if not bool(torch.isfinite(loss)):
                        if model_name == 'transformer':
                            nonfinite = True
                            break
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    if cfg['gradient_clip'] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                    optimizer.step()
                    total_loss += loss.item()
            if nonfinite:
                stop_reason = 'transformer_nonfinite_loss_original_early_stop'
                break
            vy, vp = predict('val')
            if model_name == 'transformer' and not np.isfinite(vp).all():
                stop_reason = 'transformer_nonfinite_validation_original_early_stop'
                break
            validation = metric_module.evaluate_binary_classification(vy, vp)
            score = float(validation['PR-AUC'])
            require(np.isfinite(score), 'Nonfinite validation selection score')
            improved = best_score is None or score > best_score
            if improved:
                best_score, best_epoch, stale = score, epoch, 0
                checkpoint = attempt / f'best_epoch_{epoch:03d}.pt'
                require(not checkpoint.exists(), 'Refusing checkpoint overwrite')
                temporary = checkpoint.with_suffix('.tmp')
                torch.save(model.state_dict(), temporary)
                os.replace(temporary, checkpoint)
            else:
                stale += 1
            record(dict(event='epoch', epoch=epoch, loss=total_loss, validation_average_precision=score,
                        improved=improved, best_epoch=best_epoch, epochs_without_improvement=stale))
            if stale >= cfg['patience']:
                stop_reason = 'early_stopping_patience'
                break
        require(checkpoint is not None, 'No valid checkpoint selected')
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        vy, vp = predict('val')
        threshold, val_f1 = metric_module.find_best_threshold(vy, vp, strategy='f1')
        # Test is accessed for prediction only after checkpoint and threshold are fixed.
        ty, tp = predict('test')
        for split, targets, probs in [('val', vy, vp), ('test', ty, tp)]:
            require(np.array_equal(targets, y[masks[split]]) and len(probs) == int(masks[split].sum()), 'Prediction population mismatch')
            require(np.isfinite(probs).all(), 'Nonfinite final predictions')
        result = dict(**{k: row[k] for k in ['run_id', 'family', 'dataset', 'construction', 'model', 'seed', 'graph_seed', 'feature_set', 'reference_run_id']},
                      **metric_values(ty, tp, threshold), threshold=float(threshold), threshold_strategy='f1',
                      validation_f1_at_threshold=float(val_f1), best_epoch=best_epoch,
                      selection_metric='validation_average_precision', best_validation_score=best_score,
                      stop_reason=stop_reason, completed_epochs=epoch, test_count=len(ty), validation_count=len(vy),
                      checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                      source_graph_sha256=graph['source_graph']['sha256'], resulting_topology_sha256=graph['topology']['sha256'],
                      effective_graph_sha256=graph['effective_graph_sha256'],
                      input_dimension=x.shape[1])
        with (attempt / 'predictions.tmp').open('wb') as f:
            np.savez_compressed(f, val_ids=ids[masks['val']], val_y=vy, val_probs=vp,
                                test_ids=ids[masks['test']], test_y=ty, test_probs=tp, threshold=threshold)
        os.replace(attempt / 'predictions.tmp', attempt / 'predictions.npz')
        record(dict(event='complete', best_epoch=best_epoch, stop_reason=stop_reason, threshold=float(threshold), metrics=result))
    return result, checkpoint
