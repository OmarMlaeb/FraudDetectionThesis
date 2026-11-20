# Exact sequence inputs for the selected August aggregate experiments

These descriptions apply to the August seeds 42–51 selected for thesis Tables 4.7 and 4.8. They do not use the older Elliptic final-run Table 4.10 as evidence. Source snapshots are included under `source_snapshots/2be96ee/`; the construction GNN source is also retained under `source_snapshots/5333c55/`.

## IEEE-CIS

1. Left-merge training transactions and identities on TransactionID. Sort by `(TransactionDT, TransactionID)` using mergesort. For the present raw data this equals its existing transaction order.
2. Split rows into training 472,432 (80%), validation 59,054 (10%) and test 59,054 (10%). Fit feature-column selection (missingness >80%), categorical encodings, numerical medians and StandardScaler on training rows only. Drop label and TransactionID from features. Retain 358 features, including TransactionDT. Reconstructed float32 features match every saved construction graph exactly.
3. LSTM and Transformer each receive a sequence of length 10. For target index `i` within a split, use rows `max(0, i-9)` through `i`, inclusive. Left-pad by repeating the split's first row to reach length 10. The target label is that of row `i`.
4. Windows are split-local: the first test example does not use validation rows. They are a global transaction stream, not card/account/customer-specific sequences. Every target is evaluated, including the first nine test rows. Test IDs are 3518486–3577539 inclusive.
5. Input shape is `(batch, 10, 358)`; evaluation batch size is 512. LSTM uses the last recurrent output; Transformer projects features to 128 dimensions, adds learned position embeddings and classifies the last encoder output. No attention mask is supplied, but the window contains only the target and preceding transactions. Dropout is disabled during inference.
6. LSTM checkpoint selection uses validation PR-AUC (average precision); threshold selection maximizes validation F1 on 0.01–0.99. August IEEE-CIS Transformer checkpoint selection uses validation F1 constrained to at most 20% positive predictions. Its threshold candidates combine the 0.01–0.99 grid, 501 probability quantiles and a value just above the maximum score. The originally saved threshold is used for test evaluation.

## Elliptic

1. Join transaction features/classes on txId. Sort all nodes by `(time_step, txId)`. Labels: class 1 = illicit/positive; class 2 = licit/negative. Unknown labels are excluded from supervised splits. The 165 numerical feature values are float32; NaN/inf are replaced with zero. txId and time_step are not feature columns in the model.
2. Split the 49 labeled time steps chronologically: 1–34 training, 35–41 validation and 42–49 test. This gives 29,894 / 7,829 / 8,841 labeled targets. Unknown-label nodes remain in the GNN inference graph but are not sequence inputs or targets.
3. Inside each split, first select its labeled rows in the above order. For each target, take the preceding nine labeled rows plus the target, padding on the left with the first split row when needed. Windows may cross time-step boundaries within a split. They do not cross split boundaries and do not group by wallet or follow graph paths.
4. Input shape is `(batch, 10, 165)` and evaluation batch size is 512. Both architectures evaluate all 8,841 test targets, including the first nine. Checkpoint selection is validation average precision and threshold selection is validation F1 on 0.01–0.99.

## Original versus corrected behavior

The pre-August sequence implementation returned `N-9` windows for a split of N labeled examples, with the first target at index 9. The August implementation returns N windows with left-padding. Thus the original IEEE-CIS sequence test count was 59,045 and original Elliptic count 8,832; the selected August counts are 59,054 and 8,841 respectively. The October source changes add assertions and logging for the Elliptic population, without changing the sequence implementation.

Evidence is threefold: the frozen August source defines padding, the selected per-seed confusion matrices have the corrected counts, and inference through all 40 saved August sequence checkpoints reproduces the selected results and validation threshold checks. `sequence_input_checks.json` records example target/input-ID windows and equivalence checks against the source Dataset classes. The inference runner uses vectorized window gathering but checks equivalence to the source classes; it does not train or change sequence behavior.
