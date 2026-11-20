# Connected versus isolated test-set analysis

All 200 required GNN/MLP checkpoints passed the predefined reproduction gate. These are post-hoc groups on the full saved inference graph, with self-loops ignored. MLP predictions are reused across constructions for identical IDs. Every run uses its saved validation-selected threshold; no test-group tuning occurred. PR-AUC means average precision. Mean ± sample SD across ten seeds.

## Group populations

| Dataset | Construction | Group | N | Positive | Negative | Fraud rate |
|---|---|---|---:|---:|---:|---:|
| IEEE-CIS | main | connected | 33,622 | 1,557 | 32,065 | 4.6309% |
| IEEE-CIS | main | isolated | 25,432 | 656 | 24,776 | 2.5794% |
| IEEE-CIS | card | connected | 30,936 | 1,178 | 29,758 | 3.8079% |
| IEEE-CIS | card | isolated | 28,118 | 1,035 | 27,083 | 3.6809% |
| IEEE-CIS | address | connected | 481 | 37 | 444 | 7.6923% |
| IEEE-CIS | address | isolated | 58,573 | 2,176 | 56,397 | 3.7150% |
| IEEE-CIS | email_domain | connected | 1,488 | 65 | 1,423 | 4.3683% |
| IEEE-CIS | email_domain | isolated | 57,566 | 2,148 | 55,418 | 3.7314% |
| IEEE-CIS | identifier_time_window-24h | connected | 4,746 | 655 | 4,091 | 13.8011% |
| IEEE-CIS | identifier_time_window-24h | isolated | 54,308 | 1,558 | 52,750 | 2.8688% |
| Elliptic | native | connected | 8,841 | 408 | 8,433 | 4.6149% |
| Elliptic | native | isolated | 0 | 0 | 0 | undefined |

Elliptic has zero isolated labeled test targets. All 40 Elliptic isolated per-seed rows are explicitly empty; rates/AP/F1/recall are undefined and confusion counts are zero. This is a population limitation, not missing predictions. No connected-versus-isolated performance contrast can be estimated there.

## IEEE-CIS: main

| Group | Model | Seeds | Average precision | F1 | Recall | False positives |
|---|---|---:|---:|---:|---:|---:|
| connected | GAT | 10 | 0.3131 ± 0.0889 | 0.3557 ± 0.0622 | 0.3765 ± 0.0898 | 1132.4 ± 411.2 |
| isolated | GAT | 10 | 0.1587 ± 0.0809 | 0.1833 ± 0.0586 | 0.2216 ± 0.0438 | 916.5 ± 539.9 |
| connected | GCN | 10 | 0.3565 ± 0.0786 | 0.4108 ± 0.0498 | 0.4058 ± 0.0326 | 940.9 ± 526.7 |
| isolated | GCN | 10 | 0.2022 ± 0.0767 | 0.2253 ± 0.0616 | 0.2665 ± 0.0423 | 830.2 ± 492.0 |
| connected | GraphSAGE | 10 | 0.4547 ± 0.0078 | 0.4387 ± 0.0064 | 0.4653 ± 0.0209 | 1021.8 ± 132.4 |
| isolated | GraphSAGE | 10 | 0.1745 ± 0.0423 | 0.2344 ± 0.0328 | 0.2358 ± 0.0159 | 539.6 ± 260.0 |
| connected | MLP | 10 | 0.5071 ± 0.0155 | 0.4939 ± 0.0147 | 0.4182 ± 0.0307 | 426.1 ± 94.4 |
| isolated | MLP | 10 | 0.2665 ± 0.0548 | 0.3125 ± 0.0224 | 0.2588 ± 0.0179 | 265.1 ± 91.7 |
## IEEE-CIS: card

| Group | Model | Seeds | Average precision | F1 | Recall | False positives |
|---|---|---:|---:|---:|---:|---:|
| connected | GAT | 10 | 0.3330 ± 0.0833 | 0.3389 ± 0.0767 | 0.3298 ± 0.0960 | 706.1 ± 456.5 |
| isolated | GAT | 10 | 0.1691 ± 0.0882 | 0.2205 ± 0.0610 | 0.2867 ± 0.1108 | 1450.6 ± 976.8 |
| connected | GCN | 10 | 0.3204 ± 0.0480 | 0.3621 ± 0.0380 | 0.3379 ± 0.0356 | 661.1 ± 436.6 |
| isolated | GCN | 10 | 0.2694 ± 0.1266 | 0.3185 ± 0.0936 | 0.3939 ± 0.0228 | 1387.9 ± 1041.1 |
| connected | GraphSAGE | 10 | 0.4271 ± 0.0053 | 0.4051 ± 0.0079 | 0.4304 ± 0.0105 | 819.2 ± 69.3 |
| isolated | GraphSAGE | 10 | 0.2440 ± 0.0560 | 0.3336 ± 0.0367 | 0.3549 ± 0.0121 | 830.7 ± 316.1 |
| connected | MLP | 10 | 0.4923 ± 0.0099 | 0.4716 ± 0.0110 | 0.3966 ± 0.0278 | 335.1 ± 82.6 |
| isolated | MLP | 10 | 0.3620 ± 0.0608 | 0.4062 ± 0.0303 | 0.3417 ± 0.0267 | 356.1 ± 107.6 |
## IEEE-CIS: address

| Group | Model | Seeds | Average precision | F1 | Recall | False positives |
|---|---|---:|---:|---:|---:|---:|
| connected | GAT | 10 | 0.5750 ± 0.0792 | 0.5430 ± 0.1940 | 0.4757 ± 0.1757 | 7.4 ± 4.4 |
| isolated | GAT | 10 | 0.2006 ± 0.0830 | 0.2691 ± 0.0498 | 0.3017 ± 0.0618 | 2034.8 ± 401.5 |
| connected | GCN | 10 | 0.6116 ± 0.1268 | 0.5981 ± 0.0645 | 0.5270 ± 0.0481 | 8.9 ± 4.1 |
| isolated | GCN | 10 | 0.2750 ± 0.1033 | 0.3296 ± 0.0560 | 0.4012 ± 0.0399 | 2450.1 ± 1371.0 |
| connected | GraphSAGE | 10 | 0.3810 ± 0.0329 | 0.3993 ± 0.0416 | 0.3649 ± 0.0387 | 17.2 ± 3.6 |
| isolated | GraphSAGE | 10 | 0.3669 ± 0.0576 | 0.3916 ± 0.0152 | 0.4184 ± 0.0377 | 1569.1 ± 362.3 |
| connected | MLP | 10 | 0.6803 ± 0.0336 | 0.6189 ± 0.0327 | 0.5243 ± 0.0463 | 6.3 ± 3.3 |
| isolated | MLP | 10 | 0.4202 ± 0.0431 | 0.4377 ± 0.0186 | 0.3683 ± 0.0262 | 684.9 ± 159.4 |
## IEEE-CIS: email_domain

| Group | Model | Seeds | Average precision | F1 | Recall | False positives |
|---|---|---:|---:|---:|---:|---:|
| connected | GAT | 10 | 0.2333 ± 0.0583 | 0.2120 ± 0.0692 | 0.1615 ± 0.0608 | 23.3 ± 11.9 |
| isolated | GAT | 10 | 0.1713 ± 0.0559 | 0.2587 ± 0.0398 | 0.3300 ± 0.0868 | 2594.2 ± 972.6 |
| connected | GCN | 10 | 0.1397 ± 0.0395 | 0.1203 ± 0.0162 | 0.1077 ± 0.0528 | 45.0 ± 50.0 |
| isolated | GCN | 10 | 0.2665 ± 0.1065 | 0.3283 ± 0.0586 | 0.4085 ± 0.0386 | 2558.1 ± 1492.4 |
| connected | GraphSAGE | 10 | 0.2652 ± 0.0226 | 0.2814 ± 0.0332 | 0.2908 ± 0.0432 | 50.5 ± 10.8 |
| isolated | GraphSAGE | 10 | 0.3959 ± 0.0148 | 0.4062 ± 0.0108 | 0.4135 ± 0.0332 | 1340.9 ± 316.5 |
| connected | MLP | 10 | 0.2674 ± 0.0471 | 0.2950 ± 0.0577 | 0.2108 ± 0.0470 | 13.9 ± 4.2 |
| isolated | MLP | 10 | 0.4297 ± 0.0434 | 0.4444 ± 0.0185 | 0.3758 ± 0.0264 | 677.3 ± 158.9 |
## IEEE-CIS: identifier_time_window-24h

| Group | Model | Seeds | Average precision | F1 | Recall | False positives |
|---|---|---:|---:|---:|---:|---:|
| connected | GAT | 10 | 0.2933 ± 0.0884 | 0.3648 ± 0.0764 | 0.4466 ± 0.1496 | 617.7 ± 278.3 |
| isolated | GAT | 10 | 0.2272 ± 0.1002 | 0.2621 ± 0.0649 | 0.2804 ± 0.0632 | 1388.4 ± 530.7 |
| connected | GCN | 10 | 0.4696 ± 0.1382 | 0.4857 ± 0.0883 | 0.6208 ± 0.0889 | 709.2 ± 578.9 |
| isolated | GCN | 10 | 0.2639 ± 0.0913 | 0.2901 ± 0.0568 | 0.2907 ± 0.0333 | 1241.5 ± 805.7 |
| connected | GraphSAGE | 10 | 0.4271 ± 0.1107 | 0.5027 ± 0.0324 | 0.6450 ± 0.0368 | 608.1 ± 112.0 |
| isolated | GraphSAGE | 10 | 0.3012 ± 0.0099 | 0.3186 ± 0.0061 | 0.2707 ± 0.0243 | 665.6 ± 175.8 |
| connected | MLP | 10 | 0.6228 ± 0.0342 | 0.5906 ± 0.0254 | 0.5302 ± 0.0418 | 172.8 ± 38.8 |
| isolated | MLP | 10 | 0.3436 ± 0.0392 | 0.3715 ± 0.0149 | 0.3040 ± 0.0226 | 518.4 ± 133.7 |
## Elliptic: native

| Group | Model | Seeds | Average precision | F1 | Recall | False positives |
|---|---|---:|---:|---:|---:|---:|
| connected | GAT | 10 | 0.3420 ± 0.0422 | 0.3709 ± 0.0315 | 0.3875 ± 0.0333 | 294.6 ± 113.4 |
| isolated | GAT | 10 | undefined | undefined | undefined | 0.0 ± 0.0 |
| connected | GCN | 10 | 0.2233 ± 0.0509 | 0.2472 ± 0.0307 | 0.4625 ± 0.0712 | 975.6 ± 404.4 |
| isolated | GCN | 10 | undefined | undefined | undefined | 0.0 ± 0.0 |
| connected | GraphSAGE | 10 | 0.4089 ± 0.0307 | 0.4115 ± 0.0401 | 0.4184 ± 0.0397 | 254.3 ± 68.4 |
| isolated | GraphSAGE | 10 | undefined | undefined | undefined | 0.0 ± 0.0 |
| connected | MLP | 10 | 0.3901 ± 0.0487 | 0.3827 ± 0.0534 | 0.4556 ± 0.0279 | 401.8 ± 190.2 |
| isolated | MLP | 10 | undefined | undefined | undefined | 0.0 ± 0.0 |

## Interpretation and checks

On the main IEEE-CIS graph, MLP mean AP is 0.5071 for connected and 0.2665 for isolated targets; GraphSAGE-main is 0.4547 and 0.1745 on exactly the same IDs. Connected and isolated fraud rates differ (4.6309% versus 2.5794%), so raw AP differences between groups do not by themselves measure the benefit of neighbors. The MLP control helps separate group difficulty from model behavior. The email-domain and address connected groups are small (1,488 and 481 targets), so retain their sizes and ten-seed SD when interpreting results. No subgroup significance tests or causal claims are made.

All 240 construction/model/seed evaluations reconcile their two subgroup confusion matrices exactly to the saved whole-test counts. See `subgroup_confusion_reconciliation.csv`. `subgroup_per_seed.csv` contains all 480 rows; `subgroup_summary.csv` also contains precision, ROC-AUC, counts, rates and the number of defined seed values. Undefined metrics are blank in CSV and null in JSON, with explicit status fields; they are not replaced by zero.
