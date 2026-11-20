# Figure verification

Source thesis: `C:\Users\user\Desktop\AAI_Thesis.pdf`
SHA-256: `8321c9609c6c0c7de409796e5bba1bac94706494379e38c40a91fc4e971a344f`

All 262 checked numeric cells match the supplied thesis at its printed precision. Tables 4.7 and 4.8 contain means only; their SDs are verified against the unchanged pre-edit outputs.

| Table | Numeric cells checked | Result |
|---|---:|---|
| 4.5 | 90 | All match |
| 4.7 | 30 | All match |
| 4.8 | 30 | All match |
| 4.12 | 64 | All match |
| 4.22 | 48 | All match |

All 156 plotted mean/SD pairs agree with `layout_before.json` to less than 1e-14. The requested PR-curve rendering changes; the saved per-run AP and their aggregate statistics do not.

Each default curve uses ten accepted validation-selected checkpoints (seeds 42–51). See `pr_curve_sources.csv` for their paths, hashes and prediction checks. Figure 04 uses 1001 common recall points, sample-SD bands, and excludes the synthetic endpoint. The first reversed point is retained at duplicate recall; endpoint extrapolation is constant. Legend AP is not the integral of the mean curve.

Construction comparisons: 0/180 survive Holm correction (30 hypotheses per metric); the smallest adjusted p-value is 0.05859375.

| Figure | Width (in) | Base text (pt) | Smallest text at 6.3 in (pt) |
|---|---:|---:|---:|
| 01_pipeline | 6.3 | 8 | 8 |
| 02_graph_constructions | 6.3 | 8 | 7 |
| 03_results_ieee-cis | 6.3 | 8 | 7 |
| 04_pr_curves_ieee-cis | 6.3 | 8 | 7 |
| 03_results_elliptic | 6.3 | 8 | 7 |
| 04_pr_curves_elliptic | 6.3 | 8 | 7 |
| 05_graph_construction_results | 6.3 | 8 | 7 |
| 06_connected_vs_isolated | 6.3 | 8 | 7 |
| 07_elliptic_feature_topology_controls | 6.3 | 8 | 7 |

Export checks passed for 33 PDFs and matching 300-dpi PNGs. Thesis PDFs are exactly 6.3 in wide; slide PDFs are 16:9 with one panel per file. PDFs contain embedded TrueType fonts and no raster images.

## Plotted statistics

| Figure | Dataset | Model | Group / arm | Metric | n | Mean | Sample SD |
|---|---|---|---|---|---:|---:|---:|
| 03_results_ieee-cis | IEEE-CIS | MLP | none | pr_auc | 10 | 0.4248846279 | 0.0420183771 |
| 03_results_ieee-cis | IEEE-CIS | LSTM | none | pr_auc | 10 | 0.4150954594 | 0.0088148272 |
| 03_results_ieee-cis | IEEE-CIS | Transformer | none | pr_auc | 10 | 0.4103675313 | 0.0151675526 |
| 03_results_ieee-cis | IEEE-CIS | GCN | main | pr_auc | 10 | 0.2938947783 | 0.0899282178 |
| 03_results_ieee-cis | IEEE-CIS | GraphSAGE | main | pr_auc | 10 | 0.3431741472 | 0.0534103446 |
| 03_results_ieee-cis | IEEE-CIS | GAT | main | pr_auc | 10 | 0.2479692063 | 0.1001581353 |
| 03_results_ieee-cis | IEEE-CIS | MLP | none | f1_score | 10 | 0.4406854152 | 0.0183366511 |
| 03_results_ieee-cis | IEEE-CIS | LSTM | none | f1_score | 10 | 0.4309849837 | 0.0094656732 |
| 03_results_ieee-cis | IEEE-CIS | Transformer | none | f1_score | 10 | 0.4432672372 | 0.0088198462 |
| 03_results_ieee-cis | IEEE-CIS | GCN | main | f1_score | 10 | 0.3470181739 | 0.0569265504 |
| 03_results_ieee-cis | IEEE-CIS | GraphSAGE | main | f1_score | 10 | 0.3788134067 | 0.0190126804 |
| 03_results_ieee-cis | IEEE-CIS | GAT | main | f1_score | 10 | 0.2963929771 | 0.0639664206 |
| 03_results_ieee-cis | IEEE-CIS | MLP | none | recall | 10 | 0.3709444193 | 0.0263033424 |
| 03_results_ieee-cis | IEEE-CIS | LSTM | none | recall | 10 | 0.4005874379 | 0.0180036373 |
| 03_results_ieee-cis | IEEE-CIS | Transformer | none | recall | 10 | 0.4075915047 | 0.0248015092 |
| 03_results_ieee-cis | IEEE-CIS | GCN | main | recall | 10 | 0.3645277903 | 0.0289929244 |
| 03_results_ieee-cis | IEEE-CIS | GraphSAGE | main | recall | 10 | 0.3972435608 | 0.0171705340 |
| 03_results_ieee-cis | IEEE-CIS | GAT | main | recall | 10 | 0.3305919566 | 0.0756522539 |
| 04_pr_curves_ieee-cis | IEEE-CIS | MLP | none | pr_auc | 10 | 0.4248846279 | 0.0420183771 |
| 04_pr_curves_ieee-cis | IEEE-CIS | LSTM | none | pr_auc | 10 | 0.4150954594 | 0.0088148272 |
| 04_pr_curves_ieee-cis | IEEE-CIS | Transformer | none | pr_auc | 10 | 0.4103675313 | 0.0151675526 |
| 04_pr_curves_ieee-cis | IEEE-CIS | GCN | main | pr_auc | 10 | 0.2938947783 | 0.0899282178 |
| 04_pr_curves_ieee-cis | IEEE-CIS | GraphSAGE | main | pr_auc | 10 | 0.3431741472 | 0.0534103446 |
| 04_pr_curves_ieee-cis | IEEE-CIS | GAT | main | pr_auc | 10 | 0.2479692063 | 0.1001581353 |
| 03_results_elliptic | Elliptic | MLP | native | pr_auc | 10 | 0.3900576449 | 0.0486610051 |
| 03_results_elliptic | Elliptic | LSTM | native | pr_auc | 10 | 0.4880112202 | 0.0167893776 |
| 03_results_elliptic | Elliptic | Transformer | native | pr_auc | 10 | 0.5124600542 | 0.0208483039 |
| 03_results_elliptic | Elliptic | GCN | native | pr_auc | 10 | 0.2232600406 | 0.0508528481 |
| 03_results_elliptic | Elliptic | GraphSAGE | native | pr_auc | 10 | 0.4088931444 | 0.0306895701 |
| 03_results_elliptic | Elliptic | GAT | native | pr_auc | 10 | 0.3420500110 | 0.0422230283 |
| 03_results_elliptic | Elliptic | MLP | native | f1_score | 10 | 0.3826994131 | 0.0533713924 |
| 03_results_elliptic | Elliptic | LSTM | native | f1_score | 10 | 0.5101782004 | 0.0138581967 |
| 03_results_elliptic | Elliptic | Transformer | native | f1_score | 10 | 0.5619567083 | 0.0193121884 |
| 03_results_elliptic | Elliptic | GCN | native | f1_score | 10 | 0.2472091134 | 0.0307450713 |
| 03_results_elliptic | Elliptic | GraphSAGE | native | f1_score | 10 | 0.4115446948 | 0.0400611459 |
| 03_results_elliptic | Elliptic | GAT | native | f1_score | 10 | 0.3708619551 | 0.0315262017 |
| 03_results_elliptic | Elliptic | MLP | native | recall | 10 | 0.4556372549 | 0.0279323363 |
| 03_results_elliptic | Elliptic | LSTM | native | recall | 10 | 0.4151960784 | 0.0133347750 |
| 03_results_elliptic | Elliptic | Transformer | native | recall | 10 | 0.4303921569 | 0.0245424619 |
| 03_results_elliptic | Elliptic | GCN | native | recall | 10 | 0.4625000000 | 0.0712149354 |
| 03_results_elliptic | Elliptic | GraphSAGE | native | recall | 10 | 0.4183823529 | 0.0397070591 |
| 03_results_elliptic | Elliptic | GAT | native | recall | 10 | 0.3875000000 | 0.0333359364 |
| 04_pr_curves_elliptic | Elliptic | MLP | native | pr_auc | 10 | 0.3900576449 | 0.0486610051 |
| 04_pr_curves_elliptic | Elliptic | LSTM | native | pr_auc | 10 | 0.4880112202 | 0.0167893776 |
| 04_pr_curves_elliptic | Elliptic | Transformer | native | pr_auc | 10 | 0.5124600542 | 0.0208483039 |
| 04_pr_curves_elliptic | Elliptic | GCN | native | pr_auc | 10 | 0.2232600406 | 0.0508528481 |
| 04_pr_curves_elliptic | Elliptic | GraphSAGE | native | pr_auc | 10 | 0.4088931444 | 0.0306895701 |
| 04_pr_curves_elliptic | Elliptic | GAT | native | pr_auc | 10 | 0.3420500110 | 0.0422230283 |
| 05_graph_construction_results | IEEE-CIS | GCN | main | pr_auc | 10 | 0.2938947783 | 0.0899282178 |
| 05_graph_construction_results | IEEE-CIS | GCN | card | pr_auc | 10 | 0.2746238608 | 0.1083275828 |
| 05_graph_construction_results | IEEE-CIS | GCN | address | pr_auc | 10 | 0.2777019183 | 0.1040926221 |
| 05_graph_construction_results | IEEE-CIS | GCN | email_domain | pr_auc | 10 | 0.2624464625 | 0.1041520706 |
| 05_graph_construction_results | IEEE-CIS | GCN | identifier_time_window-24h | pr_auc | 10 | 0.3256119699 | 0.1060660090 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | main | pr_auc | 10 | 0.3431741472 | 0.0534103446 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | card | pr_auc | 10 | 0.3174243256 | 0.0547339369 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | address | pr_auc | 10 | 0.3665065945 | 0.0571764099 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | email_domain | pr_auc | 10 | 0.3911952017 | 0.0141923065 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | identifier_time_window-24h | pr_auc | 10 | 0.3284371080 | 0.0490858247 |
| 05_graph_construction_results | IEEE-CIS | GAT | main | pr_auc | 10 | 0.2479692063 | 0.1001581353 |
| 05_graph_construction_results | IEEE-CIS | GAT | card | pr_auc | 10 | 0.2176449357 | 0.0992653197 |
| 05_graph_construction_results | IEEE-CIS | GAT | address | pr_auc | 10 | 0.2031836514 | 0.0837476138 |
| 05_graph_construction_results | IEEE-CIS | GAT | email_domain | pr_auc | 10 | 0.1711490735 | 0.0551689082 |
| 05_graph_construction_results | IEEE-CIS | GAT | identifier_time_window-24h | pr_auc | 10 | 0.2342316977 | 0.0926282553 |
| 05_graph_construction_results | IEEE-CIS | MLP | none | pr_auc | 10 | 0.4248846279 | 0.0420183771 |
| 05_graph_construction_results | IEEE-CIS | GCN | main | f1_score | 10 | 0.3470181739 | 0.0569265504 |
| 05_graph_construction_results | IEEE-CIS | GCN | card | f1_score | 10 | 0.3360165664 | 0.0717936458 |
| 05_graph_construction_results | IEEE-CIS | GCN | address | f1_score | 10 | 0.3328477990 | 0.0564043808 |
| 05_graph_construction_results | IEEE-CIS | GCN | email_domain | f1_score | 10 | 0.3243180665 | 0.0579304910 |
| 05_graph_construction_results | IEEE-CIS | GCN | identifier_time_window-24h | f1_score | 10 | 0.3573487163 | 0.0633868982 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | main | f1_score | 10 | 0.3788134067 | 0.0190126804 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | card | f1_score | 10 | 0.3705649152 | 0.0229136315 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | address | f1_score | 10 | 0.3916621097 | 0.0146277227 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | email_domain | f1_score | 10 | 0.4024695259 | 0.0099566861 |
| 05_graph_construction_results | IEEE-CIS | GraphSAGE | identifier_time_window-24h | f1_score | 10 | 0.3899944695 | 0.0124938138 |
| 05_graph_construction_results | IEEE-CIS | GAT | main | f1_score | 10 | 0.2963929771 | 0.0639664206 |
| 05_graph_construction_results | IEEE-CIS | GAT | card | f1_score | 10 | 0.2744922916 | 0.0688367906 |
| 05_graph_construction_results | IEEE-CIS | GAT | address | f1_score | 10 | 0.2727076849 | 0.0513854858 |
| 05_graph_construction_results | IEEE-CIS | GAT | email_domain | f1_score | 10 | 0.2577594159 | 0.0401356640 |
| 05_graph_construction_results | IEEE-CIS | GAT | identifier_time_window-24h | f1_score | 10 | 0.2941960961 | 0.0634522167 |
| 05_graph_construction_results | IEEE-CIS | MLP | none | f1_score | 10 | 0.4406854152 | 0.0183366511 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | main/connected | pr_auc | 10 | 0.5071354834 | 0.0154709335 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | main/connected | pr_auc | 10 | 0.3564579308 | 0.0786093536 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | main/connected | pr_auc | 10 | 0.4546942264 | 0.0077762793 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | main/connected | pr_auc | 10 | 0.3131270507 | 0.0889143034 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | main/isolated | pr_auc | 10 | 0.2664917523 | 0.0548376196 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | main/isolated | pr_auc | 10 | 0.2022271303 | 0.0767169151 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | main/isolated | pr_auc | 10 | 0.1744706438 | 0.0422541022 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | main/isolated | pr_auc | 10 | 0.1587461744 | 0.0808939147 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | card/connected | pr_auc | 10 | 0.4922698261 | 0.0099020968 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | card/connected | pr_auc | 10 | 0.3204226201 | 0.0480459454 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | card/connected | pr_auc | 10 | 0.4271227479 | 0.0053169633 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | card/connected | pr_auc | 10 | 0.3330430122 | 0.0832719807 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | card/isolated | pr_auc | 10 | 0.3620333313 | 0.0608346919 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | card/isolated | pr_auc | 10 | 0.2693581215 | 0.1266041048 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | card/isolated | pr_auc | 10 | 0.2439938035 | 0.0559734974 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | card/isolated | pr_auc | 10 | 0.1690974010 | 0.0881536164 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | address/connected | pr_auc | 10 | 0.6802589997 | 0.0335844215 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | address/connected | pr_auc | 10 | 0.6115883757 | 0.1268062192 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | address/connected | pr_auc | 10 | 0.3810283809 | 0.0328942406 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | address/connected | pr_auc | 10 | 0.5749885697 | 0.0791895483 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | address/isolated | pr_auc | 10 | 0.4201541187 | 0.0430691020 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | address/isolated | pr_auc | 10 | 0.2749569146 | 0.1033095321 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | address/isolated | pr_auc | 10 | 0.3668725716 | 0.0576192817 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | address/isolated | pr_auc | 10 | 0.2005941135 | 0.0829621628 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | email_domain/connected | pr_auc | 10 | 0.2673716738 | 0.0470698671 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | email_domain/connected | pr_auc | 10 | 0.1397383184 | 0.0395480955 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | email_domain/connected | pr_auc | 10 | 0.2652331271 | 0.0225560903 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | email_domain/connected | pr_auc | 10 | 0.2333481022 | 0.0583461270 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | email_domain/isolated | pr_auc | 10 | 0.4296757003 | 0.0433641500 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | email_domain/isolated | pr_auc | 10 | 0.2664816525 | 0.1065160670 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | email_domain/isolated | pr_auc | 10 | 0.3958823658 | 0.0148363421 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | email_domain/isolated | pr_auc | 10 | 0.1713280699 | 0.0559377053 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | identifier_time_window-24h/connected | pr_auc | 10 | 0.6228100104 | 0.0342364796 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | identifier_time_window-24h/connected | pr_auc | 10 | 0.4696060789 | 0.1382038020 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | identifier_time_window-24h/connected | pr_auc | 10 | 0.4271446022 | 0.1107355073 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | identifier_time_window-24h/connected | pr_auc | 10 | 0.2933291806 | 0.0883813784 |
| 06_connected_vs_isolated | IEEE-CIS | MLP | identifier_time_window-24h/isolated | pr_auc | 10 | 0.3436398784 | 0.0391604508 |
| 06_connected_vs_isolated | IEEE-CIS | GCN | identifier_time_window-24h/isolated | pr_auc | 10 | 0.2638809936 | 0.0912710320 |
| 06_connected_vs_isolated | IEEE-CIS | GraphSAGE | identifier_time_window-24h/isolated | pr_auc | 10 | 0.3011649343 | 0.0098657031 |
| 06_connected_vs_isolated | IEEE-CIS | GAT | identifier_time_window-24h/isolated | pr_auc | 10 | 0.2271881553 | 0.1002258097 |
| 07_elliptic_feature_topology_controls | Elliptic | MLP | original | pr_auc | 10 | 0.3900576449 | 0.0486610051 |
| 07_elliptic_feature_topology_controls | Elliptic | LSTM | original | pr_auc | 10 | 0.4880112202 | 0.0167893776 |
| 07_elliptic_feature_topology_controls | Elliptic | Transformer | original | pr_auc | 10 | 0.5124600542 | 0.0208483039 |
| 07_elliptic_feature_topology_controls | Elliptic | GCN | original | pr_auc | 10 | 0.2232600406 | 0.0508528481 |
| 07_elliptic_feature_topology_controls | Elliptic | GraphSAGE | original | pr_auc | 10 | 0.4088931444 | 0.0306895701 |
| 07_elliptic_feature_topology_controls | Elliptic | GAT | original | pr_auc | 10 | 0.3420500110 | 0.0422230283 |
| 07_elliptic_feature_topology_controls | Elliptic | MLP | local | pr_auc | 10 | 0.4602358187 | 0.0143404334 |
| 07_elliptic_feature_topology_controls | Elliptic | LSTM | local | pr_auc | 10 | 0.4415015175 | 0.0170912016 |
| 07_elliptic_feature_topology_controls | Elliptic | Transformer | local | pr_auc | 10 | 0.5235622219 | 0.0088626154 |
| 07_elliptic_feature_topology_controls | Elliptic | GCN | local | pr_auc | 10 | 0.1569211897 | 0.0215103776 |
| 07_elliptic_feature_topology_controls | Elliptic | GraphSAGE | local | pr_auc | 10 | 0.4154995197 | 0.0742133605 |
| 07_elliptic_feature_topology_controls | Elliptic | GAT | local | pr_auc | 10 | 0.3668854425 | 0.0855300143 |
| 07_elliptic_feature_topology_controls | Elliptic | MLP | original | f1_score | 10 | 0.3826994131 | 0.0533713924 |
| 07_elliptic_feature_topology_controls | Elliptic | LSTM | original | f1_score | 10 | 0.5101782004 | 0.0138581967 |
| 07_elliptic_feature_topology_controls | Elliptic | Transformer | original | f1_score | 10 | 0.5619567083 | 0.0193121884 |
| 07_elliptic_feature_topology_controls | Elliptic | GCN | original | f1_score | 10 | 0.2472091134 | 0.0307450713 |
| 07_elliptic_feature_topology_controls | Elliptic | GraphSAGE | original | f1_score | 10 | 0.4115446948 | 0.0400611459 |
| 07_elliptic_feature_topology_controls | Elliptic | GAT | original | f1_score | 10 | 0.3708619551 | 0.0315262017 |
| 07_elliptic_feature_topology_controls | Elliptic | MLP | local | f1_score | 10 | 0.5654503776 | 0.0211275562 |
| 07_elliptic_feature_topology_controls | Elliptic | LSTM | local | f1_score | 10 | 0.4864528061 | 0.0172665476 |
| 07_elliptic_feature_topology_controls | Elliptic | Transformer | local | f1_score | 10 | 0.5923506782 | 0.0107869364 |
| 07_elliptic_feature_topology_controls | Elliptic | GCN | local | f1_score | 10 | 0.2493925997 | 0.0258131885 |
| 07_elliptic_feature_topology_controls | Elliptic | GraphSAGE | local | f1_score | 10 | 0.3841924360 | 0.0905453690 |
| 07_elliptic_feature_topology_controls | Elliptic | GAT | local | f1_score | 10 | 0.3562394297 | 0.0782003013 |
| 07_elliptic_feature_topology_controls | Elliptic | GCN | original | pr_auc | 10 | 0.2232600406 | 0.0508528481 |
| 07_elliptic_feature_topology_controls | Elliptic | GraphSAGE | original | pr_auc | 10 | 0.4088931444 | 0.0306895701 |
| 07_elliptic_feature_topology_controls | Elliptic | GAT | original | pr_auc | 10 | 0.3420500110 | 0.0422230283 |
| 07_elliptic_feature_topology_controls | Elliptic | GCN | random | pr_auc | 10 | 0.0858608501 | 0.0094104464 |
| 07_elliptic_feature_topology_controls | Elliptic | GraphSAGE | random | pr_auc | 10 | 0.1693544896 | 0.0125010102 |
| 07_elliptic_feature_topology_controls | Elliptic | GAT | random | pr_auc | 10 | 0.1979490491 | 0.0309599542 |
| 07_elliptic_feature_topology_controls | Elliptic | GCN | original | f1_score | 10 | 0.2472091134 | 0.0307450713 |
| 07_elliptic_feature_topology_controls | Elliptic | GraphSAGE | original | f1_score | 10 | 0.4115446948 | 0.0400611459 |
| 07_elliptic_feature_topology_controls | Elliptic | GAT | original | f1_score | 10 | 0.3708619551 | 0.0315262017 |
| 07_elliptic_feature_topology_controls | Elliptic | GCN | random | f1_score | 10 | 0.1492551456 | 0.0115775820 |
| 07_elliptic_feature_topology_controls | Elliptic | GraphSAGE | random | f1_score | 10 | 0.2901707077 | 0.0164288481 |
| 07_elliptic_feature_topology_controls | Elliptic | GAT | random | f1_score | 10 | 0.2819736434 | 0.0437100796 |
