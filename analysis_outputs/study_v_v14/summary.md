# Study V prospective comparison v1.4

Real testbed observations only. No results imputed or edited.
Recovery is evaluated by **10 seconds**, as explicitly approved before collection. The full 20-second endpoint is unavailable.

Population: {"accepted_unique_primary_rows": 45, "complete_matched_blocks": 10, "recorded_attempts": 69, "rejected_attempts": 5, "planned_new_attempts": 54}
Complete 12-block primary comparison: False

The CSV tables list every accepted, rejected and unattempted row. G-pre is exploratory only.
Primary Holm family: 36 planned hypotheses across all/interference/no-interference scopes; exploratory G-pre family: 16.
Intervals use 10,000 matched-block resamples. Undefined comparisons retain null tests and intervals.

- all, G-policy, rlc_delay_outage_rate: n=11, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- all, G-policy, throughput_retention: n=11, difference=-0.020273973951210215, CI=[-0.06778785314962238, 0.010576746160521152], Holm p=1.0
- all, G-policy, rlc_drop_rate: n=11, difference=-0.004802947488425463, CI=[-0.016786157628337264, 0.005328024975829152], Holm p=1.0
- all, G-policy, recovery_by_10s: n=11, difference=-0.09090909090909091, CI=[-0.2727272727272727, 0.0], Holm p=1.0
- all, G-no_action, rlc_delay_outage_rate: n=10, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- all, G-no_action, throughput_retention: n=10, difference=-0.007888123215451012, CI=[-0.023542289288956304, 0.004013026143008819], Holm p=1.0
- all, G-no_action, rlc_drop_rate: n=10, difference=-0.008785939440572876, CI=[-0.015566135214573595, -0.0026109266200357814], Holm p=1.0
- all, G-no_action, recovery_by_10s: n=10, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- all, G-fixed_a2, rlc_delay_outage_rate: n=10, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- all, G-fixed_a2, throughput_retention: n=10, difference=0.0006364435213206656, CI=[-0.020847679443730262, 0.022744207345635686], Holm p=1.0
- all, G-fixed_a2, rlc_drop_rate: n=10, difference=-0.00023014708405620367, CI=[-0.014143195123408996, 0.01923069771524097], Holm p=1.0
- all, G-fixed_a2, recovery_by_10s: n=10, difference=0.3, CI=[0.0, 0.6], Holm p=1.0
- interference, G-policy, rlc_delay_outage_rate: n=7, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- interference, G-policy, throughput_retention: n=7, difference=-0.035858375639549515, CI=[-0.10362787185223357, -0.000534509683017313], Holm p=1.0
- interference, G-policy, rlc_drop_rate: n=7, difference=-0.007547488910382871, CI=[-0.02520292599934441, 0.008894064853295493], Holm p=1.0
- interference, G-policy, recovery_by_10s: n=7, difference=-0.14285714285714285, CI=[-0.42857142857142855, 0.0], Holm p=1.0
- interference, G-no_action, rlc_delay_outage_rate: n=6, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- interference, G-no_action, throughput_retention: n=6, difference=0.0005883337500628642, CI=[-0.0002612916596900115, 0.001431471013514737], Holm p=1.0
- interference, G-no_action, rlc_drop_rate: n=6, difference=-0.014643232400954794, CI=[-0.022386084527012223, -0.006330586673951792], Holm p=1.0
- interference, G-no_action, recovery_by_10s: n=6, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- interference, G-fixed_a2, rlc_delay_outage_rate: n=6, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- interference, G-fixed_a2, throughput_retention: n=6, difference=-0.000938957928585221, CI=[-0.004069097640450236, 0.0011104369135151715], Holm p=1.0
- interference, G-fixed_a2, rlc_drop_rate: n=6, difference=-0.0003835784734270061, CI=[-0.0223999756096727, 0.032051162858734945], Holm p=1.0
- interference, G-fixed_a2, recovery_by_10s: n=6, difference=0.5, CI=[0.16666666666666666, 0.8333333333333334], Holm p=1.0
- no_interference, G-policy, rlc_delay_outage_rate: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- no_interference, G-policy, throughput_retention: n=4, difference=0.0069987290033835525, CI=[-0.031192653904733803, 0.033375318411800914], Holm p=1.0
- no_interference, G-policy, rlc_drop_rate: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- no_interference, G-policy, recovery_by_10s: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- no_interference, G-no_action, rlc_delay_outage_rate: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- no_interference, G-no_action, throughput_retention: n=4, difference=-0.020602808663721828, CI=[-0.050835775341661726, 0.009630158014218071], Holm p=1.0
- no_interference, G-no_action, rlc_drop_rate: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- no_interference, G-no_action, recovery_by_10s: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- no_interference, G-fixed_a2, rlc_delay_outage_rate: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- no_interference, G-fixed_a2, throughput_retention: n=4, difference=0.0029995456961794953, CI=[-0.04668746524116174, 0.05394594791327989], Holm p=1.0
- no_interference, G-fixed_a2, rlc_drop_rate: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
- no_interference, G-fixed_a2, recovery_by_10s: n=4, difference=0.0, CI=[0.0, 0.0], Holm p=1.0
