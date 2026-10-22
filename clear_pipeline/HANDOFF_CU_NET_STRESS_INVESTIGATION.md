> **SUPERSEDED.** A validated fix now exists — see `FINAL_REPORT_CU_NET_STRESS.md` for the final root cause, the implemented fix (`net_diff`-only scoring + p91 threshold + 5-consecutive-row debounce), and full before/after numbers. This document is kept only for the earlier investigation history (rate-cap experiments, etc.) that led up to it.

# Handoff: CU-NET Stress Detection Investigation

## Project context

- Anomaly-detection system (TopoAR, an autoregressive/reconstruction-based model) monitoring an O-RAN srsRAN testbed (Topaz framework): CUs, DUs, UEs, 5GC, running via Docker Compose on a server (`softran3`).
- 4 topologies: `cu0_du0du1` (2 DUs), `cu1_du2` (1 DU), `cu2_du3du4du5` (3 DUs), `cu3_du6du7du8du9` (4 DUs).
- 3 stress types injected per entity: CPU (`stress-ng --cpu-load`), MEM (`stress-ng --vm-bytes`), NET (`tc qdisc ... netem`).
- Model scores each entity (CU, each DU) on an anomaly "lift score" (prediction error / calibrated feat_norm) and flags anomalies above a threshold calibrated only from held-out normal training data (no test labels used).
- Data collection: `topaz/topaz/experimental_phase0/scrapper_prometheus_backup.py`, orchestrated by `new_start_setup.sh`. cAdvisor scrapes container network/cpu/mem stats into Prometheus; srsRAN's own JSON metrics stream goes to `metrics_server` → InfluxDB (only `ue_info`, `app_resource_usage`, `ru.ofh` are forwarded — the `cu-up`/`pdcp` block is computed by srsRAN but silently dropped by `metrics_server`'s ingestion code, unused).
- ML pipeline: `clear_pipeline/build_dataset.py` (builds `train.npz`), `build_newdata_test.py` (builds `test_newdata.npz` from new per-entity CSVs, now patched to pick up multiple reps per entity via glob), `run_experiment_newdata.py` (loads frozen checkpoints, evaluates, no retraining).

## The original problem

CPU stress, MEM stress, and DU-side NET stress are all detected reliably (F1 ~0.88–0.99 across topologies). **CU-side NET stress detection was badly broken and inconsistent** (F1 ranging from ~0.02 to ~0.92 depending on topology, no stable pattern).

## Root cause chain (confirmed, in order of discovery)

1. **A scoring "fix" (`CU_NET_GATING`) was masking the real problem.** It set the CU-NET decision threshold as the 90th percentile of the *test stream being scored itself* — a circular, invalid calibration (info leakage). This made some topologies look artificially good (e.g. cu2 hit F1=0.92) while genuinely broken elsewhere. **Confirmed by disabling it**: with an honest, train-only-calibrated threshold, F1 collapsed to ~0.00–0.014 across all 4 topologies. This gating logic is now correctly disabled (`ENABLE_CU_NET_GATING = False` in `run_experiment_newdata.py`) and should stay disabled.

2. **Collection methodology drift.** Original good 3-topology result (F1 0.93–0.98, from `existing-result-csv/loo_results_v0_noratio_CU_NET_random_STRESS.csv`) came from one continuous multi-hour collection session (`scrapper_prometheus_backup_ORIGINAL.py`). The newer isolated-restart methodology (`restart_full_stack()` + 5-min warmup + 10-min normal + 5-min stress, per entity) was adopted deliberately to fix a separate suspected issue (stress-residue contaminating "normal" labels in the old continuous method) — this was a reasonable, deliberate tradeoff, not a mistake, but it introduces more baseline noise (short warmup) and interacts with an independent confound (below).

3. **Traffic-generator confound (real, confirmed, not yet fixed).** `test_traffic.py` cycles through 25 fifteen-minute windows with wildly different target load (1.9–15 Gb), running independently of and unsynchronized with whichever isolated stress test happens to be executing. Confirmed via direct measurement: `cu0`'s "quiet" baseline drifted 481,472 → 260,930 → 180,506 bytes/sec across different test runs, purely from this schedule, unrelated to any injected fault. This inflates/deflates whatever topology's test happens to land on a given schedule phase. **Not yet fixed** — user explicitly rejected forcing a constant traffic rate (would remove realistic time-varying load); a synchronized-restart-of-traffic-gen approach was proposed but not implemented.

4. **Only 1 repetition was originally collected; increased to 2** (`--fault-reps 2`) to average out the random loss-intensity draw (`intensity = random.randint(85, 95)` in `injectStress()`) and the traffic-schedule confound. Two independent reps were successfully collected and used in later evaluations.

5. **Protocol-level root cause (the deepest, most solid finding) — CONFIRMED BY DIRECT PACKET CAPTURE, not just theory:**
   - CU's `eth0` carries F1AP+NGAP (SCTP, control-plane) and F1-U/GTP-U (UDP, user-plane, tunneling the UE's `iperf -u` traffic).
   - **SCTP's byte volume is negligible — confirmed 0.01% of total bytes** in a live pcap capture (195,324,410 bytes UDP vs ~20,458 bytes SCTP combined, over 5 minutes). This DISPROVES the "SCTP retransmission cancels/neutralizes UDP's loss" hypothesis as the dominant mechanism — SCTP is too small in volume to matter, and didn't even increase during the loss window in the capture (went 70.7→66.5 bytes/sec, if anything down).
   - UDP has no congestion control / no retransmission — packet loss just proportionally reduces bytes with no compensation. Confirmed via pcap, direction-separated: CU's **TX** (egress, what `netem loss` actually touches) dropped ~4.3% under ~4.5% configured loss (315,593 vs 329,850 bytes/sec) — closely matching the configured loss rate, as expected for non-adaptive UDP. CU's **RX** stayed completely flat (~0% change, 330,341→330,418), confirming `netem`'s egress-only mechanism (root qdisc affects outbound traffic only, never inbound).
   - DU's dominant traffic is the ZMQ-emulated RF I/Q stream to the UE, running over **TCP** (confirmed from `du_zmq.conf`: `tx_port0=tcp://...`). TCP's loss-based congestion control interprets loss as congestion and multiplicatively cuts throughput — a well-documented large effect (Mathis-equation-style), unlike UDP's small linear effect. **Correction made mid-investigation**: DU's own TX is NOT near-zero (an earlier claim was wrong, verified and corrected) — it's large and roughly symmetric with RX (~89M bytes/sec each way, quiet). Under stress, BOTH DU's TX and RX collapse together by ~91% (89.3M→8.3M bytes/sec on both sides), because it's one bidirectional TCP connection whose throughput collapses as a whole, not an indirect "other side notices and backs off separately" mechanism.
   - **Bottom line**: CU's real, physically-limited signal under a safe (~4.5%) loss rate is only ~4.3% on TX — genuinely too small to separate from noise. Confirmed by testing every single CU feature (cpu, mem_pct, mem_bytes, net_tx, net_rx, net_diff, net_ratio) scored on its own against a proper threshold: **every one comes back at F1≈0** at this loss magnitude. This is not a feature-selection or scoring problem — the signal is fundamentally too small at the currently-used loss rate.

## Approaches tried

### A. Switch `netem loss` → `netem rate` (bandwidth cap) — TECHNICALLY VALIDATED BUT REJECTED BY TEAM, DO NOT PURSUE FURTHER
- Mechanism: `injectStress()`'s NET branch measures the container's own current TX rate over a window (started at 2s, found to be too short — caused real RRC Release disconnects on the busiest CUs — widened to 10s), then applies `tc qdisc replace dev eth0 root netem rate {X}kbit` at a fixed fraction of that measured baseline.
- Extensively live-smoke-tested (before/during-5min/after methodology, checking RRC Release in UE logs) at both 50% cut (worked on cu0/cu1, broke cu2/cu3 via RRC Release) and 20% cut (fully validated safe on all 4 CUs including the two that broke at 50%, with their exact affected UEs re-checked clean).
- Full recollection + honest evaluation at 20% cut: **F1 went from ~0.00–0.014 (loss-based) to 0.58–0.93 (rate-cap-based)** across all 4 topologies — a real, substantial, non-circular improvement.
- **However: explicitly rejected by the user's senior/team as not fitting the Topaz framework's anomaly-detection research fidelity — bandwidth throttling is considered a different kind of event than "network stress/anomaly" (packet loss), even though it's a realistic real-world fault (link congestion, backup-link failover, rain fade on microwave backhaul).**
- **Current code state**: `scrapper_prometheus_backup.py` on the server currently HAS this rate-cap change applied (10s window, 20% cut). This needs to be reverted back to `netem loss` for any future work, since the team has ruled out this direction.

### B. Increase the `netem loss` percentage (proposed, NOT yet tested)
- Since UDP's effect is roughly proportional to loss rate (no compensation), a higher loss percentage (e.g. 15-30%) should proportionally scale up CU's TX-side signal without switching away from genuine packet loss.
- A live smoke-test script (same before/during-5min/after + RRC-log-check methodology used to validate the rate-cap) was drafted but not yet run. This is the main open, unexplored lever for fixing `netem loss` detection while staying at "packet loss" as the fault type.

### C. Additional/unused metrics — CHECKED THOROUGHLY, CONFIRMED DEAD, NOT USABLE WITHOUT A PIPELINE CHANGE
- `new_allPromQuery.txt` (the actual query file used) already includes queries for `srscu_pdcp_tx_drop_packets_total`, `srscu_f1u_dl_loss_rate`, `srscu_pdcp_latency_seconds` (queried against cAdvisor — which never exposes app-level metrics, so these are structurally dead), and `node_netstat_Sctp_OutSctpPacks` / TCP retrans counters (queried against node-exporter — host-level, not per-container, since containers run in their own network namespaces).
- **Verified directly, not assumed**: scanned every column tagged to each CU in both `train_normal_th3.csv` (5,000+ row sample) and all test CSVs — every one of these columns is **exactly zero, always, no exceptions**. `container_fs_reads/writes_bytes_total` and `container_network_{receive,transmit}_drop_total` are also always zero for CUs.
- The only way to get genuinely new, usable CU-side signal is patching `metrics_server/__main__.py` (`RAN/srsRAN_Project/docker/metrics_server/src/metrics_server/__main__.py`) to forward the `cu-up`/`pdcp` JSON block it currently ignores (has an inline comment "Currently we only support ue_list metric"), adding a new InfluxDB measurement + Flux query + `build_dataset.py` columns, then recollecting train+test and **retraining** (new input dimension, can't reuse frozen checkpoints). Not attempted — bigger effort, flagged as the "deep fix" if loss-percentage tuning (option B) doesn't pan out.

### D. Feature re-weighting (scoring net_ratio instead of net_diff/max-pool) — TESTED, DOES NOT HELP
- Checked directly on real reconstruction-error data: every CU feature scored standalone gives F1≈0 at the current loss magnitude. This rules out a scoring/feature-selection fix — confirms the problem is signal magnitude, not which feature is used.

## Data quality checks completed (all files in `data_input/test_csv/`, all confirmed via direct row-level inspection)

- Systematic zero-value scan across all 16 CU test files (`test_srscu{0-3}_{CPU,MEM,NET}_rep{1,2}.csv` as applicable) and `train_normal_th3.csv` (full 179,494 rows via chunked read).
- **Train data confirmed clean**: 0 stress-labeled rows anywhere (as expected/designed), `net_rx` never zero, `net_tx` zero only 11–38 times out of 179,494 rows per CU (<0.02%, negligible, consistent with normal scrape-glitch background rate).
- **Test data mostly clean, with 2 known-bad files** (both explained, not mysterious):
  - `cu2_NET_rep2`: 213/252 stress rows (84.5%) have `tx=0` — caused by the old 2-second baseline-sampling bug in the rate-cap code (miscalibrated an overly aggressive cap); fixed by widening to 10s (see section A). Not relevant if reverting to `netem loss`.
  - `cu3_NET_rep1`: 45/253 stress rows (17.8%) have **both** `tx=0` AND `rx=0` simultaneously — confirmed via the collection log as a real RRC Release event (`srsue6/7/8/9` disconnected at t≈205s into the 300s stress window). This happened under the rate-cap approach at 50% cut; not yet known whether it would recur under a higher `netem loss` percentage (worth checking if pursuing option B).

## Per-CU/per-feature diagnostic plots

Generated in `data_input/test_csv/`: `srscu{0-3}_{cpu,mem_pct,mem_bytes,net_tx,net_rx}.png` — each shows that CU's value over elapsed time across all 4 of its collected sessions (CPU_rep1, MEM_rep1, NET_rep1, NET_rep2), quiet in blue / stress in red, session boundaries marked. Useful for visually spot-checking any future collection.

## Immediate next step recommended (not yet executed)

Run live smoke tests (same methodology as the rate-cap validation: before/during-5min/after TX byte-rate checks + RRC Release log checks on the specific affected UEs) at progressively higher `netem loss` percentages (e.g. start at 15-20%) on all 4 CUs, especially `cu2` and `cu3` (the two that broke under aggressive rate-capping) to find the highest loss percentage that (a) stays safely connected (no RRC Release) and (b) produces a large enough TX-side effect to be reliably detectable — then do a full recollection and honest (gating-disabled) evaluation at that rate.

## Key file locations

- `topaz/topaz/experimental_phase0/scrapper_prometheus_backup.py` — main collection script; currently has the rate-cap change applied (`keep_frac=0.8`, 10s baseline window) — **needs reverting to `netem loss` for future work**, and the `# TEMP` CU-only/NET-only restrictions in `TARGET_CONTAINERS`/`STRESS_TYPES_THIS_RUN` need reverting to the full container list / `[1,2,3]` before any future full (non-CU-NET-only) collection.
- `topaz/topaz/experimental_phase0/test_traffic.py` — traffic generator, unmodified (variable 25-window schedule kept as-is per user's decision).
- `topaz/topaz/RAN/srsRAN_Project/docker/metrics_server/src/metrics_server/__main__.py` — where the PDCP-metrics forwarding fix would go if pursuing option C.
- `clear_pipeline/build_dataset.py`, `build_newdata_test.py`, `run_experiment_newdata.py` — ML pipeline, all on the local analysis machine, not the server.
- `clear_pipeline/data_input/test_csv/` — active test data directory; `clear_pipeline/data_input/test_csv_2_reps/` (loss-based backup) and `test_csv_2_reps_ratecap/` (rate-cap backup) hold copies of both prior collection rounds.
