#!/usr/bin/env bash
# Launch Phase 4 matched-protocol Pareto sweeps in 4 tmux sessions.
#
# Sessions:
#   k2_ext  — S1+DisCo k=2, lambda=5..20  (skips existing best_metrics.json)
#   k3_full — S1+DisCo k=3, lambda=0..15
#   k2_tail — waits for k2 lambda=20, then adds lambda=30,50
#   plot    — waits for k2+k3 to finish, then generates overlay plot
#
# Usage (from anywhere):
#   bash /scope-vol/mass_perp_classifier/scripts/launch_phase4_tmux.sh
#
# Re-running is safe: the sweep script skips lambdas that already have
# best_metrics.json, so interrupted sessions resume from where they stopped.
#
# Attach to a session:
#   tmux attach -t k2_ext
# List sessions:
#   tmux ls

set -e
PROJ=/scope-vol/mass_perp_classifier
SVD=/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz

cd "$PROJ"

tmux new-session -d -s k2_ext \
  "python scripts/run_cure_disco_pareto_sweep.py \
  --save-dir runs/cure_disco_pareto__S1_massshift_ep40_k2 \
  --svd-forget $SVD \
  --lambdas 5 6 7 8 9 10 12 15 20 \
  -- --svd-k 2 \
  2>&1 | tee runs/k2_ext.log"

tmux new-session -d -s k3_full \
  "python scripts/run_cure_disco_pareto_sweep.py \
  --save-dir runs/cure_disco_pareto__S1_massshift_ep40_k3 \
  --svd-forget $SVD \
  --lambdas 0 1 2 3 4 5 6 7 8 9 10 12 15 \
  -- --svd-k 3 \
  2>&1 | tee runs/k3_full.log"

tmux new-session -d -s k2_tail \
  "until [ -f runs/cure_disco_pareto__S1_massshift_ep40_k2/lambda_20.0/best_metrics.json ]; \
  do sleep 60; done && \
  python scripts/run_cure_disco_pareto_sweep.py \
  --save-dir runs/cure_disco_pareto__S1_massshift_ep40_k2 \
  --svd-forget $SVD \
  --lambdas 30 50 \
  -- --svd-k 2 \
  2>&1 | tee runs/k2_tail.log"

tmux new-session -d -s plot \
  "until [ -f runs/cure_disco_pareto__S1_massshift_ep40_k2/lambda_20.0/best_metrics.json ] && \
        [ -f runs/cure_disco_pareto__S1_massshift_ep40_k3/lambda_15.0/best_metrics.json ]; \
  do sleep 120; done && \
  python scripts/run_cure_disco_pareto_sweep.py \
    --save-dir runs/cure_disco_pareto__S1_massshift_ep40_k2 --plot-only && \
  python scripts/run_cure_disco_pareto_sweep.py \
    --save-dir runs/cure_disco_pareto__S1_massshift_ep40_k3 --plot-only && \
  python scripts/plot_pareto_overlay.py \
  --run 'DisCo-only pretrained:runs/disco_pareto_pretrained/summary.csv' \
  --run 'S1+DisCo k=1 ep40:runs/cure_disco_pareto__S1_massshift_ep40_k1/summary.csv' \
  --run 'S1+DisCo k=2 ep40:runs/cure_disco_pareto__S1_massshift_ep40_k2/summary.csv' \
  --run 'S1+DisCo k=3 ep40:runs/cure_disco_pareto__S1_massshift_ep40_k3/summary.csv' \
  --out runs/pareto_phase4_ep40_final.png \
  --max-jsd 0.024 \
  2>&1 | tee runs/plot.log"

echo "Launched 4 tmux sessions:"
tmux ls
