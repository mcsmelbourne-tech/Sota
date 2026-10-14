#!/bin/zsh
# Counterfactual grid over the recorded LLM trades: family vetoes x exit rules, all from cache.
# Usage: scripts/replay_grid.sh  -> data/bt/grid/<variant>_<year>.log and a summary table.
set -u
cd "$(dirname "$0")/.."
mkdir -p data/bt/grid

typeset -A years
years=(y0 "2023-09-01 2024-08-30" y1 "2024-09-03 2025-08-29" holdout "2025-10-13 2026-10-09")

ALL='["ol","dv","cv","dg","ib","ic","bf"]'
NOIC='["ol","dv","cv","dg","ib","bf"]'

# variant name | families | stop_loss_frac
variants=(
  "base|$ALL|0.6"
  "noic|$NOIC|0.6"
  "stop1.0|$ALL|1.0"
  "stop1.5|$ALL|1.5"
  "nostop|$ALL|100"
  "noic_nostop|$NOIC|100"
  "noic_stop1.0|$NOIC|1.0"
)

for v in $variants; do
  name=${v%%|*}; rest=${v#*|}; fams=${rest%%|*}; stop=${rest#*|}
  for y in y0 y1 holdout; do
    SOTA_LIMITS__FAMILIES="$fams" SOTA_LIMITS__STOP_LOSS_FRAC="$stop" COLUMNS=200 \
      uv run sota backtest ${=years[$y]} -s "replay:data/bt/$y/llm_proposals.json" \
      > "data/bt/grid/${name}_${y}.log" 2>&1 &
  done
  wait
  printf "%-14s" "$name"
  for y in y0 y1 holdout; do
    row=$(grep -E '│ repl' "data/bt/grid/${name}_${y}.log" | head -1)
    tr=$(echo "$row" | awk -F'│' '{gsub(/ /,"",$3); print $3}')
    mdd=$(echo "$row" | awk -F'│' '{gsub(/ /,"",$8); print $8}')
    printf "  %s: TR %-8s MDD %-8s" "$y" "$tr" "$mdd"
  done
  echo
done
