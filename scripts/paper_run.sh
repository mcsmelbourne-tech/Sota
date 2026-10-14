#!/bin/zsh
# Daily paper-trading cycle, fired by launchd (see launchd/com.sota.paper.plist).
#
# launchd schedules in the Mac's local time, but the decision must happen at 15:50 America/New_York,
# which drifts by an hour against a fixed-offset zone (AST) across US daylight-saving changes. The plist
# therefore fires at both 15:45 and 16:45 local; whichever firing lands inside the ET window waits for
# 15:50 ET and runs, the other exits. A per-day marker guarantees one run per trading day.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="/opt/anaconda3/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

mkdir -p data/paper
today=$(TZ=America/New_York date +%F)
log="data/paper/run_${today}.log"
marker="data/paper/.ran_${today}"
[[ -e $marker ]] && exit 0

# seconds until 15:50 ET; negative when past
wait=$(.venv/bin/python - <<'EOF'
from datetime import datetime
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo("America/New_York"))
print(int((now.replace(hour=15, minute=50, second=0, microsecond=0) - now).total_seconds()))
EOF
)
if (( wait > 1800 || wait < -600 )); then
  echo "$(date '+%F %T %Z') outside the 15:50 ET window (${wait}s away); not this firing" >> "$log"
  exit 0
fi
(( wait > 0 )) && sleep "$wait"

touch "$marker"
{
  echo "=== $(date '+%F %T %Z') sota paper run"
  uv run sota paper run
  echo "=== exit $?"
} >> "$log" 2>&1
