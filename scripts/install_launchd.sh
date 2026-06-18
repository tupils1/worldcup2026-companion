#!/usr/bin/env bash
# Install (or reinstall) the daily-refresh launchd agent with REAL paths.
#
# The repo's com.worldcup.dailyrefresh.plist ships with placeholder paths
# (/path/to/worldcup) so it's safe to open-source. NEVER `cp` that template
# straight into ~/Library/LaunchAgents — launchd then can't find the script and
# fails the job with exit 78 (EX_CONFIG). This script substitutes the real repo
# path and (re)loads the agent. Idempotent — run it again to pick up plist edits.
#
#     bash scripts/install_launchd.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/scripts/com.worldcup.dailyrefresh.plist"
LABEL="com.worldcup.dailyrefresh"
DST="$HOME/Library/LaunchAgents/$LABEL.plist"

[ -f "$SRC" ] || { echo "missing template: $SRC" >&2; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents"
sed "s|/path/to/worldcup|$ROOT|g" "$SRC" > "$DST"
echo "wrote $DST (root: $ROOT)"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$DST"

echo "--- loaded; verify below (Hour should be the schedule, last exit not 78) ---"
launchctl print "gui/$(id -u)/$LABEL" 2>&1 | grep -iE '"Hour"|"Minute"|last exit code' || true
echo "Tip: 'launchctl kickstart gui/$(id -u)/$LABEL' runs it once now to test."
