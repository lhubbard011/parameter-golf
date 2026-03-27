#!/bin/bash
# Kill the running instance and log cost
cd "$(dirname "$0")"
source .env 2>/dev/null || { echo "No .env file"; exit 1; }
STATE=".instance_state"
if [[ ! -f "$STATE" ]]; then echo "No instance running."; exit 0; fi
ID=$(sed -n '1p' "$STATE")
AKEY="$LAMBDA_API_KEY"
LAUNCH=$(sed -n '3p' "$STATE")
PRICE=$(sed -n '4p' "$STATE")
ELAPSED=$(echo "($(date +%s) - $LAUNCH) / 3600" | bc -l)
COST=$(echo "$ELAPSED * $PRICE / 100" | bc -l)
printf "Terminating %s (cost: \$%.2f)...\n" "$ID" "$COST"
curl -sf -X POST -H "Authorization: Bearer $AKEY" -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"$ID\"]}" "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" >/dev/null
echo -e "$(date '+%Y-%m-%d %H:%M')\t$ID\t$(printf '%.2f' $COST)" >> cost_log.tsv
rm -f "$STATE"
echo "Done."
