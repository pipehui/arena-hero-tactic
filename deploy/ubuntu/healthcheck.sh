#!/usr/bin/env bash
set -euo pipefail

service_name="${1:-arena-hero.service}"
heartbeat_path="${2:-/var/lib/arena-hero/logs/watchdog/tactic_heartbeat.json}"
timeout_seconds="${3:-240}"

if ! systemctl is-active --quiet "$service_name"; then
    exit 0
fi

now_epoch="$(date +%s)"
uptime_seconds="${SECONDS}"
if [[ -r /proc/uptime ]]; then
    uptime_seconds="$(cut -d. -f1 </proc/uptime)"
fi
active_usec="$(systemctl show "$service_name" --property=ActiveEnterTimestampMonotonic --value)"
active_seconds="$((active_usec / 1000000))"
service_age="$((uptime_seconds - active_seconds))"

if [[ -f "$heartbeat_path" ]]; then
    heartbeat_epoch="$(stat -c %Y "$heartbeat_path")"
    heartbeat_age="$((now_epoch - heartbeat_epoch))"
    if (( heartbeat_age <= timeout_seconds )); then
        exit 0
    fi
    reason="heartbeat_age=${heartbeat_age}s"
elif (( service_age <= timeout_seconds )); then
    # Give a newly started process time to connect and write its first heartbeat.
    exit 0
else
    reason="heartbeat_missing service_age=${service_age}s"
fi

logger --tag arena-hero-healthcheck "$reason action=restart"
systemctl restart "$service_name"

