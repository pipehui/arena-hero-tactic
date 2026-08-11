#!/usr/bin/env bash
set -euo pipefail

if (( EUID != 0 )); then
    echo "Run this installer with sudo." >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_dir="$(cd -- "$script_dir/../.." && pwd)"
service_user="${ARENA_HERO_SERVICE_USER:-${SUDO_USER:-}}"
python_bin="${ARENA_HERO_PYTHON:-python3}"
state_root="${ARENA_HERO_STATE_ROOT:-/var/lib/arena-hero}"
log_dir="$state_root/logs"
venv_dir="$state_root/venv"
environment_dir="/etc/arena-hero"
environment_file="$environment_dir/arena-hero.env"

if [[ -z "$service_user" || "$service_user" == root ]]; then
    echo "Set ARENA_HERO_SERVICE_USER to the non-root account that should run the tactic." >&2
    exit 1
fi
if ! id "$service_user" >/dev/null 2>&1; then
    echo "Unknown service user: $service_user" >&2
    exit 1
fi
if [[ ! "$app_dir" =~ ^[A-Za-z0-9_./-]+$ || ! "$state_root" =~ ^[A-Za-z0-9_./-]+$ ]]; then
    echo "Application and state paths may contain only letters, digits, _, ., / and -." >&2
    exit 1
fi
if [[ ! -f "$app_dir/balanced_tactic.py" || ! -f "$app_dir/requirements.txt" ]]; then
    echo "Run the installer from a complete Arena Hero project checkout." >&2
    exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi

"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Arena Hero requires Python 3.11 or newer")
PY

service_group="$(id -gn "$service_user")"
install -d -m 0750 -o "$service_user" -g "$service_group" "$state_root" "$log_dir"
install -d -m 0700 -o root -g root "$environment_dir"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    rm -rf -- "$venv_dir"
    install -d -m 0750 -o "$service_user" -g "$service_group" "$venv_dir"
    if ! runuser -u "$service_user" -- "$python_bin" -m venv "$venv_dir"; then
        if [[ "$python_bin" != python3 ]]; then
            echo "Could not create the virtual environment with $python_bin." >&2
            exit 1
        fi
        apt-get update
        apt-get install -y python3-venv ca-certificates
        runuser -u "$service_user" -- "$python_bin" -m venv "$venv_dir"
    fi
fi
runuser -u "$service_user" -- "$venv_dir/bin/python" -m pip install --disable-pip-version-check -r "$app_dir/requirements.txt"

if [[ ! -f "$environment_file" ]]; then
    sed "s|/var/lib/arena-hero/logs|$log_dir|g" \
        "$script_dir/arena-hero.env.example" >"$environment_file"
    chown root:root "$environment_file"
    chmod 0600 "$environment_file"
fi

render_unit() {
    local source="$1"
    local destination="$2"
    sed \
        -e "s|@@APP_DIR@@|$app_dir|g" \
        -e "s|@@SERVICE_USER@@|$service_user|g" \
        -e "s|@@SERVICE_GROUP@@|$service_group|g" \
        -e "s|@@PYTHON@@|$venv_dir/bin/python|g" \
        -e "s|@@LOG_DIR@@|$log_dir|g" \
        "$source" >"$destination"
    chmod 0644 "$destination"
}

render_unit "$script_dir/arena-hero.service.in" /etc/systemd/system/arena-hero.service
render_unit "$script_dir/arena-hero-healthcheck.service.in" /etc/systemd/system/arena-hero-healthcheck.service
render_unit "$script_dir/arena-hero-maintenance.service.in" /etc/systemd/system/arena-hero-maintenance.service
install -m 0644 "$script_dir/arena-hero-healthcheck.timer" /etc/systemd/system/arena-hero-healthcheck.timer
install -m 0644 "$script_dir/arena-hero-maintenance.timer" /etc/systemd/system/arena-hero-maintenance.timer

systemctl daemon-reload
systemctl disable arena-hero.service >/dev/null 2>&1 || true
systemctl enable --now arena-hero-healthcheck.timer arena-hero-maintenance.timer

cat <<EOF
Ubuntu service files installed.

1. Edit the root-only credential file:
   sudoedit $environment_file
2. Start the on-demand tactic:
   sudo systemctl start arena-hero.service
3. Inspect it:
   systemctl status arena-hero.service
   journalctl -u arena-hero.service -f

The tactic itself is intentionally not enabled at boot. The timers never start
an inactive tactic; they only health-check/restart it while active and delete
session JSONL logs older than 48 hours.
EOF
