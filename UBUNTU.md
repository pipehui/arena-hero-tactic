# Arena Hero on Ubuntu

The Python tactic is portable to Ubuntu. The Windows PowerShell watchdog and
Task Scheduler wrappers are replaced here by systemd.

## VPS requirements

- Ubuntu 24.04 LTS (Python 3.12 by default) is the recommended baseline.
- 1 shared vCPU and 1 GB RAM are sufficient for this single process; 2 GB RAM
  gives more room for package upgrades and diagnostics.
- Use at least 10 GB of disk. The current detailed replay format has produced
  roughly 330 MB/day, so this deployment restarts daily and retains only the
  previous 48 hours of session JSONL files.
- No GPU is needed.
- The tactic needs DNS plus outbound HTTPS/WebSocket access on TCP 443 to
  `api.arenahero.io`. It does not listen on an inbound application port.

A dedicated public IPv4 address is not required by the tactic. The server does
need working outbound Internet access through IPv4, IPv6 or provider NAT. A
public address is mainly useful for SSH administration; without one, use the
provider console, a bastion, VPN or an outbound management overlay.

## Install

Clone the repository on the VPS, then run:

```bash
cd /path/to/Arena
sudo ARENA_HERO_SERVICE_USER="$USER" bash deploy/ubuntu/install.sh
sudoedit /etc/arena-hero/arena-hero.env
sudo systemctl start arena-hero.service
```

The installer creates an isolated virtual environment under
`/var/lib/arena-hero/venv`; it never installs into the system or Conda base
environment. It installs and enables two timers, but deliberately does **not**
enable the tactic at boot.

Useful commands:

```bash
sudo systemctl start arena-hero.service
sudo systemctl stop arena-hero.service
sudo systemctl restart arena-hero.service
systemctl status arena-hero.service
journalctl -u arena-hero.service -f
systemctl list-timers 'arena-hero-*'
```

While the service is active, systemd restarts it after an unexpected exit. A
one-minute health check also restarts a live process whose heartbeat is older
than four minutes. An explicit `systemctl stop` remains stopped.

At 04:15 in the VPS local timezone, the maintenance timer:

1. uses `try-restart`, so it restarts only an already-active tactic;
2. removes only `arena_hero_*.jsonl` files older than 48 hours.

It never deletes `balanced_tactic_memory.json`, the process lock or heartbeat.
Change `OnCalendar` in
`deploy/ubuntu/arena-hero-maintenance.timer` if another maintenance time is
preferred, then rerun the installer.

## Migrate the current state

Never run the laptop and VPS tactics simultaneously for the same API key. Stop
the laptop service before starting the VPS.

Code is best transferred through GitHub. Runtime state and credentials are not:

- push the source repository after checking `git status --ignored`;
- keep `key.txt`, `.env`, `logs/`, virtual environments and caches ignored;
- enter the API key only in `/etc/arena-hero/arena-hero.env` on the VPS;
- if a key is ever committed, rotate it even if the commit is later deleted.

This directory is not currently a Git repository. To publish it, create an
empty public repository on GitHub, then run locally:

```bash
git init
git add .
git status --short --ignored
git commit -m "Initial Arena Hero tactic"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/YOUR_REPOSITORY.git
git push -u origin main
```

Before committing, confirm that `key.txt`, `logs/`, `.env` and `.venv/` appear
only as ignored (`!!`) entries. Use Git CLI rather than uploading the entire
folder through the website, because the CLI applies `.gitignore` before files
are staged.

To preserve explored-map memory, separately copy
`logs/balanced_tactic_memory.json` and optionally the newest session JSONL to a
temporary directory on the VPS, then install them with the service account as
owner:

```bash
sudo install -o "$USER" -g "$(id -gn)" -m 0600 \
  /tmp/balanced_tactic_memory.json \
  /var/lib/arena-hero/logs/balanced_tactic_memory.json
```

Starting without this checkpoint is valid, but the tactic will rebuild its map
and threat memory from current visibility.

## Update

Stop, update and reinstall so dependency or unit-file changes are applied:

```bash
sudo systemctl stop arena-hero.service
git pull --ff-only
sudo ARENA_HERO_SERVICE_USER="$USER" bash deploy/ubuntu/install.sh
sudo systemctl start arena-hero.service
```

GitHub Actions runs the test suite on Ubuntu 24.04 with Python 3.12 for every
push and pull request.
