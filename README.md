# Arena Hero Balanced Tactic

An autonomous Python tactic for Arena Hero, built against Gameplay v0.14 and
the official Python SDK 0.2.9. The project includes the tactic engine, replay
logging, Windows launch/watchdog scripts, and an Ubuntu systemd deployment.

## Requirements

- Python 3.11 or newer
- `arena-hero==0.2.9`
- An Arena Hero API key

Install dependencies in an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, the existing PowerShell scripts can use the dedicated Conda
environment. On Ubuntu, follow [UBUNTU.md](UBUNTU.md) for the systemd service,
health monitoring, daily restart, and 48-hour log retention setup.

## Run

Provide the API key through the deployment-specific secret file or environment
configuration, then start the appropriate launcher. Never commit the key,
runtime logs, virtual environments, or exploration checkpoints.

The repository's `.gitignore` excludes these local and sensitive artifacts.

## Project layout

- `balanced_tactic.py`: compatible public entry point.
- `arena_tactic/`: decision engine, world model, economy, combat, and movement.
- `replay_log.py`: JSONL replay logging.
- `tests/`: strategy and runtime contract tests.
- `deploy/ubuntu/`: systemd installer, services, timers, and health checks.
- `ARCHITECTURE.md`: internal architecture overview.
- `UBUNTU.md`: VPS sizing, migration, operation, and maintenance guide.

## Verify

```bash
python -m unittest discover -s tests -v
python -m compileall balanced_tactic.py replay_log.py arena_tactic tests
python -m pip check
```

GitHub Actions runs the same checks on Ubuntu.
