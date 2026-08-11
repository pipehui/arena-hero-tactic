#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ARENA_HERO_API_KEY:-}" || "$ARENA_HERO_API_KEY" == "replace-with-your-api-key" ]]; then
    echo "ARENA_HERO_API_KEY is missing or still contains the example value." >&2
    exit 78
fi
if [[ -z "${ARENA_HERO_LOG_DIR:-}" ]]; then
    echo "ARENA_HERO_LOG_DIR is missing." >&2
    exit 78
fi
