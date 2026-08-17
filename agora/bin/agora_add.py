#!/usr/bin/env python3
"""agora_add.py — the onboarding CLI (thin wrapper over ``agora.cli``).

    python bin/agora_add.py which --occurrences 22 --multistep --continuity
    python bin/agora_add.py add-agent --charter charters/calendar-warden.json --yes
    python bin/agora_add.py add-skill --agent agent://eng/implementation \
        --name "Migration writer" --tier T2 --evals evals/eng/migration_v1.yaml
    python bin/agora_add.py add-runtime --name openclaw --url <url> --capabilities <list>
    python bin/agora_add.py audit
    python bin/agora_add.py rollback --agent agent://eng/implementation
"""
from __future__ import annotations

import sys

from _path import bootstrap

bootstrap()

from agora.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
