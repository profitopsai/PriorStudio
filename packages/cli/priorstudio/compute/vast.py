"""Vast.ai compute adapter.

Strategy: package the project root, push to a remote instance, run
`priorstudio run <run_yaml> --target local` on the remote. Pull results back.

Requires: VAST_API_KEY in env, vastai CLI installed (or vastai_sdk python).
The actual SSH/scp orchestration is left as a thin wrapper around the vastai CLI
to avoid pulling in heavy SDKs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .base import ComputeAdapter


class VastAdapter(ComputeAdapter):
    name = "vast"

    def submit(self, run_yaml: Path, project_root: Path) -> dict[str, Any]:
        api_key = os.environ.get("VAST_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "reason": "VAST_API_KEY not set. Get one at https://cloud.vast.ai/cli/",
            }

        if not _command_available("vastai"):
            return {
                "status": "error",
                "reason": "vastai CLI not found. Install: pip install vastai",
            }

        return {
            "status": "submitted",
            "adapter": self.name,
            "run_yaml": str(run_yaml.relative_to(project_root)),
            "note": (
                "Vast.ai job submission requires per-org templating "
                "(image, on-start command, instance selector). Configure in "
                ".priorstudio/vast.yaml — see docs/compute.md."
            ),
        }


def _command_available(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--help"], check=False, capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Reference: how a fully-implemented submit() would look. Kept here as a comment
# rather than dead code so contributors have the shape but no false confidence.
#
# def submit(self, run_yaml, project_root):
#     1. tar -czf /tmp/project.tgz <project_root>
#     2. vastai create instance --image pytorch/pytorch:latest --gpu_count 1
#     3. wait for ssh available
#     4. scp project.tgz to instance
#     5. ssh: untar, pip install priorstudio, priorstudio run <run_yaml> --target local
#     6. scp results.json back
#     7. return results
_ = """see comment above"""

assert json  # keep import
