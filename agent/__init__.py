"""Agent-side evaluations for LCLM.

This package implements the agentic information-retrieval experiments
described in §6 of the LCLM paper: the model receives a heavily
compressed context plus an ``EXPAND(i)`` tool that returns raw text for
selected segments, and is evaluated on RULER niah tasks, LongBench, and
LongHealth5.

Entry points
------------
- ``agent_ruler.py`` — RULER subtasks at one context length
- ``agent_longbench.py`` — all 21 LongBench subtasks
- ``agent_longhealth5.py`` — LongHealth5 multi-document QA
- ``agent_niah3.py`` — niah_single_3 with detailed compression-stats
- ``run_agent_modal.py`` — Modal orchestrator that spawns the above as
  parallel containers; pass ``--model-repo`` to point at any LCLM
  checkpoint on the Hub.

The shared building blocks (chunking, triage, generation, scoring) live
in ``_lib.py``. Post-processing utilities live in ``_postprocess/``.
"""

from __future__ import annotations

import os
import sys

# Resolve the repo root once so individual files don't each reimplement
# their own sys.path.insert. Any module under this package can do
#     from agent import PROJECT_ROOT
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
