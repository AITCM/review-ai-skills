#!/usr/bin/env python3
"""Compatibility entry point for draft-source LLM adjudication.

The implementation lives in `citation_llm_adjudicator.py`; this file exposes
the clearer workflow name used in the skill documentation.
"""

from __future__ import annotations

from citation_llm_adjudicator import main


if __name__ == "__main__":
    raise SystemExit(main())
