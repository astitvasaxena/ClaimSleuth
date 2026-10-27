"""
Pytest session setup.

Redirects the audit ledger to a temporary directory BEFORE any project module
is imported, so running the suite never modifies the git-tracked logs in
data/audit_logs/. Using an environment variable (rather than patching the
module attribute) keeps the redirect in place even when a test reloads
configs.settings.
"""
import os
import sys
import tempfile
from pathlib import Path

_AUDIT_TMP = tempfile.mkdtemp(prefix="claimsleuth_test_audit_")
os.environ["CLAIMSLEUTH_AUDIT_LOG_DIR"] = _AUDIT_TMP
os.environ["CLAIMSLEUTH_REPORTS_DIR"] = tempfile.mkdtemp(prefix="claimsleuth_test_reports_")
os.environ["CLAIMSLEUTH_TRACES_DIR"] = tempfile.mkdtemp(prefix="claimsleuth_test_traces_")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tests/crew_service needs crewai, which lives only in .venv-crewai; tests/test_crewai_a2a.py
# runs it there as a subprocess, so the main-environment collection skips it.
collect_ignore_glob = ["crew_service/*"]
