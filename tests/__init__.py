"""
Test package setup shared by pytest and `python -m unittest`.

Redirects the audit ledger to a temporary directory before project modules are
imported, so no test runner modifies the git-tracked logs in data/audit_logs/.
"""
import os
import tempfile

os.environ.setdefault("CLAIMSLEUTH_AUDIT_LOG_DIR", tempfile.mkdtemp(prefix="claimsleuth_test_audit_"))
os.environ.setdefault("CLAIMSLEUTH_REPORTS_DIR", tempfile.mkdtemp(prefix="claimsleuth_test_reports_"))
os.environ.setdefault("CLAIMSLEUTH_TRACES_DIR", tempfile.mkdtemp(prefix="claimsleuth_test_traces_"))
