"""
Pytest session setup: every file the app writes goes to temporary directories, set before any
project module is imported, so a test run never writes inside the repository.
"""
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="claimsleuth_test_data_")
os.environ["CLAIMSLEUTH_AUDIT_LOG_DIR"] = tempfile.mkdtemp(prefix="claimsleuth_test_audit_")
os.environ["CLAIMSLEUTH_TRACES_DIR"] = tempfile.mkdtemp(prefix="claimsleuth_test_traces_")
os.environ["CLAIMSLEUTH_RAG_INDEX_DIR"] = tempfile.mkdtemp(prefix="claimsleuth_test_rag_index_")



