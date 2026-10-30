"""SQLite storage: claims, documents, reports, reviews, runs, per-agent metrics and the claim history seed."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from claimsleuth import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY, source TEXT NOT NULL CHECK (source IN ('demo', 'upload', 'eval')),
    claimant TEXT NOT NULL, device TEXT NOT NULL, serial TEXT, purchase_date TEXT, incident_date TEXT NOT NULL,
    statement TEXT NOT NULL, policy_id TEXT, status TEXT NOT NULL DEFAULT 'new', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, claim_id TEXT NOT NULL REFERENCES claims(id),
    kind TEXT NOT NULL CHECK (kind IN ('invoice', 'diagnostic', 'policy', 'other')),
    filename TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, pages INTEGER NOT NULL DEFAULT 1,
    has_text INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS reports (
    claim_id TEXT NOT NULL REFERENCES claims(id), run_id TEXT NOT NULL, report_json TEXT NOT NULL,
    report_md TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
    claim_id TEXT NOT NULL REFERENCES claims(id), run_id TEXT NOT NULL, reviewer TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('approve', 'reject')), note TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, claim_id TEXT NOT NULL REFERENCES claims(id), pattern TEXT NOT NULL,
    prompt_style TEXT NOT NULL, model TEXT NOT NULL, started_at TEXT NOT NULL, seconds REAL,
    prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0,
    status TEXT NOT NULL, error TEXT);
CREATE TABLE IF NOT EXISTS agent_metrics (
    run_id TEXT NOT NULL REFERENCES runs(id), agent TEXT NOT NULL, seconds REAL DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0, llm_calls INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0, errors INTEGER DEFAULT 0, guardrail_hits INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, agent));
CREATE TABLE IF NOT EXISTS past_claims (
    serial TEXT NOT NULL, claimant TEXT NOT NULL, claim_date TEXT NOT NULL, outcome TEXT NOT NULL, amount REAL NOT NULL);
"""

PAST_CLAIMS_SEED = config.BASE_DIR / "data" / "past_claims.json"


def db_path() -> Path:
    return Path(config.DATA_DIR) / "claimsleuth.db"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_document_path(stored: str) -> Path:
    """Uploads are stored relative to DATA_DIR, demo and eval documents relative to the project folder."""
    p = Path(stored)
    if p.is_absolute():
        return p
    return (Path(config.DATA_DIR) / p) if stored.startswith("uploads/") else (config.BASE_DIR / p)


def document_kind(filename: str) -> str:
    name = filename.lower()
    if "invoice" in name or "receipt" in name:
        return "invoice"
    if "diag" in name or "estimate" in name or "repair" in name or "technician" in name:
        return "diagnostic"
    if "policy" in name:
        return "policy"
    return "other"


def insert_claim(conn: sqlite3.Connection, claim: dict) -> None:
    conn.execute(
        "INSERT INTO claims (id, source, claimant, device, serial, purchase_date, incident_date, statement,"
        " policy_id, status, created_at) VALUES (:id, :source, :claimant, :device, :serial, :purchase_date,"
        " :incident_date, :statement, :policy_id, :status, :created_at)",
        {"serial": None, "purchase_date": None, "policy_id": None, "status": "new", "created_at": now(), **claim})


def insert_document(conn: sqlite3.Connection, claim_id: str, file_path: Path, stored_path: str,
                    kind: Optional[str] = None, pages: int = 1, has_text: bool = True) -> str:
    doc_id = f"{claim_id}/{file_path.name}"
    conn.execute(
        "INSERT INTO documents (id, claim_id, kind, filename, path, sha256, pages, has_text)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (doc_id, claim_id, kind or document_kind(file_path.name), file_path.name, stored_path,
         sha256_of(file_path), pages, int(has_text)))
    return doc_id


def seed_folder_claims(conn: sqlite3.Connection, folder: Path, source: str) -> int:
    """Load claim packets (claim.json + documents) from a folder of folders, skipping claims already present."""
    added = 0
    if not folder.is_dir():
        return 0
    for packet in sorted(p for p in folder.iterdir() if (p / "claim.json").is_file()):
        meta = json.loads((packet / "claim.json").read_text(encoding="utf-8"))
        if conn.execute("SELECT 1 FROM claims WHERE id = ?", (meta["claim_id"],)).fetchone():
            continue
        insert_claim(conn, {"id": meta["claim_id"], "source": source, "claimant": meta["claimant_name"],
                            "device": meta["device_model"], "serial": meta.get("serial"),
                            "purchase_date": meta.get("purchase_date"), "incident_date": meta["incident_date"],
                            "statement": meta["reported_issue"], "policy_id": meta.get("policy_id")})
        for doc in sorted(packet.iterdir()):
            if doc.name != "claim.json" and doc.suffix.lower() in (".txt", ".md", ".pdf"):
                insert_document(conn, meta["claim_id"], doc, doc.relative_to(config.BASE_DIR).as_posix())
        added += 1
    return added


def init() -> dict:
    """Create the tables and load the demo claims and the claim-history seed. Safe to run repeatedly."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        demos = seed_folder_claims(conn, config.DEMO_CLAIMS_DIR, "demo")
        seeded = 0
        if not conn.execute("SELECT 1 FROM past_claims LIMIT 1").fetchone():
            rows = json.loads(PAST_CLAIMS_SEED.read_text(encoding="utf-8"))
            conn.executemany("INSERT INTO past_claims VALUES (:serial, :claimant, :claim_date, :outcome, :amount)", rows)
            seeded = len(rows)
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("claims", "documents", "past_claims")}
    return {"db": str(db_path()), "demos_added": demos, "past_claims_added": seeded, "counts": counts}


def get_claim(claim_id: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return dict(row) if row else None


def get_documents(claim_id: str) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM documents WHERE claim_id = ? ORDER BY id", (claim_id,))]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ClaimSleuth database")
    parser.add_argument("--init", action="store_true", help="create the tables and load the demo and seed data")
    if parser.parse_args().init:
        print(json.dumps(init(), indent=2))
