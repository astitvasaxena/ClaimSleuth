"""
Checks the fixed evaluation set before any run (no model calls):

    python eval/check_eval_set.py

8 packets, each with a valid claim.json, a known policy and readable documents; at least 4 packets with PDFs;
a label for every packet; exactly 40 persona statements (8 x 5); the abusive and injector statements trip the
guardrail patterns and the honest ones do not; E7's repeat serial is in the claim-history seed.
"""
import json
import sys
from pathlib import Path

from claimsleuth import config
from claimsleuth.callbacks.guardrails import scan_injection, scan_toxicity
from claimsleuth.db import EVAL_CLAIMS_DIR, EVAL_PERSONAS_DIR, PERSONAS
from claimsleuth.mcp_server.pdf import read_lines

DECISIONS = {"approve", "reject", "request_evidence"}


def check() -> list[str]:
    problems = []
    policies = json.loads(config.POLICIES_FILE.read_text(encoding="utf-8"))
    labels = json.loads((EVAL_CLAIMS_DIR / "labels.json").read_text(encoding="utf-8"))
    packets = sorted(p for p in EVAL_CLAIMS_DIR.iterdir() if p.is_dir())
    if [p.name for p in packets] != [f"E{i}" for i in range(1, 9)]:
        problems.append(f"expected packets E1..E8, found {[p.name for p in packets]}")
    with_pdf = 0
    for packet in packets:
        meta = json.loads((packet / "claim.json").read_text(encoding="utf-8"))
        for key in ("claim_id", "policy_id", "claimant_name", "device_model", "incident_date", "reported_issue"):
            if not meta.get(key):
                problems.append(f"{packet.name}: claim.json has no {key}")
        if meta.get("policy_id") not in policies:
            problems.append(f"{packet.name}: unknown policy {meta.get('policy_id')}")
        docs = [d for d in packet.iterdir() if d.suffix in (".txt", ".pdf")]
        if not docs:
            problems.append(f"{packet.name}: no documents")
        for doc in docs:
            try:
                lines, _ = read_lines(doc)
                if not any(line.strip() for line in lines):
                    problems.append(f"{packet.name}/{doc.name}: empty")
            except Exception as exc:
                problems.append(f"{packet.name}/{doc.name}: unreadable ({exc})")
        with_pdf += any(d.suffix == ".pdf" for d in docs)
        label = labels.get(packet.name)
        if label is None:
            problems.append(f"{packet.name}: no label")
        elif not label.get("accepted") or not set(label["accepted"]) <= DECISIONS or not isinstance(label.get("fraud_expected"), bool):
            problems.append(f"{packet.name}: malformed label {label}")
    if with_pdf < 4:
        problems.append(f"only {with_pdf} packets contain PDFs; at least 4 are required")
    if set(labels) != {p.name for p in packets}:
        problems.append(f"labels for {sorted(labels)} but packets {[p.name for p in packets]}")
    statements = sorted(EVAL_PERSONAS_DIR.glob("*.txt"))
    if len(statements) != 40:
        problems.append(f"expected 40 persona statements, found {len(statements)}")
    for packet in packets:
        for persona in PERSONAS:
            path = EVAL_PERSONAS_DIR / f"{packet.name}_{persona}.txt"
            text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
            if not text:
                problems.append(f"missing or empty {path.name}")
            elif persona == "injector" and not scan_injection(text):
                problems.append(f"{path.name}: no injection pattern found")
            elif persona == "abusive" and not scan_toxicity(text):
                problems.append(f"{path.name}: no abusive language found")
            elif persona == "honest" and (scan_injection(text) or scan_toxicity(text)):
                problems.append(f"{path.name}: honest statement trips a guardrail")
    history = json.loads((EVAL_CLAIMS_DIR / "past_claims.json").read_text(encoding="utf-8"))
    e7_invoice = (EVAL_CLAIMS_DIR / "E7" / "invoice_E7.txt").read_text(encoding="utf-8")
    if not any(row["serial"] in e7_invoice for row in history):
        problems.append("E7: its serial is not in eval/claims/past_claims.json")
    return problems


if __name__ == "__main__":
    found = check()
    print("OK: 8 packets, 40 persona statements, labels complete" if not found else "FAIL:\n- " + "\n- ".join(found))
    sys.exit(1 if found else 0)
