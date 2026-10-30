"""
Smoke test for the MCP server, run as a separate client process against a running app:

    python eval/mcp_smoke.py --url http://127.0.0.1:8000/mcp

Checks: exactly six tools are listed; each one is called with real demo data and succeeds; a request
without the shared secret gets 401; a scanned PDF (no text layer) returns the "no text layer" error.
The scanned-PDF check adds a small claim (SMOKE-SCAN) to the local database the app uses, so run it
on the same machine as the app.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from pypdf import PdfWriter

from claimsleuth import config, db

EXPECTED = {"parse_document", "search_policy", "check_coverage", "compute_payout", "claim_history",
            "check_invoice_totals"}
SCAN_CLAIM = "SMOKE-SCAN"


def add_scanned_pdf_claim() -> str:
    """A claim whose only document is a PDF page with no text, like a scan."""
    db.init()
    folder = Path(config.DATA_DIR) / "uploads" / SCAN_CLAIM
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / "scanned_invoice.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with pdf.open("wb") as f:
        writer.write(f)
    with db.connect() as conn:
        if not conn.execute("SELECT 1 FROM claims WHERE id = ?", (SCAN_CLAIM,)).fetchone():
            db.insert_claim(conn, {"id": SCAN_CLAIM, "source": "eval", "claimant": "Smoke Test", "device": "Phone",
                                   "incident_date": "2024-01-01", "statement": "Scanned invoice only"})
            db.insert_document(conn, SCAN_CLAIM, pdf, f"uploads/{SCAN_CLAIM}/{pdf.name}", kind="invoice")
    return f"{SCAN_CLAIM}/{pdf.name}"


def text_of(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


async def run(url: str, secret: str) -> list[str]:
    failures = []
    scanned_doc = add_scanned_pdf_claim()
    calls = {
        "parse_document": {"doc_id": "CLM-2024-003/invoice_003.txt"},
        "search_policy": {"claim_id": "CLM-2024-001", "query": "What deductible applies to a repair claim?"},
        "check_coverage": {"purchase_date": "2023-06-01", "incident_date": "2024-08-10", "warranty_months": 36,
                           "exclusions": ["commercial use", "pest infestation"], "cause": "compressor failure"},
        "compute_payout": {"repair_cost": "650.00", "deductible": "75.00", "coverage_limit": "800.00",
                           "depreciation_pct": "5"},
        "claim_history": {"serial": "REF-HC-99400-B", "claimant": "David Sterling", "claim_id": "CLM-2024-003"},
        "check_invoice_totals": {"line_items": ["120.00", "80.00", "40.00"], "stated_total": "390.00"},
    }
    async with create_mcp_http_client(headers={"X-ClaimSleuth-Secret": secret}) as client:
        async with streamable_http_client(url, http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                print(f"tools listed: {sorted(names)}")
                if names != EXPECTED:
                    failures.append(f"expected exactly {sorted(EXPECTED)}, got {sorted(names)}")
                for name, args in calls.items():
                    result = await session.call_tool(name, args)
                    shown = json.dumps(result.structured_content)[:160]
                    print(f"{name}: {'ERROR ' + text_of(result) if result.is_error else 'ok ' + shown}")
                    if result.is_error:
                        failures.append(f"{name} failed: {text_of(result)}")
                scan = await session.call_tool("parse_document", {"doc_id": scanned_doc})
                print(f"scanned PDF: is_error={scan.is_error} {text_of(scan)}")
                if not (scan.is_error and "no text layer" in text_of(scan)):
                    failures.append("scanned PDF did not return the 'no text layer' error")
    async with httpx.AsyncClient() as plain:
        response = await plain.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                    headers={"Accept": "application/json, text/event-stream"})
        print(f"without secret: HTTP {response.status_code}")
        if response.status_code != 401:
            failures.append(f"request without the secret got {response.status_code}, expected 401")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    args = parser.parse_args()
    if not config.A2A_SHARED_SECRET:
        sys.exit("A2A_SHARED_SECRET is not set")
    problems = asyncio.run(run(args.url, config.A2A_SHARED_SECRET))
    print("PASS" if not problems else "FAIL:\n- " + "\n- ".join(problems))
    sys.exit(1 if problems else 0)
