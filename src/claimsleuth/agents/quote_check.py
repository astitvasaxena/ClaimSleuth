"""Check a Quote against the document it cites (claim documents and policy wording)."""
from typing import Optional

from mcp.server.mcpserver.exceptions import ToolError

from claimsleuth.mcp_server.server import document_lines
from claimsleuth.models import Quote
from claimsleuth.quotes import quote_problem


def check_quote(quote: Quote, cache: Optional[dict] = None) -> Optional[str]:
    """None if the quote's text is at its line of its document, else the reason it is not."""
    cache = {} if cache is None else cache
    if quote.doc_id not in cache:
        try:
            _, lines = document_lines(quote.doc_id)
            cache[quote.doc_id] = {i + 1: line for i, line in enumerate(lines) if line.strip()}
        except ToolError as exc:
            cache[quote.doc_id] = str(exc)
    lines = cache[quote.doc_id]
    if isinstance(lines, str):
        return lines
    return quote_problem(quote.text, quote.line, lines)
