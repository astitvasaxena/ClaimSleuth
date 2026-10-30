"""Exact money arithmetic shared by the MCP tools and the crew's guardrail (no MCP or model imports)."""
from decimal import ROUND_HALF_UP, Decimal

CENTS = Decimal("0.01")


def cents(value) -> Decimal:
    return Decimal(str(value)).quantize(CENTS, ROUND_HALF_UP)


def invoice_totals(line_items, stated_total) -> dict:
    """Sum the line items (each rounded to cents) and compare with the stated total, to the cent."""
    if not line_items:
        raise ValueError("give at least one line item")
    items = [cents(v) for v in line_items]
    if any(v < 0 for v in items) or cents(stated_total) < 0:
        raise ValueError("amounts must not be negative")
    computed, stated = sum(items, Decimal("0.00")), cents(stated_total)
    return {"computed_total": computed, "stated_total": stated, "difference": stated - computed,
            "matches": computed == stated}
