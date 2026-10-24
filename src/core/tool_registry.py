"""
Restricted Tool Registry for ClaimSleuth.

Maps tool names -> callable functions + Pydantic input schemas.
The agent may only invoke tools listed here. All inputs are validated
before execution. Unknown tool names are rejected.
"""
from typing import Dict, Any, Callable, Optional, Type, List
from pydantic import BaseModel, Field, ConfigDict


# ─── Input Schemas (one per registered tool) ─────────────────────────

class ListDossierDocumentsInput(BaseModel):
    """Input for listing all documents in a claim dossier."""
    claim_id: str

class ExtractDocumentFactsInput(BaseModel):
    """Input for extracting structured facts from a specific document."""
    document_name: str
    target_fields: List[str] = Field(
        description="Fields to extract, e.g. ['purchase_date', 'serial_number', 'repair_cost', 'incident_cause']"
    )
    document_text: Optional[str] = ""
    claim_id: Optional[str] = ""

class CheckCoverageWindowInput(BaseModel):
    """Input for the deterministic coverage-window date check."""
    purchase_date: str = Field(description="YYYY-MM-DD format")
    incident_date: str = Field(description="YYYY-MM-DD format")
    duration_months: int

class ComputeSettlementInput(BaseModel):
    """Input for the deterministic financial settlement calculator."""
    repair_cost: float
    deductible: float
    max_coverage_limit: float
    depreciation_rate: float = 0.0

class CheckExclusionsInput(BaseModel):
    """Input for the deterministic warranty exclusion checker."""
    reported_cause: str
    exclusions: List[str]

class FlagSerialAnomaliesInput(BaseModel):
    """Input for the deterministic serial-number anomaly detector."""
    claim_serial: str
    invoice_serial: str
    diagnostic_serial: str


# ─── Tool Specification & Registry ───────────────────────────────────

class ToolSpec(BaseModel):
    """Specification for a single registered tool."""
    name: str
    description: str
    input_schema: Any  # Type[BaseModel] — stored as reference
    callable_fn: Any   # Callable — stored as reference

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolRegistry:
    """
    Restricted tool registry. Only tools explicitly registered here
    can be invoked by the agent. All inputs are Pydantic-validated
    before execution.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, input_schema: Type[BaseModel], callable_fn: Callable):
        """Register a tool. Overwrites if name already exists."""
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            callable_fn=callable_fn
        )

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """
        Returns a formatted string describing all registered tools and
        their input schemas. Used in the system prompt for the LLM.
        """
        lines = []
        for name, spec in self._tools.items():
            schema = spec.input_schema.model_json_schema()
            props = schema.get("properties", {})
            param_descs = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                param_descs.append(f"    - {pname} ({ptype}): {pdesc}")
            params_str = "\n".join(param_descs) if param_descs else "    (no parameters)"
            lines.append(f"Tool: {name}\n  Description: {spec.description}\n  Parameters:\n{params_str}")
        return "\n\n".join(lines)

    def validate_and_execute(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates tool_args against the registered Pydantic schema,
        then executes the tool. Returns the result dict.
        Raises ValueError for unknown tools or validation failures.
        """
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool '{tool_name}'. Allowed tools: {self.list_tools()}")

        spec = self._tools[tool_name]

        # Validate inputs against Pydantic schema
        try:
            validated_input = spec.input_schema(**tool_args)
        except Exception as e:
            raise ValueError(f"Input validation failed for tool '{tool_name}': {e}")

        # Execute with validated inputs
        result = spec.callable_fn(**validated_input.model_dump())
        return result


def build_default_registry(evidence_store=None, extractor=None) -> ToolRegistry:
    """
    Constructs the default tool registry with all 6 allowed tools.
    Wired with deterministic policy, math, anomaly, and evidence tools.
    """
    from src.tools.policy_rules import calc_coverage_window, check_exclusions
    from src.tools.math_engine import compute_settlement
    from src.tools.anomaly_detector import flag_serial_anomalies
    from src.tools.evidence_store import EvidenceStore

    registry = ToolRegistry()

    # Tool 1: list_dossier_documents
    def _list_docs(claim_id: str):
        store = evidence_store or EvidenceStore()
        return {"documents": store.list_documents(claim_id)}

    registry.register(
        name="list_dossier_documents",
        description="Lists all document filenames available in the claim dossier.",
        input_schema=ListDossierDocumentsInput,
        callable_fn=_list_docs
    )

    # Tool 2: extract_document_facts
    def _extract_facts(document_name: str, target_fields: List[str], document_text: Optional[str] = "", claim_id: Optional[str] = ""):
        from src.agents.extractor import EvidenceExtractor
        store = evidence_store or EvidenceStore()
        ext = extractor or EvidenceExtractor()
        text = document_text or ""
        if not text and claim_id:
            text = store.read_document(claim_id, document_name) or ""
        if not text:
            all_docs = store.get_all_claim_documents(claim_id) if claim_id else {}
            text = all_docs.get(document_name, "")
        extracted = ext.extract_from_document(document_name, text, target_fields)
        return {"extracted": extracted, "doc_name": document_name}

    registry.register(
        name="extract_document_facts",
        description="Extracts structured facts (dates, costs, serials, causes) from a document in the dossier.",
        input_schema=ExtractDocumentFactsInput,
        callable_fn=_extract_facts
    )

    # Tool 3: check_coverage_window
    registry.register(
        name="check_coverage_window",
        description="Deterministic date check: is the incident date within the warranty coverage period?",
        input_schema=CheckCoverageWindowInput,
        callable_fn=lambda purchase_date, incident_date, duration_months:
            calc_coverage_window(purchase_date, incident_date, duration_months)
    )

    # Tool 4: compute_settlement
    registry.register(
        name="compute_settlement",
        description="Deterministic financial settlement calculator. Applies deductible, coverage cap, and depreciation.",
        input_schema=ComputeSettlementInput,
        callable_fn=lambda repair_cost, deductible, max_coverage_limit, depreciation_rate=0.0:
            {"settlement": compute_settlement(repair_cost, deductible, max_coverage_limit, depreciation_rate)}
    )

    # Tool 5: check_exclusions
    registry.register(
        name="check_exclusions",
        description="Deterministic check: does the reported cause match any warranty exclusion?",
        input_schema=CheckExclusionsInput,
        callable_fn=lambda reported_cause, exclusions:
            {"is_excluded": check_exclusions(reported_cause, exclusions)[0],
             "matched_exclusion": check_exclusions(reported_cause, exclusions)[1]}
    )

    # Tool 6: flag_serial_anomalies
    registry.register(
        name="flag_serial_anomalies",
        description="Deterministic serial-number consistency checker across claim, invoice, and diagnostic report.",
        input_schema=FlagSerialAnomaliesInput,
        callable_fn=lambda claim_serial, invoice_serial, diagnostic_serial:
            flag_serial_anomalies(claim_serial, invoice_serial, diagnostic_serial)
    )

    return registry
