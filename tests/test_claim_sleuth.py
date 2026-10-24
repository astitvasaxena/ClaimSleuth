import sys
import unittest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.state import ClaimInvestigationState, EvidenceItem
from src.tools.policy_rules import load_policy, calc_coverage_window, check_exclusions
from src.tools.math_engine import compute_settlement
from src.tools.anomaly_detector import flag_serial_anomalies
from src.tools.evidence_store import EvidenceStore
from src.agents.investigator import LeadInvestigatorAgent
from src.verifier.independent_verifier import DualGateVerifier

class TestDeterministicRules(unittest.TestCase):
    def test_calc_coverage_window_active(self):
        # 24-month policy, incident 6 months in
        res = calc_coverage_window("2024-01-01", "2024-07-01", 24)
        self.assertTrue(res["is_valid"])
        self.assertTrue(res["is_covered"])
        self.assertGreater(res["days_remaining_or_overdue"], 0)

    def test_calc_coverage_window_expired(self):
        # 12-month policy, incident 15 months later
        res = calc_coverage_window("2023-01-01", "2024-04-01", 12)
        self.assertTrue(res["is_valid"])
        self.assertFalse(res["is_covered"])
        self.assertLess(res["days_remaining_or_overdue"], 0)

    def test_calc_coverage_incident_before_purchase(self):
        res = calc_coverage_window("2024-05-01", "2024-04-01", 12)
        self.assertTrue(res["is_valid"])
        self.assertFalse(res["is_covered"])

    def test_check_exclusions(self):
        exclusions = ["intentional damage", "liquid submersion"]
        is_exc, match = check_exclusions("Device dropped in swimming pool, liquid submersion detected", exclusions)
        self.assertTrue(is_exc)
        self.assertEqual(match, "liquid submersion")

class TestMathEngine(unittest.TestCase):
    def test_basic_deductible_math(self):
        # Cost 280, Ded 29, Cap 1200, Dep 0% -> Payout 251.00
        calc = compute_settlement(repair_cost=280.0, deductible=29.0, max_coverage_limit=1200.0, depreciation_rate=0.0)
        self.assertEqual(calc.net_eligible_amount, 251.00)
        self.assertEqual(calc.approved_payout, 251.00)
        self.assertTrue(calc.is_arithmetically_valid)

    def test_cap_and_depreciation(self):
        # Cost 1000, Ded 50 -> 950 eligible. Cap 800 -> 800. Dep 10% (80) -> 720.00
        calc = compute_settlement(repair_cost=1000.0, deductible=50.0, max_coverage_limit=800.0, depreciation_rate=0.10)
        self.assertEqual(calc.net_eligible_amount, 800.00)
        self.assertEqual(calc.depreciation_applied, 80.00)
        self.assertEqual(calc.approved_payout, 720.00)

class TestAnomalyDetector(unittest.TestCase):
    def test_serial_mismatch_detected(self):
        report = flag_serial_anomalies(
            claim_serial="APPL-12345",
            invoice_serial="APPL-12345",
            diagnostic_serial="APPL-99999"
        )
        self.assertTrue(report["has_anomalies"])
        self.assertTrue(any("mismatch" in a.lower() for a in report["anomalies"]))

class TestDualGateVerifier(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy("POL-TECH-CARE-01")
        self.verifier = DualGateVerifier()

    def test_verifier_blocks_arithmetic_mismatch(self):
        # Create a state where an agent maliciously or mistakenly altered payout
        state = ClaimInvestigationState(
            claim_id="CLM-TEST-01",
            policy_id="POL-TECH-CARE-01",
            claimant_id="USR-1",
            device_model="iPhone 15",
            incident_date="2024-05-01",
            reported_issue="Screen broken",
            evidence={
                "purchase_date": EvidenceItem(
                    evidence_id="1", source_document="inv.txt", field_name="purchase_date",
                    extracted_value="2024-01-01", raw_snippet="Date: 2024-01-01", status="VERIFIED", confidence=0.99
                ),
                "serial_number": EvidenceItem(
                    evidence_id="2", source_document="inv.txt", field_name="serial_number",
                    extracted_value="SN-123", raw_snippet="SN: SN-123", status="VERIFIED", confidence=0.99
                ),
                "repair_cost": EvidenceItem(
                    evidence_id="3", source_document="inv.txt", field_name="repair_cost",
                    extracted_value=300.0, raw_snippet="Cost: $300.00", status="VERIFIED", confidence=0.95
                )
            }
        )
        # 300 - 29 = 271, but we inject a fake payout of 500.00
        fake_calc = compute_settlement(300.0, 29.0, 1200.0)
        fake_calc.approved_payout = 500.00
        state.deterministic_settlement = fake_calc

        report = self.verifier.verify_investigation(state, self.policy)
        self.assertFalse(report.is_approved_by_verifier)
        self.assertFalse(report.gate_2_arithmetic_matches)
        self.assertTrue(any("Arithmetic mismatch" in d for d in report.discrepancies))

class TestEndToEndScenarios(unittest.TestCase):
    def setUp(self):
        self.investigator = LeadInvestigatorAgent()

    def test_scenario_1_valid_claim(self):
        # Load Scenario 1 claim
        claim_path = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_1_valid" / "claim.json"
        with open(claim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = ClaimInvestigationState(**data)
        
        final_state = self.investigator.run_investigation(state)
        
        # Valid claim should reach VERIFIED_PENDING_APPROVAL
        self.assertEqual(final_state.status, "VERIFIED_PENDING_APPROVAL")
        self.assertTrue(final_state.verifier_report.is_approved_by_verifier)
        self.assertEqual(final_state.deterministic_settlement.approved_payout, 251.00)
        self.assertTrue(final_state.requires_human_approval)

    def test_scenario_2_expired_claim(self):
        # Load Scenario 2 claim
        claim_path = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_2_expired" / "claim.json"
        with open(claim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = ClaimInvestigationState(**data)
        
        final_state = self.investigator.run_investigation(state)
        
        # Expired warranty must be deterministically rejected
        self.assertEqual(final_state.status, "REJECTED_DETERMINISTICALLY")
        self.assertFalse(final_state.verifier_report.is_approved_by_verifier)
        self.assertTrue(any("expired" in d.lower() for d in final_state.verifier_report.discrepancies))

    def test_scenario_3_conflicting_serial(self):
        # Load Scenario 3 claim
        claim_path = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_3_conflict" / "claim.json"
        with open(claim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = ClaimInvestigationState(**data)
        
        final_state = self.investigator.run_investigation(state)
        
        # Conflicting serial numbers must halt and escalate to human
        self.assertEqual(final_state.status, "CONFLICT_ESCALATED")
        self.assertTrue(len(final_state.conflicts) > 0)
        self.assertFalse(final_state.verifier_report.is_approved_by_verifier)
        self.assertTrue(final_state.requires_human_approval)


# ── Phase 1 Foundation Tests ─────────────────────────────────────────

class TestAuditChainVerification(unittest.TestCase):
    """Tests for the SHA-256 hash-chain integrity verification."""

    def setUp(self):
        import tempfile, shutil
        self._tmpdir = tempfile.mkdtemp()
        # Patch AUDIT_LOG_DIR so tests don't touch real logs
        import configs.settings as settings
        self._orig_dir = settings.AUDIT_LOG_DIR
        settings.AUDIT_LOG_DIR = Path(self._tmpdir)
        from src.core import audit as audit_module
        audit_module.AUDIT_LOG_DIR = Path(self._tmpdir)

    def tearDown(self):
        import shutil
        import configs.settings as settings
        from src.core import audit as audit_module
        settings.AUDIT_LOG_DIR = self._orig_dir
        audit_module.AUDIT_LOG_DIR = self._orig_dir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_valid_chain_passes(self):
        from src.core.audit import AuditLogger
        logger = AuditLogger("TEST-CHAIN-OK")
        logger.log_step(1, "PLAN", "Test step 1")
        logger.log_step(2, "EXECUTE", "Test step 2")
        logger.log_evidence_extracted("ev1", "field1", "value1", 0.95, "doc.txt")
        result = logger.verify_chain()
        self.assertTrue(result["valid"])
        self.assertGreaterEqual(result["entries_checked"], 4)  # init + 3 entries
        self.assertIsNone(result["error"])

    def test_tampered_chain_fails(self):
        from src.core.audit import AuditLogger
        logger = AuditLogger("TEST-CHAIN-TAMPER")
        logger.log_step(1, "PLAN", "Step 1")
        logger.log_step(2, "EXECUTE", "Step 2")
        # Tamper with the log file: modify a field in the second entry
        with open(logger.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entry = json.loads(lines[1])
        entry["details"]["phase"] = "TAMPERED"
        lines[1] = json.dumps(entry) + "\n"
        with open(logger.log_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        # Re-create logger to get fresh state (verify_chain reads from disk)
        logger2 = AuditLogger.__new__(AuditLogger)
        logger2.claim_id = "TEST-CHAIN-TAMPER"
        logger2.log_file = logger.log_file
        logger2._last_hash = "GENESIS"
        result = logger2.verify_chain()
        self.assertFalse(result["valid"])
        self.assertIn("mismatch", result["error"].lower())


class TestToolRegistry(unittest.TestCase):
    """Tests for the restricted tool registry."""

    def test_registry_lists_registered_tools(self):
        from src.core.tool_registry import build_default_registry
        registry = build_default_registry()
        tools = registry.list_tools()
        self.assertIn("check_coverage_window", tools)
        self.assertIn("compute_settlement", tools)
        self.assertIn("check_exclusions", tools)
        self.assertIn("flag_serial_anomalies", tools)

    def test_registry_rejects_unknown_tool(self):
        from src.core.tool_registry import build_default_registry
        registry = build_default_registry()
        with self.assertRaises(ValueError) as ctx:
            registry.validate_and_execute("delete_all_claims", {})
        self.assertIn("Unknown tool", str(ctx.exception))

    def test_registry_validates_inputs(self):
        from src.core.tool_registry import build_default_registry
        registry = build_default_registry()
        # Missing required field 'incident_date'
        with self.assertRaises(ValueError) as ctx:
            registry.validate_and_execute("check_coverage_window", {
                "purchase_date": "2024-01-01",
                # incident_date missing
                "duration_months": 24
            })
        self.assertIn("validation failed", str(ctx.exception).lower())

    def test_registry_executes_valid_tool(self):
        from src.core.tool_registry import build_default_registry
        registry = build_default_registry()
        result = registry.validate_and_execute("check_coverage_window", {
            "purchase_date": "2024-01-01",
            "incident_date": "2024-07-01",
            "duration_months": 24
        })
        self.assertTrue(result["is_valid"])
        self.assertTrue(result["is_covered"])

    def test_get_tool_descriptions_nonempty(self):
        from src.core.tool_registry import build_default_registry
        registry = build_default_registry()
        desc = registry.get_tool_descriptions()
        self.assertIn("check_coverage_window", desc)
        self.assertIn("compute_settlement", desc)


class TestAdjudicationAuth(unittest.TestCase):
    """Tests for the adjudication endpoint authorization gate."""

    def test_auth_blocks_when_token_configured(self):
        """Directly test the _check_adjudication_auth helper function."""
        import os
        from unittest.mock import MagicMock
        from fastapi import HTTPException

        # Set a known token
        os.environ["CLAIMSLEUTH_ADJUDICATION_TOKEN"] = "test-secret-token-123"
        try:
            # Reload settings to pick up the env var
            import importlib, configs.settings
            importlib.reload(configs.settings)
            import src.ui.app as app_module
            importlib.reload(app_module)

            # 1. No Authorization header → 401
            mock_request = MagicMock()
            mock_request.headers = {}
            with self.assertRaises(HTTPException) as ctx:
                app_module._check_adjudication_auth(mock_request)
            self.assertEqual(ctx.exception.status_code, 401)

            # 2. Wrong token → 403
            mock_request.headers = {"Authorization": "Bearer wrong-token"}
            with self.assertRaises(HTTPException) as ctx:
                app_module._check_adjudication_auth(mock_request)
            self.assertEqual(ctx.exception.status_code, 403)

            # 3. Correct token → no exception
            mock_request.headers = {"Authorization": "Bearer test-secret-token-123"}
            app_module._check_adjudication_auth(mock_request)  # should not raise

        finally:
            os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
            import importlib, configs.settings
            importlib.reload(configs.settings)

    def test_auth_allows_when_no_token_configured(self):
        """When no token is set (demo mode), adjudication should be allowed."""
        import os
        from unittest.mock import MagicMock

        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        try:
            import importlib, configs.settings
            importlib.reload(configs.settings)
            import src.ui.app as app_module
            importlib.reload(app_module)

            mock_request = MagicMock()
            mock_request.headers = {}
            # Should not raise — demo mode allows unauthenticated
            app_module._check_adjudication_auth(mock_request)
        finally:
            import importlib, configs.settings
            importlib.reload(configs.settings)


class TestGeneralizedConflictDetection(unittest.TestCase):
    """Tests that conflict detection works for all evidence fields, not just serial_number."""

    def test_purchase_date_conflict_detected(self):
        """If two documents report different purchase dates, a conflict should be raised."""
        state = ClaimInvestigationState(
            claim_id="CLM-CONFLICT-DATE",
            policy_id="POL-TECH-CARE-01",
            claimant_id="USR-TEST",
            device_model="Test Device",
            incident_date="2024-09-01",
            reported_issue="Test issue",
            evidence={
                "purchase_date": EvidenceItem(
                    evidence_id="ev1", source_document="invoice_A.txt",
                    field_name="purchase_date", extracted_value="2024-01-15",
                    raw_snippet="Date: 2024-01-15", confidence=0.95, status="VERIFIED"
                )
            }
        )
        # Simulate the investigator recording an observation with a different purchase_date
        from src.core.audit import AuditLogger
        from src.agents.investigator import LeadInvestigatorAgent
        agent = LeadInvestigatorAgent()
        audit = AuditLogger("CLM-CONFLICT-DATE")
        conflicting_observation = {
            "extracted": {
                "purchase_date": EvidenceItem(
                    evidence_id="ev2", source_document="receipt_B.txt",
                    field_name="purchase_date", extracted_value="2023-06-01",
                    raw_snippet="Date: 2023-06-01", confidence=0.93, status="VERIFIED"
                )
            }
        }
        agent._record_observation("extract_invoice", conflicting_observation, state, audit)
        self.assertTrue(len(state.conflicts) > 0)
        self.assertTrue(any("purchase_date" in c for c in state.conflicts))


# ── Phase 2: Evidence Extractor Tests ─────────────────────────────────

class TestEvidenceExtractorDualMode(unittest.TestCase):
    """Tests for the Evidence Extractor dual-mode (Gemini + regex fallback) and provenance verification."""

    def test_regex_fallback_extraction(self):
        """Regex fallback extracts purchase_date and serial_number with valid snippets."""
        from src.agents.extractor import EvidenceExtractor
        extractor = EvidenceExtractor(llm_client=None)
        doc_text = "Acme Electronics\nInvoice Date: 2024-03-15\nSerial: SN-998877\nTotal: $350.00"
        results = extractor.extract_from_document(
            document_name="test_invoice.txt",
            document_text=doc_text,
            target_fields=["purchase_date", "serial_number"],
            use_llm=False
        )
        self.assertIn("purchase_date", results)
        self.assertIn("serial_number", results)
        self.assertEqual(results["purchase_date"].extracted_value, "2024-03-15")
        self.assertEqual(results["serial_number"].extracted_value, "SN-998877")
        self.assertEqual(results["purchase_date"].status, "VERIFIED")
        self.assertEqual(results["serial_number"].status, "VERIFIED")

    def test_snippet_provenance_verification_passes(self):
        """Verbatim snippets present in source document receive VERIFIED status."""
        from src.agents.extractor import EvidenceExtractor
        extractor = EvidenceExtractor()
        doc_text = "Device diagnostics report. Repair Cost: $280.00 for replacement screen."
        results = extractor.extract_from_document(
            document_name="diag.txt",
            document_text=doc_text,
            target_fields=["repair_cost"],
            use_llm=False
        )
        self.assertIn("repair_cost", results)
        self.assertEqual(results["repair_cost"].status, "VERIFIED")
        self.assertIn(results["repair_cost"].raw_snippet, doc_text)

    def test_snippet_provenance_verification_fails_on_hallucination(self):
        """If an extractor/LLM provides a snippet not in the source text, it is marked UNVERIFIED."""
        from src.agents.extractor import EvidenceExtractor, GeminiExtractionResponse, GeminiExtractedField
        mock_client = MagicMock()
        # Mock LLM hallucinating a snippet that does not exist in the source document
        mock_client.generate_structured.return_value = GeminiExtractionResponse(
            fields=[
                GeminiExtractedField(
                    field_name="purchase_date",
                    extracted_value="2024-01-01",
                    raw_snippet="Fictional Invoice Date: 2024-01-01",  # NOT in doc_text
                    confidence=0.99
                )
            ]
        )
        extractor = EvidenceExtractor(llm_client=mock_client)
        doc_text = "Real document text without any date mention."
        results = extractor.extract_from_document(
            document_name="receipt.txt",
            document_text=doc_text,
            target_fields=["purchase_date"],
            use_llm=True
        )
        self.assertIn("purchase_date", results)
        self.assertEqual(results["purchase_date"].status, "UNVERIFIED")
        self.assertIn("Snippet mismatch", results["purchase_date"].verification_notes)

    def test_gemini_extraction_mocked(self):
        """Gemini extraction parses structured response and verifies valid snippets."""
        from src.agents.extractor import EvidenceExtractor, GeminiExtractionResponse, GeminiExtractedField
        doc_text = "Purchase Date: 2024-02-20 | Serial Number: SN-12345"
        mock_client = MagicMock()
        mock_client.generate_structured.return_value = GeminiExtractionResponse(
            fields=[
                GeminiExtractedField(
                    field_name="purchase_date",
                    extracted_value="2024-02-20",
                    raw_snippet="Purchase Date: 2024-02-20",
                    confidence=0.98
                ),
                GeminiExtractedField(
                    field_name="serial_number",
                    extracted_value="SN-12345",
                    raw_snippet="Serial Number: SN-12345",
                    confidence=0.99
                )
            ]
        )
        extractor = EvidenceExtractor(llm_client=mock_client)
        results = extractor.extract_from_document(
            document_name="invoice.txt",
            document_text=doc_text,
            target_fields=["purchase_date", "serial_number"],
            use_llm=True
        )
        self.assertEqual(results["purchase_date"].status, "VERIFIED")
        self.assertEqual(results["serial_number"].status, "VERIFIED")
        self.assertEqual(results["purchase_date"].extracted_value, "2024-02-20")

    def test_gemini_extraction_falls_back_on_error(self):
        """When Gemini client raises an exception, extractor seamlessly falls back to regex."""
        from src.agents.extractor import EvidenceExtractor
        doc_text = "Invoice Date: 2024-05-10 | Serial: SN-998811"
        mock_client = MagicMock()
        mock_client.generate_structured.side_effect = RuntimeError("Gemini API network timeout")

        extractor = EvidenceExtractor(llm_client=mock_client)
        results = extractor.extract_from_document(
            document_name="inv.txt",
            document_text=doc_text,
            target_fields=["purchase_date", "serial_number"],
            use_llm=True
        )
        # Should gracefully fall back to regex
        self.assertIn("purchase_date", results)
        self.assertEqual(results["purchase_date"].extracted_value, "2024-05-10")
        self.assertEqual(results["purchase_date"].status, "VERIFIED")


# ── Phase 3: Adaptive Agent Investigator Tests ─────────────────────────

class TestAdaptiveAgentInvestigator(unittest.TestCase):
    """Tests proving agency: tool order adaptation, replanning, early stopping, and safety."""

    def test_agent_chooses_different_tool_order(self):
        """Mock Gemini selecting diagnostics before invoice; agent completes and verifier approves."""
        from src.agents.investigator import LeadInvestigatorAgent
        from src.core.state import AgentAction

        # Prepare actions: diagnostic facts first, then invoice facts, then settlement, then STOP
        actions = [
            AgentAction(
                tool_name="extract_document_facts",
                tool_args={
                    "document_name": "repair_estimate_001.txt",
                    "target_fields": ["repair_cost", "incident_cause", "serial_number"]
                },
                reasoning="Investigate diagnostic estimate first"
            ),
            AgentAction(
                tool_name="extract_document_facts",
                tool_args={
                    "document_name": "invoice_001.txt",
                    "target_fields": ["purchase_date", "serial_number"]
                },
                reasoning="Now extract purchase invoice proof"
            ),
            AgentAction(
                tool_name="compute_settlement",
                tool_args={},
                reasoning="Compute mathematical settlement"
            ),
            AgentAction(
                tool_name="STOP",
                tool_args={},
                reasoning="All required evidence acquired"
            )
        ]

        def planner_side_effect(prompt, response_schema=None, **kwargs):
            from src.core.state import AgentAction
            if response_schema == AgentAction:
                return next(action_iter)
            raise RuntimeError("Fallback to regex for document extraction in test")

        action_iter = iter(actions)
        mock_client = MagicMock()
        mock_client.generate_structured.side_effect = planner_side_effect

        agent = LeadInvestigatorAgent(llm_client=mock_client)
        claim_path = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_1_valid" / "claim.json"
        with open(claim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = ClaimInvestigationState(**data)

        final_state = agent.run_investigation(state)
        self.assertEqual(final_state.status, "VERIFIED_PENDING_APPROVAL")
        self.assertTrue(final_state.verifier_report.is_approved_by_verifier)
        self.assertEqual(final_state.deterministic_settlement.approved_payout, 251.00)

        # Verify steps followed non-standard order
        first_step = final_state.steps_history[0]
        self.assertIn("repair_estimate", first_step.tool_input.get("document_name", ""))

    def test_agent_replans_on_failed_extraction(self):
        """When an extraction returns missing fields, the agent issues a REPLAN phase step."""
        from src.agents.investigator import LeadInvestigatorAgent
        from src.core.state import AgentAction

        # Step 1: Extract from document that lacks repair_cost
        # Step 2: Agent recognizes failure and issues a REPLAN step with replan_reason
        # Step 3: Compute settlement, Step 4: STOP
        actions = [
            AgentAction(
                tool_name="extract_document_facts",
                tool_args={
                    "document_name": "invoice_001.txt",
                    "target_fields": ["purchase_date", "repair_cost"]
                },
                reasoning="Try extracting repair cost from invoice"
            ),
            AgentAction(
                tool_name="extract_document_facts",
                tool_args={
                    "document_name": "repair_estimate_001.txt",
                    "target_fields": ["repair_cost", "serial_number"]
                },
                reasoning="Extract repair cost from diagnostic report",
                replan_reason="repair_cost not present in invoice_001.txt; replanning with repair_estimate_001.txt"
            ),
            AgentAction(
                tool_name="compute_settlement",
                tool_args={},
                reasoning="Calculate final settlement"
            ),
            AgentAction(
                tool_name="STOP",
                tool_args={},
                reasoning="Investigation complete"
            )
        ]

        def replan_side_effect(prompt, response_schema=None, **kwargs):
            from src.core.state import AgentAction
            if response_schema == AgentAction:
                return next(replan_iter)
            raise RuntimeError("Fallback to regex for document extraction in test")

        replan_iter = iter(actions)
        mock_client = MagicMock()
        mock_client.generate_structured.side_effect = replan_side_effect

        agent = LeadInvestigatorAgent(llm_client=mock_client)
        claim_path = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_1_valid" / "claim.json"
        with open(claim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = ClaimInvestigationState(**data)

        final_state = agent.run_investigation(state)
        # Verify that a REPLAN step occurred in steps_history
        phases = [s.phase for s in final_state.steps_history]
        self.assertIn("REPLAN", phases)
        replan_step = next(s for s in final_state.steps_history if s.phase == "REPLAN")
        self.assertIsNotNone(replan_step.replan_reason)
        self.assertIn("repair_cost not present", replan_step.replan_reason)

    def test_agent_stops_early_on_expired_policy(self):
        """Investigation of expired policy halts early without computing settlement."""
        from src.agents.investigator import LeadInvestigatorAgent
        agent = LeadInvestigatorAgent()  # deterministic fallback mode
        claim_path = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_2_expired" / "claim.json"
        with open(claim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = ClaimInvestigationState(**data)

        final_state = agent.run_investigation(state)
        self.assertEqual(final_state.status, "REJECTED_DETERMINISTICALLY")
        # Should NOT have computed settlement because policy is expired
        self.assertIsNone(final_state.deterministic_settlement)

    def test_agent_calls_anomaly_detector(self):
        """Serial number mismatch triggers anomaly detection and escalates to human."""
        from src.agents.investigator import LeadInvestigatorAgent
        agent = LeadInvestigatorAgent()
        claim_path = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_3_conflict" / "claim.json"
        with open(claim_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = ClaimInvestigationState(**data)

        final_state = agent.run_investigation(state)
        self.assertEqual(final_state.status, "CONFLICT_ESCALATED")
        self.assertTrue(len(final_state.conflicts) > 0)
        self.assertTrue(final_state.requires_human_approval)

    def test_graceful_degradation_no_api_key(self):
        """When USE_LLM is false or no key is present, all 3 standard scenarios succeed deterministically."""
        import os
        from src.agents.investigator import LeadInvestigatorAgent

        os.environ.pop("GOOGLE_API_KEY", None)
        agent = LeadInvestigatorAgent(use_llm=False)

        # Scenario 1 passes
        p1 = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_1_valid" / "claim.json"
        with open(p1, "r", encoding="utf-8") as f:
            s1 = agent.run_investigation(ClaimInvestigationState(**json.load(f)))
        self.assertEqual(s1.status, "VERIFIED_PENDING_APPROVAL")

        # Scenario 2 rejected
        p2 = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_2_expired" / "claim.json"
        with open(p2, "r", encoding="utf-8") as f:
            s2 = agent.run_investigation(ClaimInvestigationState(**json.load(f)))
        self.assertEqual(s2.status, "REJECTED_DETERMINISTICALLY")

        # Scenario 3 conflict
        p3 = Path(__file__).resolve().parent.parent / "data" / "sample_claims" / "scenario_3_conflict" / "claim.json"
        with open(p3, "r", encoding="utf-8") as f:
            s3 = agent.run_investigation(ClaimInvestigationState(**json.load(f)))
        self.assertEqual(s3.status, "CONFLICT_ESCALATED")


if __name__ == "__main__":
    unittest.main()

