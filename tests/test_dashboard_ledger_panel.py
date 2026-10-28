"""
The dashboard no longer shows the audit-ledger panel. The audit log itself is unchanged:
it is still written for every run, returned by the API and checked by /verify.
"""
import importlib
import os
import unittest

from fastapi.testclient import TestClient


class TestLedgerPanelRemovedFromDashboard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ.pop("CLAIMSLEUTH_ADJUDICATION_TOKEN", None)
        import configs.settings
        importlib.reload(configs.settings)
        import src.ui.app as app_module
        cls.client = TestClient(importlib.reload(app_module).app)
        cls.page = cls.client.get("/").text

    def test_panel_markup_script_and_styles_are_gone(self):
        for removed in ("Tamper-Evident", 'id="auditTrail"', "renderAuditTrail", "loadAuditTrail",
                        ".audit-entry", ".audit-trail", "audit-hash"):
            self.assertNotIn(removed, self.page, removed)

    def test_rest_of_the_dashboard_is_still_served(self):
        for kept in ('id="hitlBox"', "adjudicate('APPROVED')", 'id="blackboardFactsContainer"',
                     'id="a2aMessagesContainer"', 'id="stepperTimeline"', 'id="rPayout"',
                     'value="legacy"', 'value="adk_local"', 'value="adk_mcp"', 'value="adk_mcp_crewai"'):
            self.assertIn(kept, self.page, kept)

    def test_audit_log_and_endpoints_are_unaffected(self):
        data = self.client.post("/api/adk/investigate/scenario_1_valid?tools=local").json()
        self.assertTrue(data["audit_trail"])                       # still returned with every investigation
        trail = self.client.get("/api/audit/CLM-2024-001").json()
        self.assertTrue(trail)
        self.assertTrue(self.client.get("/api/audit/CLM-2024-001/verify").json()["valid"])


if __name__ == "__main__":
    unittest.main()
