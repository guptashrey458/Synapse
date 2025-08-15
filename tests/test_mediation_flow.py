import pytest
from src.tools.mediation_tools import (
    InitiateMediationFlowTool, CollectEvidenceTool, AnalyzeEvidenceTool,
    IssueInstantRefundTool, ExonerateDriverTool, LogMerchantPackagingFeedbackTool
)

def test_mediation_happy_path():
    init = InitiateMediationFlowTool().execute(order_id="O1", dispute_type="packaging_damage", _seed=42)
    assert init.success
    
    ce = CollectEvidenceTool().execute(order_id="O1", _seed=42)
    assert ce.success and ce.data["evidence_count"] > 0
    
    ae = AnalyzeEvidenceTool().execute(evidence_items=ce.data["evidence_items"], dispute_type="packaging_damage", _seed=42)
    assert ae.success and 0.6 <= ae.data["analysis_confidence"] <= 0.95

    if ae.data["primary_fault"] == "merchant_fault":
        rr = IssueInstantRefundTool().execute(order_id="O1", amount=150, reason="merchant_fault",
                                             requires_manager_approval=False, _seed=42)
        assert rr.success
        
        ex = ExonerateDriverTool().execute(order_id="O1", _seed=42)
        assert ex.success
        
        fm = LogMerchantPackagingFeedbackTool().execute(merchant_id="M1", report={"fault":"packaging"}, _seed=42)
        assert fm.success