"""
Mediation and Dispute Resolution Tools

This module provides tools for handling customer-merchant disputes,
evidence collection, analysis, and fair resolution processes.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .interfaces import ToolResult

logger = logging.getLogger(__name__)

@dataclass
class MediationCase:
    """Represents a mediation case."""
    case_id: str
    order_id: str
    dispute_type: str
    parties: List[str]
    status: str
    created_at: datetime
    evidence_items: List[Dict[str, Any]]
    resolution: Optional[Dict[str, Any]] = None

class InitiateMediationFlowTool:
    """Tool for initiating formal mediation processes."""
    
    def __init__(self):
        self.name = "initiate_mediation_flow"
        self.description = "Initiate formal mediation process for disputes"
        self._seed = None
    
    def execute(self, order_id: str, dispute_type: str, 
                parties: Optional[List[str]] = None, _seed: Optional[int] = None) -> ToolResult:
        """
        Initiate a mediation flow for a dispute.
        
        Args:
            order_id: The order ID involved in the dispute
            dispute_type: Type of dispute (e.g., 'food_quality', 'delivery_delay', 'packaging_damage')
            parties: List of parties involved (defaults to ['customer', 'merchant'])
            _seed: Random seed for deterministic testing
        """
        try:
            if _seed is not None:
                self._seed = _seed
                random.seed(_seed)
            
            if not order_id or not dispute_type:
                return self._err_result("Missing required parameters: order_id and dispute_type")
            
            parties = parties or ['customer', 'merchant']
            case_id = f"MED_{order_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Simulate mediation initiation
            success_rate = 0.95  # High success rate for initiation
            if random.random() < success_rate:
                
                mediation_data = {
                    "case_id": case_id,
                    "order_id": order_id,
                    "dispute_type": dispute_type,
                    "parties": parties,
                    "status": "initiated",
                    "created_at": datetime.now().isoformat(),
                    "estimated_resolution_time": "24-48 hours",
                    "mediation_process": {
                        "evidence_collection_phase": "active",
                        "analysis_phase": "pending",
                        "resolution_phase": "pending"
                    },
                    "next_steps": [
                        "Collect evidence from all parties",
                        "Analyze evidence objectively",
                        "Determine fair resolution"
                    ]
                }
                
                logger.info(f"Mediation initiated for order {order_id}, case {case_id}")
                return self._ok_result(mediation_data)
            else:
                return self._err_result("Failed to initiate mediation process")
                
        except Exception as e:
            logger.error(f"Error initiating mediation: {e}")
            return self._err_result(f"Mediation initiation failed: {str(e)}")
    
    def _ok_result(self, data: Dict[str, Any]) -> ToolResult:
        """Helper to create successful result."""
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            execution_time=random.uniform(0.5, 2.0) if self._seed else 1.0
        )
    
    def _err_result(self, error: str) -> ToolResult:
        """Helper to create error result."""
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={"error": error},
            execution_time=random.uniform(0.2, 0.8) if self._seed else 0.5
        )

class CollectEvidenceTool:
    """Tool for collecting evidence from all parties in a dispute."""
    
    def __init__(self):
        self.name = "collect_evidence"
        self.description = "Collect evidence from parties involved in a dispute"
        self._seed = None
    
    def execute(self, order_id: str, parties: Optional[List[str]] = None,
                evidence_types: Optional[List[str]] = None, _seed: Optional[int] = None) -> ToolResult:
        """
        Collect evidence from all parties involved in a dispute.
        
        Args:
            order_id: The order ID for evidence collection
            parties: Parties to collect evidence from
            evidence_types: Types of evidence to collect
            _seed: Random seed for deterministic testing
        """
        try:
            if _seed is not None:
                self._seed = _seed
                random.seed(_seed)
            
            if not order_id:
                return self._err_result("Missing required parameter: order_id")
            
            parties = parties or ['customer', 'merchant', 'driver']
            evidence_types = evidence_types or ['photos', 'timestamps', 'communications', 'receipts']
            
            # Simulate evidence collection
            success_rate = 0.85
            if random.random() < success_rate:
                
                evidence_items = []
                evidence_count = random.randint(3, 8)
                
                for i in range(evidence_count):
                    evidence_type = random.choice(evidence_types)
                    party = random.choice(parties)
                    
                    evidence_item = {
                        "evidence_id": f"EV_{order_id}_{i+1:03d}",
                        "type": evidence_type,
                        "source": party,
                        "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                        "reliability_score": round(random.uniform(0.6, 0.95), 2),
                        "description": f"{evidence_type.title()} evidence from {party}",
                        "metadata": {
                            "collection_method": "automated_system",
                            "verification_status": "verified" if random.random() > 0.2 else "pending"
                        }
                    }
                    evidence_items.append(evidence_item)
                
                collection_data = {
                    "order_id": order_id,
                    "evidence_count": evidence_count,
                    "evidence_items": evidence_items,
                    "collection_status": "completed",
                    "parties_responded": len(parties),
                    "collection_time": datetime.now().isoformat(),
                    "quality_metrics": {
                        "average_reliability": round(sum(item["reliability_score"] for item in evidence_items) / len(evidence_items), 2),
                        "verified_items": sum(1 for item in evidence_items if item["metadata"]["verification_status"] == "verified"),
                        "completeness_score": round(random.uniform(0.7, 0.95), 2)
                    }
                }
                
                logger.info(f"Evidence collected for order {order_id}: {evidence_count} items")
                return self._ok_result(collection_data)
            else:
                return self._err_result("Evidence collection failed or incomplete")
                
        except Exception as e:
            logger.error(f"Error collecting evidence: {e}")
            return self._err_result(f"Evidence collection failed: {str(e)}")
    
    def _ok_result(self, data: Dict[str, Any]) -> ToolResult:
        """Helper to create successful result."""
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            execution_time=random.uniform(1.0, 3.0) if self._seed else 2.0
        )
    
    def _err_result(self, error: str) -> ToolResult:
        """Helper to create error result."""
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={"error": error},
            execution_time=random.uniform(0.5, 1.5) if self._seed else 1.0
        )

class AnalyzeEvidenceTool:
    """Tool for objective analysis of collected evidence."""
    
    def __init__(self):
        self.name = "analyze_evidence"
        self.description = "Analyze evidence objectively to determine fault and resolution"
        self._seed = None
    
    def execute(self, evidence_items: List[Dict[str, Any]], dispute_type: str,
                _seed: Optional[int] = None) -> ToolResult:
        """
        Analyze evidence to determine fault and appropriate resolution.
        
        Args:
            evidence_items: List of evidence items to analyze
            dispute_type: Type of dispute being analyzed
            _seed: Random seed for deterministic testing
        """
        try:
            if _seed is not None:
                self._seed = _seed
                random.seed(_seed)
            
            if not evidence_items or not dispute_type:
                return self._err_result("Missing required parameters: evidence_items and dispute_type")
            
            # Simulate evidence analysis
            success_rate = 0.9
            if random.random() < success_rate:
                
                # Calculate analysis metrics
                total_items = len(evidence_items)
                avg_reliability = sum(item.get("reliability_score", 0.5) for item in evidence_items) / total_items
                verified_items = sum(1 for item in evidence_items if item.get("metadata", {}).get("verification_status") == "verified")
                
                # Determine primary fault based on evidence and dispute type
                fault_options = ["customer_fault", "merchant_fault", "driver_fault", "system_fault", "shared_fault", "no_fault"]
                
                # Weight fault determination based on dispute type
                if dispute_type == "packaging_damage":
                    fault_weights = [0.1, 0.6, 0.1, 0.1, 0.1, 0.0]
                elif dispute_type == "delivery_delay":
                    fault_weights = [0.1, 0.2, 0.4, 0.2, 0.1, 0.0]
                elif dispute_type == "food_quality":
                    fault_weights = [0.1, 0.7, 0.0, 0.1, 0.1, 0.0]
                elif dispute_type == "wrong_order":
                    fault_weights = [0.1, 0.5, 0.1, 0.2, 0.1, 0.0]
                else:
                    fault_weights = [0.15, 0.25, 0.15, 0.15, 0.25, 0.05]  # Default distribution
                
                primary_fault = random.choices(fault_options, weights=fault_weights)[0]
                
                # Calculate confidence based on evidence quality
                base_confidence = avg_reliability * 0.6 + (verified_items / total_items) * 0.4
                confidence_variance = random.uniform(-0.1, 0.1)
                analysis_confidence = max(0.6, min(0.95, base_confidence + confidence_variance))
                
                # Determine recommended resolution
                resolution_recommendations = self._generate_resolution_recommendations(
                    primary_fault, dispute_type, analysis_confidence
                )
                
                analysis_data = {
                    "dispute_type": dispute_type,
                    "evidence_analyzed": total_items,
                    "analysis_confidence": round(analysis_confidence, 2),
                    "primary_fault": primary_fault,
                    "fault_distribution": {
                        fault: round(weight, 2) for fault, weight in zip(fault_options, fault_weights)
                    },
                    "evidence_quality": {
                        "average_reliability": round(avg_reliability, 2),
                        "verified_percentage": round((verified_items / total_items) * 100, 1),
                        "total_evidence_items": total_items
                    },
                    "resolution_recommendations": resolution_recommendations,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "key_findings": self._generate_key_findings(evidence_items, primary_fault)
                }
                
                logger.info(f"Evidence analysis completed: {primary_fault} with {analysis_confidence:.2f} confidence")
                return self._ok_result(analysis_data)
            else:
                return self._err_result("Evidence analysis failed due to insufficient or conflicting evidence")
                
        except Exception as e:
            logger.error(f"Error analyzing evidence: {e}")
            return self._err_result(f"Evidence analysis failed: {str(e)}")
    
    def _generate_resolution_recommendations(self, fault: str, dispute_type: str, confidence: float) -> Dict[str, Any]:
        """Generate resolution recommendations based on fault determination."""
        
        recommendations = {
            "primary_action": "",
            "compensation_amount": 0,
            "compensation_type": "none",
            "additional_actions": [],
            "requires_manager_approval": False
        }
        
        if fault == "merchant_fault":
            if dispute_type in ["food_quality", "wrong_order", "packaging_damage"]:
                recommendations["primary_action"] = "full_refund_and_replacement"
                recommendations["compensation_amount"] = random.randint(15, 50)
                recommendations["compensation_type"] = "refund_plus_credit"
                recommendations["additional_actions"] = ["merchant_feedback", "quality_review"]
                recommendations["requires_manager_approval"] = recommendations["compensation_amount"] > 30
            else:
                recommendations["primary_action"] = "partial_refund"
                recommendations["compensation_amount"] = random.randint(8, 25)
                recommendations["compensation_type"] = "refund"
                
        elif fault == "driver_fault":
            recommendations["primary_action"] = "delivery_credit"
            recommendations["compensation_amount"] = random.randint(5, 15)
            recommendations["compensation_type"] = "credit"
            recommendations["additional_actions"] = ["driver_coaching"]
            
        elif fault == "system_fault":
            recommendations["primary_action"] = "service_credit"
            recommendations["compensation_amount"] = random.randint(10, 20)
            recommendations["compensation_type"] = "credit"
            recommendations["additional_actions"] = ["system_improvement"]
            
        elif fault == "shared_fault":
            recommendations["primary_action"] = "goodwill_gesture"
            recommendations["compensation_amount"] = random.randint(5, 15)
            recommendations["compensation_type"] = "credit"
            
        elif fault == "customer_fault":
            recommendations["primary_action"] = "explanation_and_education"
            recommendations["additional_actions"] = ["customer_education"]
            
        # Adjust based on confidence
        if confidence < 0.7:
            recommendations["requires_manager_approval"] = True
            recommendations["additional_actions"].append("manual_review")
        
        return recommendations
    
    def _generate_key_findings(self, evidence_items: List[Dict[str, Any]], fault: str) -> List[str]:
        """Generate key findings from evidence analysis."""
        
        findings = []
        
        # Sample findings based on evidence types present
        evidence_types = set(item.get("type", "unknown") for item in evidence_items)
        
        if "photos" in evidence_types:
            findings.append("Photographic evidence supports the claim")
        if "timestamps" in evidence_types:
            findings.append("Timeline analysis reveals service delays")
        if "communications" in evidence_types:
            findings.append("Communication records show customer service interactions")
        
        # Add fault-specific findings
        if fault == "merchant_fault":
            findings.append("Evidence indicates merchant responsibility for the issue")
        elif fault == "driver_fault":
            findings.append("Delivery process shows driver-related complications")
        elif fault == "system_fault":
            findings.append("System logs indicate technical issues during order processing")
        
        return findings[:4]  # Limit to top 4 findings
    
    def _ok_result(self, data: Dict[str, Any]) -> ToolResult:
        """Helper to create successful result."""
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            execution_time=random.uniform(2.0, 5.0) if self._seed else 3.0
        )
    
    def _err_result(self, error: str) -> ToolResult:
        """Helper to create error result."""
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={"error": error},
            execution_time=random.uniform(1.0, 2.0) if self._seed else 1.5
        )

class IssueInstantRefundTool:
    """Tool for issuing instant refunds and compensation."""
    
    def __init__(self):
        self.name = "issue_instant_refund"
        self.description = "Issue instant refunds and compensation to customers"
        self._seed = None
    
    def execute(self, order_id: str, amount: float, reason: str,
                refund_type: str = "refund", requires_manager_approval: bool = False,
                _seed: Optional[int] = None) -> ToolResult:
        """
        Issue instant refund or compensation to customer.
        
        Args:
            order_id: The order ID for the refund
            amount: Amount to refund/compensate
            reason: Reason for the refund
            refund_type: Type of refund ('refund', 'credit', 'refund_plus_credit')
            requires_manager_approval: Whether manager approval is needed
            _seed: Random seed for deterministic testing
        """
        try:
            if _seed is not None:
                self._seed = _seed
                random.seed(_seed)
            
            if not order_id or amount <= 0 or not reason:
                return self._err_result("Missing or invalid parameters: order_id, amount, reason")
            
            # Simulate refund processing
            success_rate = 0.95 if not requires_manager_approval else 0.85
            if random.random() < success_rate:
                
                transaction_id = f"REF_{order_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                refund_data = {
                    "transaction_id": transaction_id,
                    "order_id": order_id,
                    "refund_amount": amount,
                    "refund_type": refund_type,
                    "reason": reason,
                    "status": "approved" if not requires_manager_approval else "pending_approval",
                    "processed_at": datetime.now().isoformat(),
                    "estimated_completion": (datetime.now() + timedelta(hours=2 if not requires_manager_approval else 24)).isoformat(),
                    "payment_method": "original_payment_method",
                    "additional_details": {
                        "requires_manager_approval": requires_manager_approval,
                        "approval_status": "auto_approved" if not requires_manager_approval else "pending",
                        "customer_notification_sent": True
                    }
                }
                
                # Add type-specific details
                if refund_type == "refund_plus_credit":
                    refund_data["credit_amount"] = amount * 0.5
                    refund_data["total_compensation"] = amount * 1.5
                elif refund_type == "credit":
                    refund_data["credit_amount"] = amount
                    refund_data["credit_expiry"] = (datetime.now() + timedelta(days=90)).isoformat()
                
                logger.info(f"Refund processed for order {order_id}: ${amount} ({refund_type})")
                return self._ok_result(refund_data)
            else:
                return self._err_result("Refund processing failed - system error or approval denied")
                
        except Exception as e:
            logger.error(f"Error processing refund: {e}")
            return self._err_result(f"Refund processing failed: {str(e)}")
    
    def _ok_result(self, data: Dict[str, Any]) -> ToolResult:
        """Helper to create successful result."""
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            execution_time=random.uniform(0.5, 2.0) if self._seed else 1.0
        )
    
    def _err_result(self, error: str) -> ToolResult:
        """Helper to create error result."""
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={"error": error},
            execution_time=random.uniform(0.3, 1.0) if self._seed else 0.5
        )

class ExonerateDriverTool:
    """Tool for exonerating drivers when fault lies elsewhere."""
    
    def __init__(self):
        self.name = "exonerate_driver"
        self.description = "Clear driver of fault when evidence shows they are not responsible"
        self._seed = None
    
    def execute(self, order_id: str, driver_id: Optional[str] = None,
                reason: Optional[str] = None, _seed: Optional[int] = None) -> ToolResult:
        """
        Exonerate driver from fault in a dispute.
        
        Args:
            order_id: The order ID related to the dispute
            driver_id: ID of the driver to exonerate (optional)
            reason: Reason for exoneration (optional)
            _seed: Random seed for deterministic testing
        """
        try:
            if _seed is not None:
                self._seed = _seed
                random.seed(_seed)
            
            if not order_id:
                return self._err_result("Missing required parameter: order_id")
            
            driver_id = driver_id or f"DRV_{random.randint(1000, 9999)}"
            reason = reason or "Evidence shows driver followed proper procedures"
            
            # Simulate exoneration process
            success_rate = 0.95
            if random.random() < success_rate:
                
                exoneration_data = {
                    "order_id": order_id,
                    "driver_id": driver_id,
                    "exoneration_status": "cleared",
                    "reason": reason,
                    "processed_at": datetime.now().isoformat(),
                    "actions_taken": [
                        "Removed negative mark from driver record",
                        "Updated dispute resolution with driver exoneration",
                        "Notified driver of cleared status"
                    ],
                    "driver_record_impact": {
                        "rating_restored": True,
                        "performance_metrics_updated": True,
                        "negative_marks_removed": 1
                    }
                }
                
                logger.info(f"Driver {driver_id} exonerated for order {order_id}")
                return self._ok_result(exoneration_data)
            else:
                return self._err_result("Driver exoneration process failed")
                
        except Exception as e:
            logger.error(f"Error exonerating driver: {e}")
            return self._err_result(f"Driver exoneration failed: {str(e)}")
    
    def _ok_result(self, data: Dict[str, Any]) -> ToolResult:
        """Helper to create successful result."""
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            execution_time=random.uniform(0.5, 1.5) if self._seed else 1.0
        )
    
    def _err_result(self, error: str) -> ToolResult:
        """Helper to create error result."""
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={"error": error},
            execution_time=random.uniform(0.2, 0.8) if self._seed else 0.5
        )

class LogMerchantPackagingFeedbackTool:
    """Tool for logging feedback to merchants about packaging issues."""
    
    def __init__(self):
        self.name = "log_merchant_packaging_feedback"
        self.description = "Log feedback to merchants about packaging and quality issues"
        self._seed = None
    
    def execute(self, merchant_id: str, report: Dict[str, Any],
                severity: str = "medium", _seed: Optional[int] = None) -> ToolResult:
        """
        Log feedback to merchant about packaging or quality issues.
        
        Args:
            merchant_id: ID of the merchant to receive feedback
            report: Report details about the issue
            severity: Severity level ('low', 'medium', 'high', 'critical')
            _seed: Random seed for deterministic testing
        """
        try:
            if _seed is not None:
                self._seed = _seed
                random.seed(_seed)
            
            if not merchant_id or not report:
                return self._err_result("Missing required parameters: merchant_id and report")
            
            # Simulate feedback logging
            success_rate = 0.9
            if random.random() < success_rate:
                
                feedback_id = f"FB_{merchant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                feedback_data = {
                    "feedback_id": feedback_id,
                    "merchant_id": merchant_id,
                    "report": report,
                    "severity": severity,
                    "logged_at": datetime.now().isoformat(),
                    "status": "submitted",
                    "follow_up_required": severity in ["high", "critical"],
                    "merchant_notification": {
                        "method": "email_and_dashboard",
                        "sent_at": datetime.now().isoformat(),
                        "response_deadline": (datetime.now() + timedelta(days=3 if severity != "critical" else 1)).isoformat()
                    },
                    "tracking": {
                        "case_number": feedback_id,
                        "priority": severity,
                        "assigned_to": "merchant_relations_team"
                    }
                }
                
                # Add severity-specific actions
                if severity == "critical":
                    feedback_data["immediate_actions"] = [
                        "Merchant contacted immediately",
                        "Quality assurance team notified",
                        "Temporary quality monitoring activated"
                    ]
                elif severity == "high":
                    feedback_data["immediate_actions"] = [
                        "Merchant notified within 4 hours",
                        "Quality review scheduled"
                    ]
                
                logger.info(f"Merchant feedback logged: {feedback_id} for merchant {merchant_id}")
                return self._ok_result(feedback_data)
            else:
                return self._err_result("Failed to log merchant feedback")
                
        except Exception as e:
            logger.error(f"Error logging merchant feedback: {e}")
            return self._err_result(f"Merchant feedback logging failed: {str(e)}")
    
    def _ok_result(self, data: Dict[str, Any]) -> ToolResult:
        """Helper to create successful result."""
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            execution_time=random.uniform(0.5, 1.5) if self._seed else 1.0
        )
    
    def _err_result(self, error: str) -> ToolResult:
        """Helper to create error result."""
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={"error": error},
            execution_time=random.uniform(0.2, 0.8) if self._seed else 0.5
        )