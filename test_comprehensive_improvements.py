#!/usr/bin/env python3
"""
Comprehensive test script to validate all improvements made to the autonomous delivery coordinator.
Tests the original problematic scenarios and new complex scenarios.
"""

import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    from src.scenarios.interactive_input import InteractiveScenarioTester
except ImportError:
    # Try alternative import
    import os
    os.chdir('src')
    from scenarios.interactive_input import InteractiveScenarioTester


class ComprehensiveTestSuite:
    """Comprehensive test suite for the improved autonomous delivery coordinator."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.tester = InteractiveScenarioTester()
        self.test_results = []
        
    def run_all_tests(self):
        """Run all comprehensive tests."""
        print("🧪 COMPREHENSIVE TEST SUITE - Autonomous Delivery Coordinator")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if not self.tester.agent:
            print("❌ CRITICAL: Agent initialization failed - cannot run tests")
            return
        
        print(f"✅ Agent initialized successfully with {len(self.tester.tool_manager.get_available_tools())} tools")
        print()
        
        # Test categories
        test_categories = [
            ("Original Problematic Scenarios", self._test_original_problems),
            ("Multi-Delivery Priority Scenarios", self._test_multi_delivery_priority),
            ("Emergency Medical Scenarios", self._test_emergency_medical),
            ("Complex Multi-Factor Scenarios", self._test_complex_multi_factor),
            ("Edge Case Scenarios", self._test_edge_cases)
        ]
        
        for category_name, test_method in test_categories:
            print(f"📋 {category_name}")
            print("-" * 60)
            test_method()
            print()
        
        # Generate summary report
        self._generate_summary_report()
    
    def _test_original_problems(self):
        """Test the original problematic scenarios that were identified."""
        scenarios = [
            {
                "name": "Multi-Factor Crisis (Original Problem)",
                "scenario": "URGENT: Emergency medical supplies delivery DEL555 to hospital is delayed due to both traffic accident on Route 9 and restaurant kitchen fire. Patient waiting for critical medication. Driver Mike needs immediate rerouting and customer notification required.",
                "expected_type": "multi_factor",
                "expected_urgency": "critical",
                "expected_min_steps": 4,
                "expected_min_probability": 0.80
            },
            {
                "name": "Traffic Emergency (Missing Tools)",
                "scenario": "Major highway closure affecting multiple deliveries. Emergency situation requires immediate escalation to support team.",
                "expected_tools": ["check_traffic", "escalate_to_support", "notify_customer"],
                "expected_min_steps": 3,
                "expected_min_probability": 0.75
            },
            {
                "name": "Customer Complaint (Generic Plans)",
                "scenario": "Customer Jane at 789 Pine Avenue is very angry about cold food delivery DEL123. She demands immediate refund and compensation.",
                "expected_specific_plan": True,
                "expected_min_steps": 4,
                "expected_min_probability": 0.85
            }
        ]
        
        for scenario_data in scenarios:
            self._run_single_test(scenario_data)
    
    def _test_multi_delivery_priority(self):
        """Test scenarios with multiple deliveries requiring prioritization."""
        scenarios = [
            {
                "name": "Medical Priority Multi-Delivery",
                "scenario": "Driver Sarah has 3 deliveries: DEL001 (pizza to office), DEL002 (medical supplies to hospital), DEL003 (groceries to home). Traffic jam on main route affecting all deliveries. Hospital patient needs medication urgently.",
                "expected_type": "multi_factor",
                "expected_urgency": "critical",
                "expected_min_steps": 5,
                "expected_min_probability": 0.75,
                "priority_keywords": ["medical", "priority", "hospital"]
            },
            {
                "name": "Time-Sensitive Multi-Order",
                "scenario": "Restaurant fire affects 5 orders: DEL100, DEL101, DEL102, DEL103, DEL104. Need to find alternative merchants and coordinate with customers. One order is for a business meeting in 30 minutes.",
                "expected_type": "multi_factor",
                "expected_urgency": "high",
                "expected_min_steps": 4,
                "expected_min_probability": 0.70
            }
        ]
        
        for scenario_data in scenarios:
            self._run_single_test(scenario_data)
    
    def _test_emergency_medical(self):
        """Test emergency medical delivery scenarios."""
        scenarios = [
            {
                "name": "Critical Medical Emergency",
                "scenario": "EMERGENCY: Insulin delivery DEL999 for diabetic patient stuck due to bridge collapse. Patient in critical condition, needs medication within 1 hour. Driver Tom cannot reach destination.",
                "expected_type": "multi_factor",
                "expected_urgency": "critical",
                "expected_min_steps": 5,
                "expected_min_probability": 0.80,
                "required_stakeholders": ["Emergency Response Team", "Medical Logistics Team"]
            },
            {
                "name": "Hospital Supply Chain Disruption",
                "scenario": "Multiple medical supply deliveries (DEL800, DEL801, DEL802) to City Hospital delayed due to supplier warehouse fire. ICU needs supplies for 3 critical patients. Alternative suppliers must be found immediately.",
                "expected_type": "multi_factor",
                "expected_urgency": "critical",
                "expected_min_steps": 6,
                "expected_min_probability": 0.75
            }
        ]
        
        for scenario_data in scenarios:
            self._run_single_test(scenario_data)
    
    def _test_complex_multi_factor(self):
        """Test complex multi-factor scenarios with multiple simultaneous issues."""
        scenarios = [
            {
                "name": "System-Wide Disruption",
                "scenario": "City-wide power outage affects GPS systems, restaurant POS systems down, and traffic lights not working. 15 active deliveries affected including 2 medical deliveries. Drivers cannot navigate or communicate effectively.",
                "expected_type": "multi_factor",
                "expected_urgency": "critical",
                "expected_min_steps": 6,
                "expected_min_probability": 0.70,
                "complexity_indicators": ["city-wide", "multiple systems", "15 deliveries"]
            },
            {
                "name": "Weather + Infrastructure Crisis",
                "scenario": "Severe storm causes flooding on Highway 1, power outage at main distribution center, and 3 restaurants closed. 8 deliveries affected: DEL200-DEL207. Customer complaints increasing. Need comprehensive coordination.",
                "expected_type": "multi_factor",
                "expected_urgency": "high",
                "expected_min_steps": 5,
                "expected_min_probability": 0.75
            }
        ]
        
        for scenario_data in scenarios:
            self._run_single_test(scenario_data)
    
    def _test_edge_cases(self):
        """Test edge cases and unusual scenarios."""
        scenarios = [
            {
                "name": "Driver Emergency",
                "scenario": "Driver Carlos had accident while delivering DEL500. He's okay but vehicle damaged. Customer waiting, food getting cold. Need immediate replacement driver and customer communication.",
                "expected_urgency": "high",
                "expected_min_steps": 4,
                "expected_min_probability": 0.80
            },
            {
                "name": "Address Validation Crisis",
                "scenario": "Customer provided wrong address for DEL600. Driver at location but no one there. Customer not answering phone. Food will be wasted if not delivered soon. Need address verification and customer contact.",
                "expected_type": "other",
                "expected_urgency": "medium",
                "expected_min_steps": 3,
                "expected_min_probability": 0.75
            }
        ]
        
        for scenario_data in scenarios:
            self._run_single_test(scenario_data)
    
    def _run_single_test(self, scenario_data: Dict[str, Any]):
        """Run a single test scenario and validate results."""
        print(f"🔍 Testing: {scenario_data['name']}")
        
        result = self.tester.test_scenario(scenario_data['scenario'])
        
        # Track result
        test_result = {
            "name": scenario_data['name'],
            "scenario": scenario_data['scenario'],
            "result": result,
            "validations": {}
        }
        
        if result['success']:
            print(f"   ✅ Scenario processed successfully")
            print(f"   📊 Type: {result['scenario_type']}")
            print(f"   ⚡ Urgency: {result['urgency_level']}")
            print(f"   🔢 Entities: {result['entities_found']}")
            print(f"   🧠 Reasoning Steps: {result['reasoning_steps']}")
            print(f"   📋 Plan Steps: {result['plan_steps']}")
            print(f"   🎯 Success Probability: {result['success_probability']:.1%}")
            print(f"   👥 Stakeholders: {', '.join(result['stakeholders'])}")
            
            # Validate expectations
            validations = self._validate_expectations(result, scenario_data)
            test_result["validations"] = validations
            
            # Print validation results
            for validation, passed in validations.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {validation}")
        else:
            print(f"   ❌ FAILED: {result['error']}")
            test_result["validations"]["processing"] = False
        
        self.test_results.append(test_result)
        print()
    
    def _validate_expectations(self, result: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, bool]:
        """Validate test results against expectations."""
        validations = {}
        
        # Scenario type validation
        if "expected_type" in expected:
            validations["Scenario Type"] = result['scenario_type'].lower() == expected['expected_type']
        
        # Urgency validation
        if "expected_urgency" in expected:
            validations["Urgency Level"] = result['urgency_level'].lower() == expected['expected_urgency']
        
        # Minimum steps validation
        if "expected_min_steps" in expected:
            validations[f"Min {expected['expected_min_steps']} Steps"] = result['plan_steps'] >= expected['expected_min_steps']
        
        # Minimum probability validation
        if "expected_min_probability" in expected:
            validations[f"Min {expected['expected_min_probability']:.0%} Success Rate"] = result['success_probability'] >= expected['expected_min_probability']
        
        # Priority keywords validation
        if "priority_keywords" in expected:
            scenario_lower = expected['scenario'].lower()
            has_keywords = any(keyword in scenario_lower for keyword in expected['priority_keywords'])
            validations["Priority Keywords Present"] = has_keywords
        
        # Required stakeholders validation
        if "required_stakeholders" in expected:
            stakeholder_text = ' '.join(result['stakeholders']).lower()
            has_required = all(stakeholder.lower() in stakeholder_text 
                             for stakeholder in expected['required_stakeholders'])
            validations["Required Stakeholders"] = has_required
        
        # Complexity indicators validation
        if "complexity_indicators" in expected:
            scenario_lower = expected['scenario'].lower()
            has_complexity = any(indicator in scenario_lower for indicator in expected['complexity_indicators'])
            validations["Complexity Indicators"] = has_complexity
        
        return validations
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("📊 COMPREHENSIVE TEST SUMMARY REPORT")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['result']['success'])
        
        print(f"Total Tests Run: {total_tests}")
        print(f"Successful Executions: {successful_tests}/{total_tests} ({successful_tests/total_tests:.1%})")
        print()
        
        # Success probability analysis
        success_probs = [result['result']['success_probability'] 
                        for result in self.test_results if result['result']['success']]
        if success_probs:
            avg_success_prob = sum(success_probs) / len(success_probs)
            min_success_prob = min(success_probs)
            max_success_prob = max(success_probs)
            
            print(f"Success Probability Analysis:")
            print(f"  Average: {avg_success_prob:.1%}")
            print(f"  Range: {min_success_prob:.1%} - {max_success_prob:.1%}")
            print()
        
        # Plan complexity analysis
        plan_steps = [result['result']['plan_steps'] 
                     for result in self.test_results if result['result']['success']]
        if plan_steps:
            avg_steps = sum(plan_steps) / len(plan_steps)
            min_steps = min(plan_steps)
            max_steps = max(plan_steps)
            
            print(f"Plan Complexity Analysis:")
            print(f"  Average Steps: {avg_steps:.1f}")
            print(f"  Range: {min_steps} - {max_steps} steps")
            print()
        
        # Validation analysis
        all_validations = {}
        for result in self.test_results:
            for validation, passed in result['validations'].items():
                if validation not in all_validations:
                    all_validations[validation] = []
                all_validations[validation].append(passed)
        
        print("Validation Success Rates:")
        for validation, results in all_validations.items():
            success_rate = sum(results) / len(results)
            print(f"  {validation}: {sum(results)}/{len(results)} ({success_rate:.1%})")
        print()
        
        # Improvement recommendations
        print("🚀 IMPROVEMENT RECOMMENDATIONS:")
        
        if avg_success_prob < 0.85:
            print("  • Enhance success probability calculation - target 85%+")
        
        if avg_steps < 4:
            print("  • Increase plan detail - target 4+ steps for comprehensive coverage")
        
        low_validation_rates = [(v, sum(r)/len(r)) for v, r in all_validations.items() if sum(r)/len(r) < 0.8]
        if low_validation_rates:
            print("  • Address validation failures:")
            for validation, rate in low_validation_rates:
                print(f"    - {validation}: {rate:.1%} success rate")
        
        print()
        print("🎯 OVERALL ASSESSMENT:")
        
        if successful_tests == total_tests and avg_success_prob >= 0.80 and avg_steps >= 3:
            print("  ✅ EXCELLENT: System performing at production-ready levels")
        elif successful_tests >= total_tests * 0.9 and avg_success_prob >= 0.75:
            print("  ✅ GOOD: System performing well with minor improvements needed")
        elif successful_tests >= total_tests * 0.8:
            print("  ⚠️  ACCEPTABLE: System functional but needs improvement")
        else:
            print("  ❌ NEEDS WORK: Significant improvements required")
        
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function to run comprehensive tests."""
    test_suite = ComprehensiveTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()