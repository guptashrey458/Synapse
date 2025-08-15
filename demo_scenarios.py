#!/usr/bin/env python3
"""
Demo script showing the input-based scenario system for the autonomous delivery coordinator.
"""

import json
from src.scenarios import ScenarioGenerator, ScenarioCategory, UrgencyLevel


def demo_scenario_generation():
    """Demonstrate scenario generation capabilities."""
    print("🚚 Autonomous Delivery Coordinator - Scenario Generation Demo")
    print("=" * 70)
    
    generator = ScenarioGenerator()
    
    # Show statistics
    stats = generator.get_scenario_statistics()
    print(f"📊 Available Templates: {stats['total_templates']}")
    print(f"📂 Categories: {', '.join(stats['categories'].keys())}")
    print(f"🎯 Average Complexity: {stats['average_complexity']:.1f}/10")
    print()
    
    # Generate examples from each category
    print("📚 Example Scenarios by Category:")
    print("-" * 50)
    
    for category in ScenarioCategory:
        try:
            scenario = generator.generate_scenario(category=category)
            print(f"\n📂 {category.value.upper()}")
            print(f"⚡ Urgency: {scenario['urgency'].title()}")
            print(f"🎯 Complexity: {scenario['complexity_score']}/10")
            print(f"📝 Scenario: {scenario['scenario_text']}")
            print(f"🔧 Expected Tools: {', '.join(scenario['expected_tools'])}")
            print(f"🎯 Expected Actions: {', '.join(scenario['expected_actions'])}")
            
        except ValueError:
            print(f"\n📂 {category.value.upper()}: No templates available")
    
    print("\n" + "=" * 70)
    print("🎲 Random Scenario Generation Examples:")
    print("-" * 50)
    
    # Generate random scenarios with different complexity levels
    for complexity in [3, 6, 9]:
        try:
            scenario = generator.generate_scenario(
                complexity_min=complexity, 
                complexity_max=complexity
            )
            print(f"\n🎯 Complexity Level {complexity}:")
            print(f"📂 Category: {scenario['category'].title()}")
            print(f"⚡ Urgency: {scenario['urgency'].title()}")
            print(f"📝 {scenario['scenario_text']}")
            
        except ValueError:
            print(f"\n🎯 Complexity Level {complexity}: No templates available")
    
    print("\n" + "=" * 70)
    print("📊 Training Dataset Generation:")
    print("-" * 50)
    
    # Generate a small training dataset
    dataset = generator.generate_training_dataset(scenarios_per_category=2)
    print(f"Generated {len(dataset)} training scenarios")
    
    # Show distribution
    category_counts = {}
    urgency_counts = {}
    
    for scenario in dataset:
        cat = scenario['category']
        urg = scenario['urgency']
        category_counts[cat] = category_counts.get(cat, 0) + 1
        urgency_counts[urg] = urgency_counts.get(urg, 0) + 1
    
    print("\n📂 Category Distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat.title()}: {count}")
    
    print("\n⚡ Urgency Distribution:")
    for urg, count in urgency_counts.items():
        print(f"  {urg.title()}: {count}")
    
    # Save sample dataset
    with open('sample_training_dataset.json', 'w') as f:
        json.dump(dataset[:5], f, indent=2, default=str)
    
    print(f"\n💾 Saved 5 sample scenarios to 'sample_training_dataset.json'")


def demo_specific_scenarios():
    """Demonstrate specific scenario types that the model should handle."""
    print("\n" + "=" * 70)
    print("🎯 Specific Scenario Types for Model Training:")
    print("=" * 70)
    
    generator = ScenarioGenerator()
    
    # Critical emergency scenarios
    print("\n🚨 CRITICAL EMERGENCY SCENARIOS:")
    print("-" * 40)
    
    for i in range(3):
        scenario = generator.generate_scenario(urgency=UrgencyLevel.CRITICAL)
        print(f"\n{i+1}. {scenario['scenario_text']}")
        print(f"   Expected Response: {', '.join(scenario['expected_actions'])}")
    
    # Multi-factor complex scenarios
    print("\n🔄 MULTI-FACTOR COMPLEX SCENARIOS:")
    print("-" * 40)
    
    for i in range(3):
        try:
            scenario = generator.generate_scenario(
                category=ScenarioCategory.MULTI_FACTOR,
                complexity_min=8
            )
            print(f"\n{i+1}. {scenario['scenario_text']}")
            print(f"   Tools Needed: {', '.join(scenario['expected_tools'])}")
            print(f"   Complexity: {scenario['complexity_score']}/10")
        except ValueError:
            continue
    
    # Customer service scenarios
    print("\n👥 CUSTOMER SERVICE SCENARIOS:")
    print("-" * 40)
    
    for i in range(3):
        try:
            scenario = generator.generate_scenario(category=ScenarioCategory.CUSTOMER)
            print(f"\n{i+1}. {scenario['scenario_text']}")
            print(f"   Resolution Focus: {', '.join(scenario['expected_actions'])}")
        except ValueError:
            continue


def demo_model_input_format():
    """Show the format that the model should expect as input."""
    print("\n" + "=" * 70)
    print("🤖 MODEL INPUT FORMAT EXAMPLES:")
    print("=" * 70)
    
    generator = ScenarioGenerator()
    
    # Generate a few scenarios and show the input format
    scenarios = generator.generate_scenario_batch(3)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 SCENARIO {i}:")
        print("-" * 20)
        print("INPUT FORMAT:")
        print(f"```")
        print(f"Scenario: {scenario['scenario_text']}")
        print(f"```")
        
        print("\nEXPECTED MODEL ANALYSIS:")
        print(f"- Category: {scenario['category']}")
        print(f"- Urgency: {scenario['urgency']}")
        print(f"- Complexity: {scenario['complexity_score']}/10")
        print(f"- Tools to use: {scenario['expected_tools']}")
        print(f"- Actions to take: {scenario['expected_actions']}")
        
        print("\nMODEL SHOULD DEDUCE:")
        print("1. Extract entities (delivery IDs, names, addresses, etc.)")
        print("2. Classify scenario type and urgency")
        print("3. Select appropriate tools for information gathering")
        print("4. Generate step-by-step reasoning")
        print("5. Create resolution plan with success probability")
        print("6. Identify stakeholders and communication needs")


if __name__ == "__main__":
    demo_scenario_generation()
    demo_specific_scenarios()
    demo_model_input_format()
    
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS:")
    print("=" * 70)
    print("1. Run interactive tester: python -m src.cli.main scenario-tester")
    print("2. Generate training data: Use ScenarioGenerator.generate_training_dataset()")
    print("3. Test specific scenarios: Use the generated JSON files")
    print("4. Train your model on the diverse scenario types")
    print("5. Validate model responses against expected tools/actions")
    print("\n🎯 Your model should learn to autonomously reason through")
    print("   delivery disruptions and generate intelligent solutions!")