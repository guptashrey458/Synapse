#!/usr/bin/env python3
"""
Demonstration of the tool management system with error handling and resilience features.
"""

import time
import logging
from src.tools import (
    ConcreteToolManager, CheckTrafficTool, GetMerchantStatusTool, 
    NotifyCustomerTool, ErrorCategory
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate tool management system capabilities."""
    print("🚀 Autonomous Delivery Coordinator - Tool Management System Demo")
    print("=" * 70)
    
    # Initialize tool manager with advanced features
    manager = ConcreteToolManager(
        max_workers=4,
        default_timeout=10,
        enable_caching=True,
        cache_ttl=300,
        enable_circuit_breaker=True
    )
    
    try:
        # Register logistics tools
        print("\n📋 Registering logistics tools...")
        traffic_tool = CheckTrafficTool()
        merchant_tool = GetMerchantStatusTool()
        notify_tool = NotifyCustomerTool()
        
        manager.register_tool(traffic_tool)
        manager.register_tool(merchant_tool)
        manager.register_tool(notify_tool)
        
        available_tools = manager.get_available_tools()
        print(f"✅ Registered {len(available_tools)} tools: {[tool.name for tool in available_tools]}")
        
        # Demonstrate tool execution
        print("\n🔧 Executing tools...")
        
        # 1. Check traffic conditions
        print("\n1. Checking traffic conditions...")
        traffic_result = manager.execute_tool(
            "check_traffic",
            {
                "origin": "123 Main Street, San Francisco, CA",
                "destination": "456 Oak Avenue, San Francisco, CA",
                "delivery_id": "DEL_12345"
            }
        )
        
        if traffic_result.success:
            route_data = traffic_result.data["route"]
            print(f"   ✅ Traffic check successful!")
            print(f"   📍 Distance: {route_data['distance_miles']} miles")
            print(f"   ⏱️  Base time: {route_data['base_time_minutes']} minutes")
            print(f"   🚦 Total delay: {route_data['total_delay_minutes']} minutes")
            print(f"   💡 Recommendations: {len(traffic_result.data['recommendations'])} items")
        else:
            print(f"   ❌ Traffic check failed: {traffic_result.error_message}")
        
        # 2. Check merchant status
        print("\n2. Checking merchant status...")
        merchant_result = manager.execute_tool(
            "get_merchant_status",
            {
                "merchant_id": "MERCH_789",
                "include_queue_info": True
            }
        )
        
        if merchant_result.success:
            print(f"   ✅ Merchant check successful!")
            print(f"   🏪 Status: {merchant_result.data['status']}")
            print(f"   ⏲️  Prep time: {merchant_result.data['current_prep_time_minutes']} minutes")
            print(f"   📊 Capacity: {merchant_result.data['capacity_utilization']*100:.1f}%")
            if 'queue_info' in merchant_result.data:
                queue_info = merchant_result.data['queue_info']
                print(f"   📋 Queue: {queue_info['orders_in_queue']} orders ({queue_info['queue_status']})")
        else:
            print(f"   ❌ Merchant check failed: {merchant_result.error_message}")
        
        # 3. Send customer notification
        print("\n3. Sending customer notification...")
        notify_result = manager.execute_tool(
            "notify_customer",
            {
                "delivery_id": "DEL_12345",
                "customer_id": "CUST_456",
                "message_type": "delay_notification",
                "urgency": "medium"
            }
        )
        
        if notify_result.success:
            print(f"   ✅ Notification sent successfully!")
            print(f"   📱 Channel: {notify_result.data['channel']}")
            print(f"   📨 Notification ID: {notify_result.data['notification_id']}")
            print(f"   ✉️  Delivered: {notify_result.data['delivery_confirmation']}")
            if notify_result.data.get('customer_response'):
                print(f"   💬 Customer response: {notify_result.data['customer_response']}")
        else:
            print(f"   ❌ Notification failed: {notify_result.error_message}")
        
        # Demonstrate caching
        print("\n🗄️  Demonstrating caching...")
        print("   Executing same traffic check again...")
        start_time = time.time()
        cached_result = manager.execute_tool(
            "check_traffic",
            {
                "origin": "123 Main Street, San Francisco, CA",
                "destination": "456 Oak Avenue, San Francisco, CA",
                "delivery_id": "DEL_12345"
            }
        )
        cache_time = time.time() - start_time
        print(f"   ⚡ Cached result returned in {cache_time:.3f} seconds")
        
        # Show performance metrics
        print("\n📊 Performance Metrics:")
        for tool_name in ["check_traffic", "get_merchant_status", "notify_customer"]:
            metrics = manager.get_performance_metrics(tool_name)
            if "error" not in metrics:
                print(f"   🔧 {tool_name}:")
                print(f"      Executions: {metrics['execution_count']}")
                print(f"      Success rate: {metrics['success_rate']*100:.1f}%")
                print(f"      Avg time: {metrics['average_execution_time']:.3f}s")
        
        # Show circuit breaker status
        print("\n🔌 Circuit Breaker Status:")
        cb_status = manager.get_circuit_breaker_status()
        if cb_status.get("circuit_breaker_enabled"):
            for tool_name, status in cb_status["tools"].items():
                print(f"   🔧 {tool_name}: {status['state']} (failures: {status['failure_count']})")
        else:
            print("   Circuit breaker disabled")
        
        # Show error statistics
        print("\n📈 Error Statistics (last 24 hours):")
        error_stats = manager.get_error_statistics(hours=24)
        if not error_stats.get("no_data"):
            print(f"   Total executions: {error_stats['total_executions']}")
            print(f"   Success rate: {error_stats['success_rate']*100:.1f}%")
            print(f"   Average execution time: {error_stats['average_execution_time']:.3f}s")
            if error_stats['error_categories']:
                print(f"   Error categories: {error_stats['error_categories']}")
        else:
            print("   No error data available")
        
        # Show execution history
        print("\n📜 Recent Execution History:")
        history = manager.get_execution_history(limit=5)
        for i, exec_info in enumerate(history[-5:], 1):
            status_icon = "✅" if exec_info['success'] else "❌"
            print(f"   {i}. {status_icon} {exec_info['tool_name']} - {exec_info['status']}")
            if exec_info['retry_count'] > 0:
                print(f"      (retried {exec_info['retry_count']} times)")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   Total tools registered: {len(manager.get_available_tools())}")
        print(f"   Total executions: {len(manager.get_execution_history())}")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        manager.shutdown()
        print("   Tool manager shutdown complete")

if __name__ == "__main__":
    main()