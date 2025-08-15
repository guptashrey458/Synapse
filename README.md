# 🚚 Autonomous Delivery Coordinator

An AI-powered system for autonomous resolution of delivery disruption scenarios using advanced reasoning and tool execution.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

## 🎯 Overview

The Autonomous Delivery Coordinator is a sophisticated AI system that processes delivery disruption scenarios and generates intelligent resolution plans. It uses advanced reasoning patterns, tool execution, and performance optimization to handle complex multi-factor disruptions autonomously.

### ✨ Key Features

- **🧠 Intelligent Reasoning**: ReAct pattern implementation with multi-step reasoning
- **🔧 Tool Execution**: Dynamic tool selection and execution for scenario resolution
- **⚡ Performance Optimized**: Caching, concurrent execution, and response time optimization
- **🎨 Beautiful CLI**: Real-time progress display with rich formatting
- **📊 Comprehensive Analytics**: Performance metrics and reasoning trace analysis
- **🔄 Multi-Factor Handling**: Traffic, merchant, customer, and address disruptions
- **⚠️ Urgency Classification**: Dynamic priority assessment (Low → Critical)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/guptashrey458/Synapse.git
   cd Synapse
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system**
   ```bash
   cp config.example.json config.json
   # Edit config.json with your API keys and preferences
   ```

### Usage

#### Command Line Interface

**Process a single scenario:**
```bash
python -m src.cli.main --scenario "Traffic jam on Highway 101 affecting delivery DEL123 to customer John" --verbose
```

**Interactive mode:**
```bash
python -m src.cli.main interactive
```

**Check system status:**
```bash
python -m src.cli.main status
```

**View configuration:**
```bash
python -m src.cli.main config-info
```

#### Example Scenarios

**Simple Traffic Delay:**
```bash
python -m src.cli.main --scenario "Minor traffic backup on Route 1 for delivery DEL100"
```

**Multi-Factor Disruption:**
```bash
python -m src.cli.main --scenario "Restaurant fire at Pizza Palace, driver stuck in traffic, customer calling about delivery DEL456"
```

**Critical Emergency:**
```bash
python -m src.cli.main --scenario "URGENT: Medical delivery DEL911 carrying insulin stuck due to bridge collapse, 30-minute deadline"
```

## 🏗️ Architecture

### Core Components

```
src/
├── agent/                 # Autonomous agent core
│   ├── autonomous_agent.py    # Main agent orchestrator
│   ├── cache.py               # Performance caching
│   ├── concurrent_executor.py # Concurrent tool execution
│   └── models.py              # Data models and validation
├── reasoning/             # Reasoning engine
│   ├── engine.py              # ReAct reasoning implementation
│   ├── plan_generator.py      # Resolution plan generation
│   └── logger.py              # Reasoning trace logging
├── tools/                 # Tool execution system
│   ├── tool_manager.py        # Tool management and execution
│   ├── traffic_tools.py       # Traffic and routing tools
│   ├── merchant_tools.py      # Merchant status tools
│   └── communication_tools.py # Customer communication tools
├── llm/                   # LLM integration
│   ├── providers.py           # LLM provider implementations
│   └── usage_tracker.py       # Token usage tracking
├── cli/                   # Command line interface
│   ├── main.py                # CLI entry point
│   ├── progress_display.py    # Real-time progress display
│   └── output_formatter.py    # Result formatting
└── monitoring/            # System monitoring
    └── metrics_collector.py   # Performance metrics
```

### Key Design Patterns

- **ReAct Reasoning**: Thought → Action → Observation cycles
- **Tool Abstraction**: Pluggable tool system with validation
- **Performance Optimization**: Caching, concurrency, and batching
- **Error Resilience**: Graceful degradation and recovery
- **Modular Architecture**: Loosely coupled components

## 📊 Performance

### Benchmarks

| Scenario Type | Avg Response Time | Success Rate | Confidence Score |
|---------------|------------------|--------------|------------------|
| Simple Traffic | 22.2s | 100% | 70% |
| Multi-Factor | 22.4s | 100% | 70% |
| Critical Emergency | 22.4s | 100% | 60% |

### Optimization Features

- **Caching**: Tool result and scenario caching with TTL
- **Concurrent Execution**: Parallel tool execution for complex scenarios
- **Prompt Optimization**: Token-efficient LLM interactions
- **Batch Processing**: Grouped operations for efficiency

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test Categories

**Core Functionality:**
```bash
pytest tests/test_autonomous_agent_integration.py -v
```

**Performance Optimization:**
```bash
pytest tests/test_performance_optimization.py -v
```

**End-to-End Integration:**
```bash
pytest tests/test_comprehensive_scenarios.py -v
```

**Specific Disruption Types:**
```bash
pytest tests/test_traffic_disruption_scenarios.py -v
pytest tests/test_merchant_customer_scenarios.py -v
```

### Test Coverage

- ✅ **Unit Tests**: Individual component testing
- ✅ **Integration Tests**: Cross-component interaction testing
- ✅ **End-to-End Tests**: Complete workflow validation
- ✅ **Performance Tests**: Response time and scalability testing
- ✅ **Error Handling Tests**: Failure scenario validation

## 📈 Monitoring & Analytics

### Performance Metrics

The system tracks comprehensive performance metrics:

- **Response Times**: Average processing time per scenario type
- **Success Rates**: Tool execution and scenario resolution success
- **Resource Usage**: Memory and CPU utilization
- **Cache Performance**: Hit rates and efficiency gains
- **Reasoning Quality**: Confidence scores and completeness

### Accessing Metrics

```bash
python -m src.cli.main status  # Current system status
```

Or programmatically:
```python
from src.agent.autonomous_agent import AutonomousAgent

agent = AutonomousAgent(llm_provider, tool_manager)
metrics = agent.get_performance_metrics()
print(metrics)
```

## 🔧 Configuration

### Configuration File Structure

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-api-key",
    "temperature": 0.1,
    "max_tokens": 4000
  },
  "tools": {
    "traffic_api": {
      "enabled": true,
      "timeout": 10,
      "cache_ttl": 300
    },
    "communication": {
      "enabled": true,
      "sms_provider": "twilio"
    }
  },
  "reasoning": {
    "max_steps": 20,
    "confidence_threshold": 0.8
  },
  "cli": {
    "verbose": false,
    "output_format": "structured",
    "show_reasoning": true
  }
}
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export TWILIO_API_KEY="your-twilio-api-key"
export LOG_LEVEL="INFO"
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Run tests**
   ```bash
   pytest tests/ -v
   ```
5. **Commit changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for public methods
- Maintain test coverage above 90%

## 📝 API Reference

### Core Classes

#### AutonomousAgent
```python
from src.agent.autonomous_agent import AutonomousAgent, AgentConfig

config = AgentConfig(
    max_reasoning_steps=10,
    enable_caching=True,
    concurrent_tools=True
)

agent = AutonomousAgent(llm_provider, tool_manager, config)
result = agent.process_scenario("Traffic delay for delivery DEL123")
```

#### Tool Development
```python
from src.tools.interfaces import Tool, ToolResult

class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="Custom tool description",
            parameters={"param": {"type": "string"}}
        )
    
    def execute(self, parameters):
        # Tool implementation
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": "success"},
            execution_time=0.5
        )
```

## 🐛 Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
Error: OpenAI API key not found
Solution: Set OPENAI_API_KEY environment variable or update config.json
```

**2. Tool Execution Failures**
```bash
Error: Tool validation failed
Solution: Check tool parameters match expected schema
```

**3. Performance Issues**
```bash
Issue: Slow response times
Solution: Enable caching and concurrent execution in config
```

### Debug Mode

Enable verbose logging:
```bash
python -m src.cli.main --scenario "your scenario" --verbose
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4 language model capabilities
- **Click** for beautiful command-line interfaces
- **Pytest** for comprehensive testing framework
- **Rich** for terminal formatting and progress displays

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/guptashrey458/Synapse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/guptashrey458/Synapse/discussions)
- **Documentation**: [Wiki](https://github.com/guptashrey458/Synapse/wiki)

---

**Built with ❤️ for autonomous delivery coordination**

*Ready to revolutionize delivery disruption management with AI-powered autonomous reasoning!* 🚚✨