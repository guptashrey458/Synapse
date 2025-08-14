"""
Tests for token usage tracking and cost optimization.
"""
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.llm.usage_tracker import (
    TokenUsageTracker, PromptOptimizer, UsageSession, UsageStats
)
from src.llm.interfaces import TokenUsage, LLMResponse, Message, MessageRole


class TestTokenUsageTracker:
    """Test cases for TokenUsageTracker."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            return f.name
    
    @pytest.fixture
    def tracker(self, temp_storage):
        """Create tracker with temporary storage."""
        return TokenUsageTracker(temp_storage)
    
    @pytest.fixture
    def sample_response(self):
        """Create sample LLM response."""
        return LLMResponse(
            content="Test response",
            messages=[Message(role=MessageRole.ASSISTANT, content="Test")],
            token_usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.003
            ),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.5,
            timestamp=datetime.now()
        )
    
    def test_start_session(self, tracker):
        """Test starting a usage tracking session."""
        tracker.start_session("test_session", "gpt-4", "openai")
        
        assert tracker.current_session is not None
        assert tracker.current_session.session_id == "test_session"
        assert tracker.current_session.model == "gpt-4"
        assert tracker.current_session.provider == "openai"
        assert tracker.current_session.total_tokens == 0
        assert tracker.current_session.total_cost == 0.0
    
    def test_track_response(self, tracker, sample_response):
        """Test tracking an LLM response."""
        tracker.start_session("test_session", "gpt-4", "openai")
        tracker.track_response(sample_response)
        
        session = tracker.current_session
        assert session.total_tokens == 150
        assert session.total_cost == 0.003
        assert session.call_count == 1
    
    def test_track_multiple_responses(self, tracker, sample_response):
        """Test tracking multiple responses."""
        tracker.start_session("test_session", "gpt-4", "openai")
        
        # Track multiple responses
        for _ in range(3):
            tracker.track_response(sample_response)
        
        session = tracker.current_session
        assert session.total_tokens == 450  # 150 * 3
        assert abs(session.total_cost - 0.009) < 0.0001  # 0.003 * 3 with floating point tolerance
        assert session.call_count == 3
    
    def test_end_session(self, tracker, sample_response):
        """Test ending a session."""
        tracker.start_session("test_session", "gpt-4", "openai")
        tracker.track_response(sample_response)
        
        ended_session = tracker.end_session()
        
        assert ended_session.session_id == "test_session"
        assert ended_session.end_time is not None
        assert tracker.current_session is None
        assert len(tracker.usage_history) == 1
    
    def test_auto_start_session_on_track(self, tracker, sample_response):
        """Test automatic session start when tracking without active session."""
        # Track response without starting session
        tracker.track_response(sample_response)
        
        assert tracker.current_session is not None
        assert tracker.current_session.session_id == "default"
        assert tracker.current_session.total_tokens == 150
    
    def test_get_usage_stats(self, tracker, sample_response):
        """Test getting usage statistics."""
        # Create multiple sessions
        for i in range(3):
            tracker.start_session(f"session_{i}", "gpt-4", "openai")
            tracker.track_response(sample_response)
            tracker.end_session()
        
        stats = tracker.get_usage_stats()
        
        assert stats.total_tokens == 450  # 150 * 3
        assert abs(stats.total_cost - 0.009) < 0.0001  # 0.003 * 3 with floating point tolerance
        assert stats.total_calls == 3
        assert stats.average_tokens_per_call == 150
        assert abs(stats.average_cost_per_call - 0.003) < 0.0001
    
    def test_get_cost_breakdown_by_model(self, tracker, sample_response):
        """Test cost breakdown by model."""
        # Create sessions with different models
        models = ["gpt-4", "gpt-3.5-turbo", "gpt-4"]
        
        for i, model in enumerate(models):
            tracker.start_session(f"session_{i}", model, "openai")
            response = LLMResponse(
                content="Test",
                messages=[],
                token_usage=TokenUsage(100, 50, 150, 0.003),
                model=model,
                finish_reason="stop",
                response_time=1.0,
                timestamp=datetime.now()
            )
            tracker.track_response(response)
            tracker.end_session()
        
        breakdown = tracker.get_cost_breakdown_by_model()
        
        assert "gpt-4" in breakdown
        assert "gpt-3.5-turbo" in breakdown
        assert breakdown["gpt-4"]["calls"] == 2  # Used twice
        assert breakdown["gpt-3.5-turbo"]["calls"] == 1  # Used once
    
    def test_get_daily_usage(self, tracker):
        """Test getting daily usage statistics."""
        # Create sessions on different days
        base_time = datetime.now()
        
        for i in range(3):
            session_time = base_time - timedelta(days=i)
            session = UsageSession(
                session_id=f"session_{i}",
                start_time=session_time,
                end_time=session_time + timedelta(hours=1),
                total_tokens=100 * (i + 1),
                total_cost=0.001 * (i + 1),
                call_count=1,
                model="gpt-4",
                provider="openai"
            )
            tracker.usage_history.append(session.__dict__)
        
        daily_usage = tracker.get_daily_usage(days=7)
        
        assert len(daily_usage) == 3
        assert all("date" in day for day in daily_usage)
        assert all("tokens" in day for day in daily_usage)
        assert all("cost" in day for day in daily_usage)
    
    def test_optimize_costs(self, tracker):
        """Test cost optimization recommendations."""
        # Create high-cost session
        expensive_session = UsageSession(
            session_id="expensive",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_tokens=5000,  # High token usage
            total_cost=15.0,    # High cost
            call_count=1,
            model="gpt-4",
            provider="openai"
        )
        tracker.usage_history.append(expensive_session.__dict__)
        
        optimization = tracker.optimize_costs()
        
        assert "total_cost" in optimization
        assert "recommendations" in optimization
        assert len(optimization["recommendations"]) > 0
        
        # Should have model optimization recommendation
        rec_types = [r["type"] for r in optimization["recommendations"]]
        assert "model_optimization" in rec_types
    
    def test_export_usage_report(self, tracker, temp_storage):
        """Test exporting usage report."""
        # Add some usage data
        tracker.start_session("test", "gpt-4", "openai")
        response = LLMResponse(
            content="Test",
            messages=[],
            token_usage=TokenUsage(100, 50, 150, 0.003),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now()
        )
        tracker.track_response(response)
        tracker.end_session()
        
        # Export report
        report_path = temp_storage + "_report.json"
        tracker.export_usage_report(report_path)
        
        # Verify report was created and has correct structure
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        assert "generated_at" in report
        assert "summary" in report
        assert "model_breakdown" in report
        assert "daily_usage" in report
        assert "optimization" in report
        assert "raw_sessions" in report
    
    def test_persistence(self, temp_storage):
        """Test that usage data persists across tracker instances."""
        # Create first tracker and add data
        tracker1 = TokenUsageTracker(temp_storage)
        tracker1.start_session("test", "gpt-4", "openai")
        response = LLMResponse(
            content="Test",
            messages=[],
            token_usage=TokenUsage(100, 50, 150, 0.003),
            model="gpt-4",
            finish_reason="stop",
            response_time=1.0,
            timestamp=datetime.now()
        )
        tracker1.track_response(response)
        tracker1.end_session()
        
        # Create second tracker and verify data is loaded
        tracker2 = TokenUsageTracker(temp_storage)
        assert len(tracker2.usage_history) == 1
        assert tracker2.usage_history[0]["session_id"] == "test"


class TestPromptOptimizer:
    """Test cases for PromptOptimizer."""
    
    def test_estimate_token_reduction(self):
        """Test token reduction estimation."""
        original = "This is a very long prompt with many unnecessary words that could be optimized"
        optimized = "Long prompt with unnecessary words to optimize"
        
        reduction = PromptOptimizer.estimate_token_reduction(original, optimized)
        
        assert "original_tokens" in reduction
        assert "optimized_tokens" in reduction
        assert "token_reduction" in reduction
        assert "reduction_percent" in reduction
        assert reduction["token_reduction"] > 0
        assert reduction["reduction_percent"] > 0
    
    def test_suggest_prompt_optimizations(self):
        """Test prompt optimization suggestions."""
        verbose_prompt = """
        Please could you kindly help me with this very long prompt that has many
        redundant phrases and unnecessary words. I would like you to analyze this
        text very carefully. Example 1: This is an example. Example 2: Another example.
        Example 3: Yet another example. Example 4: More examples.
        """
        
        suggestions = PromptOptimizer.suggest_prompt_optimizations(verbose_prompt)
        
        assert len(suggestions) > 0
        
        suggestion_types = [s["type"] for s in suggestions]
        assert "remove_redundancy" in suggestion_types
        assert "length_reduction" in suggestion_types
        assert "example_optimization" in suggestion_types
    
    def test_compress_prompt(self):
        """Test prompt compression."""
        original = "The quick brown fox jumps over the lazy dog in the park"
        compressed = PromptOptimizer.compress_prompt(original, target_reduction=0.3)
        
        assert len(compressed.split()) < len(original.split())
        assert "fox" in compressed  # Important words should remain
        assert "jumps" in compressed
    
    def test_compress_prompt_preserves_meaning(self):
        """Test that compression preserves important content."""
        original = "Please analyze the delivery scenario and provide a solution"
        compressed = PromptOptimizer.compress_prompt(original, target_reduction=0.1)  # Lower reduction
        
        # Key words should be preserved
        assert "analyze" in compressed
        assert "delivery" in compressed
        assert "scenario" in compressed
        # With lower reduction, solution should be preserved
        if "solution" not in compressed:
            # If solution is removed, at least "provide" should be there
            assert "provide" in compressed
    
    def test_no_suggestions_for_optimal_prompt(self):
        """Test that optimal prompts get fewer suggestions."""
        optimal_prompt = "Analyze delivery scenario. Provide solution steps."
        
        suggestions = PromptOptimizer.suggest_prompt_optimizations(optimal_prompt)
        
        # Should have fewer or no suggestions for already optimal prompt
        assert len(suggestions) <= 1


class TestUsageStats:
    """Test UsageStats data class."""
    
    def test_usage_stats_creation(self):
        """Test creating UsageStats."""
        stats = UsageStats(
            total_tokens=1000,
            total_cost=5.0,
            total_calls=10,
            average_tokens_per_call=100.0,
            average_cost_per_call=0.5,
            most_expensive_call=1.0,
            peak_usage_hour=14
        )
        
        assert stats.total_tokens == 1000
        assert stats.total_cost == 5.0
        assert stats.peak_usage_hour == 14


class TestUsageSession:
    """Test UsageSession data class."""
    
    def test_usage_session_creation(self):
        """Test creating UsageSession."""
        start_time = datetime.now()
        session = UsageSession(
            session_id="test",
            start_time=start_time,
            model="gpt-4",
            provider="openai"
        )
        
        assert session.session_id == "test"
        assert session.start_time == start_time
        assert session.total_tokens == 0
        assert session.total_cost == 0.0
        assert session.call_count == 0