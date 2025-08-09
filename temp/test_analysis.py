#!/usr/bin/env python3
"""
Test TradingAgents analysis execution to find where it gets stuck
"""

import os
import sys
import signal
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

print("🔍 TradingAgents Analysis Test")
print("=" * 50)

# Create minimal config to prevent loops
config = DEFAULT_CONFIG.copy()
config.update({
    "max_recur_limit": 2,  # Very low retry limit
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "online_tools": True,
    "deep_think_llm": "gpt-4o-mini",
    "quick_think_llm": "gpt-4o-mini"
})

print("⚙️ Configuration:")
print(f"   Max retries: {config['max_recur_limit']}")
print(f"   Debate rounds: {config['max_debate_rounds']}")
print(f"   LLM model: {config['deep_think_llm']}")

# Initialize framework
print("\n🤖 Initializing TradingAgents...")
ta = TradingAgentsGraph(debug=True, config=config)
print("✅ Framework initialized")

# Test with a simple, reliable stock
stock_symbol = "AAPL"
date_str = "2024-01-15"  # A known good trading day

print(f"\n📊 Testing analysis: {stock_symbol} on {date_str}")
print("⏱️ Starting analysis with 60-second timeout...")

# Set up timeout mechanism
analysis_completed = False
analysis_result = None
analysis_error = None

def run_analysis():
    global analysis_completed, analysis_result, analysis_error
    try:
        print("🔄 Starting ta.propagate()...")
        result, decision = ta.propagate(stock_symbol, date_str)
        analysis_result = (result, decision)
        analysis_completed = True
        print("✅ Analysis completed successfully!")
    except Exception as e:
        analysis_error = str(e)
        analysis_completed = True
        print(f"❌ Analysis failed: {e}")

# Run analysis in thread with timeout
analysis_thread = threading.Thread(target=run_analysis)
analysis_thread.daemon = True
analysis_thread.start()

# Wait with timeout
timeout_seconds = 60
for i in range(timeout_seconds):
    if analysis_completed:
        break
    if i % 10 == 0:
        print(f"⏳ Waiting... ({i}s elapsed)")
    analysis_thread.join(timeout=1)

if not analysis_completed:
    print(f"⚠️ Analysis timed out after {timeout_seconds} seconds")
    print("🔄 This confirms the infinite loop issue in ta.propagate()")
    print("\n💡 Recommendations:")
    print("1. The TradingAgents framework has internal retry loops")
    print("2. Need to patch the framework or implement workaround")
    print("3. Consider using mock data for demo purposes")
else:
    if analysis_error:
        print(f"❌ Analysis failed: {analysis_error}")
    else:
        print("🎉 Analysis completed successfully!")
        print(f"📊 Result type: {type(analysis_result[0])}")
        print(f"🎯 Decision type: {type(analysis_result[1])}")
