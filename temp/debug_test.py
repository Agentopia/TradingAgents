#!/usr/bin/env python3
"""
Debug script to test TradingAgents framework step by step
"""

import os
import sys
from dotenv import load_dotenv

print("🔍 TradingAgents Debug Test")
print("=" * 50)

# Load environment variables
load_dotenv()
print("✅ Environment loaded")

# Check API keys
openai_key = os.getenv("OPENAI_API_KEY")
finnhub_key = os.getenv("FINNHUB_API_KEY")

print(f"🔑 OpenAI API Key: {'✅ Present' if openai_key else '❌ Missing'}")
print(f"🔑 Finnhub API Key: {'✅ Present' if finnhub_key else '❌ Missing'}")

if not openai_key or not finnhub_key:
    print("❌ Missing required API keys!")
    sys.exit(1)

# Test basic imports
try:
    print("\n📦 Testing imports...")
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test configuration
try:
    print("\n⚙️ Testing configuration...")
    config = DEFAULT_CONFIG.copy()
    config.update({
        "max_recur_limit": 2,  # Very low to prevent loops
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "online_tools": True
    })
    print("✅ Configuration created")
    print(f"📊 Config: {config}")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Test framework initialization
try:
    print("\n🤖 Testing framework initialization...")
    ta = TradingAgentsGraph(debug=True, config=config)
    print("✅ TradingAgents initialized successfully")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

print("\n🎉 All basic tests passed!")
print("The TradingAgents framework can be initialized.")
print("The issue is likely in the analysis execution (ta.propagate)")
