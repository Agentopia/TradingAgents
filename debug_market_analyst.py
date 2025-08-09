#!/usr/bin/env python3
"""
Deep debugging script to monitor Market Analyst execution and identify hang points
"""

import os
import sys
import logging
import time
import threading
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_analyst_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

print("🔍 Market Analyst Deep Debug Monitor")
print("=" * 60)

# Import TradingAgents with debug logging
logger.info("Importing TradingAgents framework...")
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create minimal config
config = DEFAULT_CONFIG.copy()
config.update({
    "max_recur_limit": 5,  # Low but not too aggressive
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "online_tools": True,
    "deep_think_llm": "gpt-4o-mini",
    "quick_think_llm": "gpt-4o-mini"
})

logger.info(f"Configuration: {config}")

# Initialize framework with debug mode
logger.info("Initializing TradingAgents with debug=True...")
ta = TradingAgentsGraph(debug=True, config=config)
logger.info("✅ TradingAgents initialized successfully")

# Test parameters
stock_symbol = "AAPL"
date_str = "2024-01-15"

print(f"\n📊 Starting monitored analysis: {stock_symbol} on {date_str}")
print("🔍 Monitoring every 5 seconds with detailed logging...")

# Create a monitoring thread to track progress
analysis_start_time = time.time()
last_log_time = time.time()

def monitor_progress():
    """Monitor and log analysis progress"""
    global last_log_time
    
    while True:
        current_time = time.time()
        elapsed = current_time - analysis_start_time
        
        if current_time - last_log_time >= 5:  # Log every 5 seconds
            logger.info(f"⏱️ Analysis running for {elapsed:.1f} seconds...")
            last_log_time = current_time
        
        time.sleep(1)

# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
monitor_thread.start()

# Monkey patch key TradingAgents methods to add logging
original_propagate = ta.propagate

def logged_propagate(stock_symbol, date_str):
    logger.info(f"🚀 Starting ta.propagate({stock_symbol}, {date_str})")
    
    try:
        # Try to access the internal graph to add more logging
        if hasattr(ta, 'graph'):
            logger.info(f"📊 TradingAgents graph type: {type(ta.graph)}")
            
            # If it's a LangGraph, we can potentially monitor node execution
            if hasattr(ta.graph, 'nodes'):
                logger.info(f"🔗 Graph nodes: {list(ta.graph.nodes.keys()) if hasattr(ta.graph.nodes, 'keys') else 'Unknown'}")
        
        logger.info("🔄 Calling original propagate method...")
        result = original_propagate(stock_symbol, date_str)
        logger.info("✅ ta.propagate completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"❌ ta.propagate failed with error: {e}")
        logger.exception("Full traceback:")
        raise

# Replace the method
ta.propagate = logged_propagate

try:
    logger.info("🎯 Starting analysis execution...")
    result, decision = ta.propagate(stock_symbol, date_str)
    
    elapsed_time = time.time() - analysis_start_time
    logger.info(f"🎉 Analysis completed successfully in {elapsed_time:.1f} seconds!")
    logger.info(f"📊 Result type: {type(result)}")
    logger.info(f"🎯 Decision type: {type(decision)}")
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS: Market Analyst completed without hanging!")
    print(f"⏱️ Total time: {elapsed_time:.1f} seconds")
    print("📋 Check 'market_analyst_debug.log' for detailed execution log")

except KeyboardInterrupt:
    elapsed_time = time.time() - analysis_start_time
    logger.warning(f"⚠️ Analysis interrupted by user after {elapsed_time:.1f} seconds")
    print(f"\n⚠️ Analysis interrupted after {elapsed_time:.1f} seconds")
    print("📋 Check 'market_analyst_debug.log' for execution details up to interruption")

except Exception as e:
    elapsed_time = time.time() - analysis_start_time
    logger.error(f"❌ Analysis failed after {elapsed_time:.1f} seconds: {e}")
    logger.exception("Full error traceback:")
    print(f"\n❌ Analysis failed after {elapsed_time:.1f} seconds")
    print(f"Error: {e}")
    print("📋 Check 'market_analyst_debug.log' for detailed error information")

print("\n🔍 Debug session completed. Log file: market_analyst_debug.log")
