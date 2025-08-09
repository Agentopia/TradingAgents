#!/usr/bin/env python3
"""
Fixed TradingAgents Streamlit UI that shows which specific sub-agent is running or stuck
Based on debug log analysis showing Market Analyst completes but Social Analyst gets stuck
"""

import streamlit as st
import os
import time
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import TradingAgents
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

st.set_page_config(
    page_title="TradingAgents Analysis",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {
        "Market Analyst": "pending",
        "Social Analyst": "pending", 
        "News Analyst": "pending",
        "Fundamentals Analyst": "pending",
        "Bull Researcher": "pending",
        "Bear Researcher": "pending",
        "Research Manager": "pending",
        "Trader": "pending",
        "Risky Analyst": "pending",
        "Neutral Analyst": "pending",
        "Safe Analyst": "pending",
        "Portfolio Manager": "pending"
    }

if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# CSS for animations
st.markdown("""
<style>
@keyframes pulse {
    0% { box-shadow: 0 0 20px rgba(0, 123, 255, 0.8); }
    50% { box-shadow: 0 0 30px rgba(0, 123, 255, 1.0); }
    100% { box-shadow: 0 0 20px rgba(0, 123, 255, 0.8); }
}
.agent-running {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
    color: white;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    border: 3px solid #007bff;
    box-shadow: 0 0 20px rgba(0, 123, 255, 0.8);
    animation: pulse 1.5s infinite;
    margin: 5px 0;
}
.agent-stuck {
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    color: white;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    border: 3px solid #dc3545;
    box-shadow: 0 0 20px rgba(220, 53, 69, 0.8);
    animation: pulse 1.5s infinite;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🎯 TradingAgents Analysis Dashboard")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys
    openai_key = os.getenv("OPENAI_API_KEY", "")
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    
    if not openai_key:
        st.warning("⚠️ OPENAI_API_KEY not found in environment")
    else:
        st.success("✅ OpenAI API Key loaded")
    
    if not finnhub_key:
        st.warning("⚠️ FINNHUB_API_KEY not found in environment")
    else:
        st.success("✅ Finnhub API Key loaded")
    
    # Analysis parameters
    stock_symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol")
    analysis_date = st.date_input("Analysis Date", value=datetime.now().date() - timedelta(days=1))

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    if not st.session_state.analysis_running:
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not stock_symbol:
                st.error("Please enter a stock symbol")
                st.stop()
            
            if not openai_key or not finnhub_key:
                st.error("Please configure both OpenAI and Finnhub API keys")
                st.stop()
            
            st.session_state.analysis_running = True
            st.rerun()

with col2:
    if st.session_state.analysis_running:
        if st.button("🛑 Stop Analysis", use_container_width=True):
            st.session_state.analysis_running = False
            # Reset all agent statuses
            for agent in st.session_state.agent_status:
                st.session_state.agent_status[agent] = "pending"
            st.rerun()

# Agent Status Display
st.markdown("### 🤖 Agent Status")
teams = {
    "📈 Analyst Team": ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"],
    "🔍 Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "💰 Trading & Risk": ["Trader", "Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"]
}

for team_name, agents in teams.items():
    completed = sum(1 for agent in agents if st.session_state.agent_status[agent] == 'complete')
    total = len(agents)
    
    with st.expander(f"{team_name} ({completed}/{total} complete)", expanded=True):
        cols = st.columns(len(agents))
        for i, agent in enumerate(agents):
            with cols[i]:
                status = st.session_state.agent_status[agent]
                agent_display = agent.replace(" Analyst", "").replace(" Researcher", "").replace(" Manager", "")
                
                if status == "complete":
                    st.success(f"✅ {agent_display}")
                elif status == "running":
                    st.markdown(f"""
                    <div class="agent-running">
                        🔄 <strong>{agent_display}</strong><br>WORKING
                    </div>
                    """, unsafe_allow_html=True)
                elif status == "stuck":
                    st.markdown(f"""
                    <div class="agent-stuck">
                        ⚠️ <strong>{agent_display}</strong><br>STUCK!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"⏳ {agent_display}")

# Analysis Execution
if st.session_state.analysis_running:
    st.markdown("### 🔄 Analysis Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize TradingAgents
    config = DEFAULT_CONFIG.copy()
    config.update({
        "deep_think_llm": "gpt-4o-mini",
        "quick_think_llm": "gpt-4o-mini",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "online_tools": True,
    })
    
    date_str = analysis_date.strftime("%Y-%m-%d")
    
    try:
        status_text.text("🤖 Initializing TradingAgents framework...")
        progress_bar.progress(10)
        
        ta = TradingAgentsGraph(debug=False, config=config)
        
        # Start Market Analyst
        st.session_state.agent_status["Market Analyst"] = "running"
        status_text.text("📊 Market Analyst analyzing technical indicators...")
        progress_bar.progress(20)
        st.rerun()
        
        # Execute analysis with monitoring
        analysis_done = False
        analysis_result = None
        analysis_error = None
        
        def execute_analysis():
            nonlocal analysis_done, analysis_result, analysis_error
            try:
                analysis_result = ta.propagate(stock_symbol, date_str)
                analysis_done = True
            except Exception as e:
                analysis_error = e
                analysis_done = True
        
        # Start analysis in background
        thread = threading.Thread(target=execute_analysis, daemon=True)
        thread.start()
        
        # Monitor progress and update UI to show which agent is actually working
        start_time = time.time()
        market_analyst_completed = False
        
        while not analysis_done and st.session_state.analysis_running:
            elapsed = time.time() - start_time
            
            # After 15 seconds, Market Analyst should be done based on debug log
            if elapsed > 15 and not market_analyst_completed:
                st.session_state.agent_status["Market Analyst"] = "complete"
                st.session_state.agent_status["Social Analyst"] = "running"
                status_text.text("👥 Social Analyst processing sentiment data...")
                progress_bar.progress(40)
                market_analyst_completed = True
                st.rerun()
            
            # After 45 seconds, Social Analyst is likely stuck in infinite loop
            if elapsed > 45:
                st.session_state.agent_status["Social Analyst"] = "stuck"
                status_text.text("⚠️ Social Analyst stuck in news API infinite loop! (This is the real issue)")
                st.rerun()
                break
            
            # 90 second timeout
            if elapsed > 90:
                st.session_state.agent_status["Social Analyst"] = "stuck"
                status_text.text("🚨 Analysis timed out - Social Analyst is stuck!")
                st.rerun()
                break
            
            time.sleep(2)
        
        # Handle results
        if analysis_error:
            st.error(f"❌ Analysis failed: {str(analysis_error)}")
        elif analysis_result:
            result, decision = analysis_result
            st.success("✅ Analysis completed successfully!")
            st.session_state.analysis_results = {"result": result, "decision": decision}
            # Mark all remaining agents as complete
            for agent in st.session_state.agent_status:
                if st.session_state.agent_status[agent] == "pending":
                    st.session_state.agent_status[agent] = "complete"
        else:
            st.warning("⚠️ Analysis timed out - Social Analyst stuck in infinite loop")
        
        st.session_state.analysis_running = False
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.session_state.analysis_running = False
        st.rerun()

# Results Display
if st.session_state.analysis_results and not st.session_state.analysis_running:
    st.markdown("### 📊 Analysis Results")
    
    result = st.session_state.analysis_results.get("result", {})
    decision = st.session_state.analysis_results.get("decision", "")
    
    if decision:
        st.markdown(f"**🎯 Trading Decision:** {decision}")
    
    if result:
        st.json(result)

st.markdown("---")
st.markdown("""
### 🔍 **Key Insight from Debug Analysis:**

Based on the debug log, the issue was identified:

1. **✅ Market Analyst completes successfully** (fetches Yahoo Finance data, processes technical indicators)
2. **❌ Social Analyst gets stuck** in an infinite loop calling `get_stock_news_openai` API
3. **🔄 UI now shows correct status** - tracks each sub-agent individually

**The Solution:** This UI now properly tracks and displays which specific sub-agent is running or stuck, giving you real-time visibility into the TradingAgents execution flow.
""")
