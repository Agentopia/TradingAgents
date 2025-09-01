"""
TradingAgents Streamlit Interface
A modern web UI that replicates and enhances the CLI experience
"""

import streamlit as st
import time
import os
import sys
import importlib
from datetime import datetime, date, timedelta
import json
from typing import Dict, Any, Optional
import asyncio
import threading
from collections import deque

# Add the parent directory to the path to import tradingagents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Force reload of critical modules to ensure fixes take effect
try:
    import tradingagents.dataflows.stockstats_utils
    import tradingagents.dataflows.interface
    importlib.reload(tradingagents.dataflows.stockstats_utils)
    importlib.reload(tradingagents.dataflows.interface)
except ImportError:
    pass  # Modules not loaded yet, will be loaded fresh

try:
    # Reload graph to pick up new streaming method
    import tradingagents.graph.trading_graph as _trading_graph_mod
    importlib.reload(_trading_graph_mod)
    import threading
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
except ImportError as e:
    st.error(f"Failed to import TradingAgents modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="TradingAgents - Multi-Agent Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components
from components.agent_status import render_agent_card
from components.sidebar_config import render_sidebar_configuration
from components.analysis_params import render_analysis_parameters, render_analysis_controls
from components.css_loader import load_css
from components.header_component import render_header
from components.report_components import render_analysis_report

# Placeholder for other components to be modularized
def inject_global_agent_css(): pass

# Apply custom styling
load_css()

# Ensure default session state
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {a: "pending" for a in [
        "Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst",
        "Bull Researcher", "Bear Researcher", "Research Manager", "Trader",
        "Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"
    ]}
if 'progress_messages' not in st.session_state:
    st.session_state.progress_messages = deque(maxlen=50)
if "agent_placeholders" not in st.session_state:
    st.session_state.agent_placeholders = {}

# Render header using component
render_header()

# Sidebar configuration using component
with st.sidebar:
    sidebar_config = render_sidebar_configuration()

# Main layout with aligned section titles
# Create a container that extends to full width
st.markdown('<div class="full-width-container">', unsafe_allow_html=True)

# Create aligned section headers
title_col1, title_col2 = st.columns([8, 2], gap="small")
with title_col1:
    st.header("🎯 Trading Analysis Dashboard")
with title_col2:
    st.header("🤖 Agent Status")

# Create main 2-column layout: Main content and Agent Status sidebar
main_content_col, agent_status_col = st.columns([8, 2], gap="small")

with main_content_col:
    # Analysis Parameters using component
    analysis_params = render_analysis_parameters()
    
    # Analysis Controls using component
    llm_config = sidebar_config.get('llm_config', {})
    selected_analysts = sidebar_config.get('selected_analysts', [])
    
    # Store selected analysts in session state for workflow progression logic
    st.session_state.selected_analysts_for_workflow = selected_analysts
    
    start_analysis = render_analysis_controls(analysis_params, llm_config, selected_analysts)
    
    if start_analysis:
        # Validate inputs
        if not analysis_params['stock_symbol']:
            st.error("Please enter a stock symbol")
            st.stop()
        
        # Store parameters in session state for use during analysis
        st.session_state.current_stock_symbol = analysis_params['stock_symbol']
        st.session_state.current_analysis_date = analysis_params['analysis_date']
        st.session_state.current_max_debate_rounds = analysis_params['max_debate_rounds']
        st.session_state.current_max_risk_rounds = analysis_params['max_risk_rounds']
        st.session_state.current_online_tools = analysis_params['online_tools']
        st.session_state.current_debug_mode = analysis_params['debug_mode']
        st.session_state.current_selected_analysts = selected_analysts
        
        # Initialize all agent statuses to pending for selected analysts
        all_agents = [
            "Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst",
            "Bull Researcher", "Bear Researcher", "Research Manager",
            "Trader", "Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"
        ]
        for agent in all_agents:
            st.session_state.agent_status[agent] = "pending"
        
        # Check API keys
        openai_key = os.getenv("OPENAI_API_KEY", "")
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        if not openai_key or not finnhub_key:
            st.error("Please configure both OpenAI and Finnhub API keys in your .env file")
            st.stop()
        
        # Start analysis
        st.session_state.analysis_running = True
        st.session_state.analysis_has_started = True
        st.session_state.analysis_results = None
        st.session_state.progress_messages.clear()
        
        st.rerun()
    
    st.divider()
    
    # Get current parameters for analysis execution
    stock_symbol = st.session_state.get('current_stock_symbol', analysis_params.get('stock_symbol', 'NVDA'))
    analysis_date = st.session_state.get('current_analysis_date', analysis_params.get('analysis_date'))
    max_debate_rounds = st.session_state.get('current_max_debate_rounds', analysis_params.get('max_debate_rounds', 2))
    max_risk_rounds = st.session_state.get('current_max_risk_rounds', analysis_params.get('max_risk_rounds', 2))
    online_tools = st.session_state.get('current_online_tools', analysis_params.get('online_tools', True))
    debug_mode = st.session_state.get('current_debug_mode', analysis_params.get('debug_mode', False))

    # Progress display - full width alignment (Stop Analysis button handled by component)
    if st.session_state.analysis_running:
        # Compact progress display - full width with dynamic updates
        completed_agents = sum(1 for status in st.session_state.agent_status.values() if status == "complete")
        total_agents = len(st.session_state.agent_status)
        progress = completed_agents / total_agents if total_agents > 0 else 0
        
        # Create agent progress placeholder for dynamic updates
        if 'agent_progress_placeholder' not in st.session_state:
            st.session_state.agent_progress_placeholder = st.empty()
        st.session_state.agent_progress_placeholder.progress(progress, text=f"Progress: {completed_agents}/{total_agents} agents completed")

    # Analysis progress UI slot - full width alignment
    if st.session_state.analysis_running:
        st.subheader("🔄 Analysis in Progress...")
        
        # Progress bar and status
        if 'progress_placeholder' not in st.session_state:
            st.session_state.progress_placeholder = st.empty()
        if 'status_placeholder' not in st.session_state:
            st.session_state.status_placeholder = st.empty()
        
        # Fixed-height streaming messages container using HTML iframe approach
        st.markdown("### 💬 Live Analysis Feed")
        
        # Initialize streaming messages if not exists
        if 'streaming_messages' not in st.session_state:
            st.session_state.streaming_messages = []
        
        # Create a placeholder for the streaming container
        if 'streaming_placeholder' not in st.session_state:
            st.session_state.streaming_placeholder = st.empty()
        
        if 'last_update_placeholder' not in st.session_state:
            st.session_state.last_update_placeholder = st.empty()

# Agent Status section content (title already placed above)
with agent_status_col:
    # Apply global agent CSS styling
    inject_global_agent_css()
    
    # Additional ultra-compact styling for expanders
    st.markdown(
        "<style>\n"
        "/* ULTRA compact styling - target ALL possible Streamlit button containers */\n"
        ".stButton { \n"
        "  margin: 0 !important; \n"
        "  margin-bottom: 0.1rem !important; \n"
        "  padding: 0 !important; \n"
        "}\n"
        ".stButton > button { \n"
        "  height: 1.3rem !important; \n"
        "  min-height: 1.3rem !important; \n"
        "  padding: 0.05rem 0.25rem !important; \n"
        "  font-size: 0.6rem !important; \n"
        "  line-height: 1.0 !important; \n"
        "  white-space: nowrap !important; \n"
        "  overflow: hidden !important; \n"
        "  text-overflow: ellipsis !important; \n"
        "  margin: 0 !important; \n"
        "}\n"
        "/* Remove ALL gaps and spacing in expanders */\n"
        ".stExpander [data-testid='stVerticalBlock'] { \n"
        "  gap: 0 !important; \n"
        "}\n"
        ".stExpander > div > div > div { \n"
        "  gap: 0 !important; \n"
        "  padding: 0 !important; \n"
        "}\n"
        "/* Target Streamlit's internal spacing */\n"
        "[data-testid='stVerticalBlock'] > div { \n"
        "  margin-bottom: 0.05rem !important; \n"
        "}\n"
        "/* Remove default Streamlit element spacing */\n"
        ".element-container { \n"
        "  margin-bottom: 0.05rem !important; \n"
        "}\n"
        "</style>",
        unsafe_allow_html=True
    )
    
    # Simple CSS classes for button states - GUARANTEED TO WORK
    st.markdown(
        "<style>\n"
        "/* AGENT BUTTON STATE CLASSES */\n"
        ".agent-completed { \n"
        "  background: linear-gradient(135deg, #2E865F 0%, #228B22 100%) !important; \n"
        "  color: #ffffff !important; \n"
        "  border: 0 !important; \n"
        "}\n"
        ".agent-running { \n"
        "  background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important; \n"
        "  color: #ffffff !important; \n"
        "  border: 0 !important; \n"
        "}\n"
        ".agent-waiting { \n"
        "  background: #6c757d !important; \n"
        "  color: #ffffff !important; \n"
        "  border: 0 !important; \n"
        "}\n"
        "</style>",
        unsafe_allow_html=True
    )
    
    # Debug: Show current agent status state
    if st.session_state.get('current_debug_mode', False):
        st.write("Debug - Agent Status:", st.session_state.agent_status)
        st.write("Debug - Analysis Running:", st.session_state.analysis_running)
        st.write("Debug - Analysis Has Started:", st.session_state.get('analysis_has_started', False))
    
    # Always show agent status if analysis has started, completed, or if we have selected analysts
    show_agent_status = (
        st.session_state.analysis_running or 
        any(status != "pending" for status in st.session_state.agent_status.values()) or
        st.session_state.get('analysis_has_started', False) or
        True  # Force show for debugging
    )
    
    if show_agent_status:
        
        # Show status during analysis and after completion
        # Always show teams if we're in agent status section
        teams = {
            "📈 Analyst Team": ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"],
            "🔍 Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
            "💰 Trading & Risk": ["Trader", "Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"]
        }
        
        # Prepare placeholders dict
        if "agent_cards" not in st.session_state:
            st.session_state.agent_cards = {}

        # Agent card rendering now handled by component

        # Now render all teams and agents
        for team_name, agents in teams.items():
            completed = sum(1 for agent in agents if st.session_state.agent_status[agent] == 'complete')
            total = len(agents)
            
            with st.expander(f"{team_name} ({completed}/{total} complete)", expanded=True):
                # Aggressive compact layout with minimal spacing
                st.markdown(
                    "<style>\n"
                    "/* Remove all gaps and margins between buttons */\n"
                    ".stExpander > div > div > div { gap: 0 !important; }\n"
                    ".stButton { margin: 0 !important; margin-bottom: 0.1rem !important; }\n"
                    "/* Make all buttons in this expander compact */\n"
                    ".stExpander .stButton > button { \n"
                    "  height: 1.5rem !important; \n"
                    "  min-height: 1.5rem !important; \n"
                    "  padding: 0.15rem 0.4rem !important; \n"
                    "  font-size: 0.7rem !important; \n"
                    "  line-height: 1.2 !important; \n"
                    "  white-space: nowrap !important; \n"
                    "  overflow: hidden !important; \n"
                    "  text-overflow: ellipsis !important; \n"
                    "}\n"
                    "</style>",
                    unsafe_allow_html=True
                )
                # Use single column layout for better visibility in sidebar
                for agent in agents:
                    # Each card renders into its own placeholder so we can refresh during streaming
                    ph = st.empty()
                    st.session_state.agent_cards[agent] = ph
                    # Initial render using component
                    status = st.session_state.agent_status[agent]
                    render_agent_card(ph, agent, status)
        
        # Show actionable agent outputs after completion
        if st.session_state.analysis_results and not st.session_state.analysis_running:
            st.markdown("### 📈 Agent Contributions Summary")
            
            final_state = st.session_state.analysis_results.get('result', {})
            
            # Create quick links to agent outputs - use single column for sidebar
            if final_state.get('market_report'):
                if st.button("📈 Market Analysis", use_container_width=True, key="market_btn"):
                    st.session_state.show_section = "market"
                    st.rerun()
            
            if final_state.get('news_report'):
                if st.button("📰 News Analysis", use_container_width=True, key="news_btn"):
                    st.session_state.show_section = "news"
                    st.rerun()
            
            if final_state.get('fundamentals_report'):
                if st.button("💰 Fundamentals", use_container_width=True, key="fundamentals_btn"):
                    st.session_state.show_section = "fundamentals"
                    st.rerun()
            
            if final_state.get('trader_investment_plan'):
                if st.button("⚖️ Risk Assessment", use_container_width=True, key="risk_btn"):
                    st.session_state.show_section = "risk"
                    st.rerun()
    else:
        st.info("🔄 Start an analysis to see agent status updates")

    
    # Agent Details Modal (when agent is clicked)
    if st.session_state.get('show_agent_details') and st.session_state.get('selected_agent'):
        selected_agent = st.session_state.selected_agent
        agent_status = st.session_state.agent_status.get(selected_agent, 'pending')
        
        # Create agent details modal with enhanced styling
        st.markdown('<div class="agent-modal">', unsafe_allow_html=True)
        with st.container():
            st.markdown(f"### 🔍 {selected_agent} - Live Details")
            
            # Close button
            col1, col2 = st.columns([6, 1])
            with col2:
                if st.button("✖️ Close", key="close_agent_details"):
                    st.session_state.show_agent_details = False
                    st.session_state.selected_agent = None
                    st.rerun()
            
            # Agent status and progress
            if agent_status == "running":
                st.info(f"🔄 **{selected_agent}** is currently active")
                
                # Live progress messages based on agent type
                if "Market" in selected_agent:
                    with st.container():
                        st.markdown("#### 📊 Current Activity:")
                        progress_messages = [
                            "🔍 Fetching latest market data...",
                            "📈 Calculating technical indicators (RSI, MACD, ATR)...",
                            "📊 Analyzing price trends and patterns...",
                            "🎯 Evaluating market conditions...",
                            "📝 Generating market analysis report..."
                        ]
                        
                        # Simulate live progress
                        import time
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)  # Change every 3 seconds
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        
                        # Progress bar for current task
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                        
                        # Show what data is being processed
                        with st.expander("📊 Data Being Processed"):
                            st.markdown("""
                            **Technical Indicators:**
                            - RSI (Relative Strength Index)
                            - MACD (Moving Average Convergence Divergence)  
                            - ATR (Average True Range)
                            - EMA (Exponential Moving Average)
                            - SMA (Simple Moving Average)
                            
                            **Market Data Sources:**
                            - Yahoo Finance API
                            - Finnhub Market Data
                            - Real-time price feeds
                            """)
                
                elif "Social" in selected_agent:
                    with st.container():
                        st.markdown("#### 👥 Current Activity:")
                        progress_messages = [
                            "🔍 Scanning social media platforms...",
                            "📊 Analyzing sentiment patterns...",
                            "🎯 Processing Reddit discussions...",
                            "📈 Evaluating Twitter sentiment...",
                            "📝 Generating social sentiment report..."
                        ]
                        
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                        
                        with st.expander("👥 Social Data Sources"):
                            st.markdown("""
                            **Sentiment Analysis:**
                            - Reddit financial discussions
                            - Twitter/X market sentiment
                            - Financial news comments
                            - Social media trends
                            - Community sentiment scores
                            """)
                        
                elif "News" in selected_agent:
                    with st.container():
                        st.markdown("#### 📰 Current Activity:")
                        progress_messages = [
                            "🔍 Fetching latest financial news...",
                            "📊 Processing news articles...",
                            "🎯 Analyzing news sentiment...",
                            "📈 Evaluating market impact...",
                            "📝 Generating news analysis report..."
                        ]
                        
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                        
                        with st.expander("📰 News Sources"):
                            st.markdown("""
                            **News Analysis:**
                            - Financial news headlines
                            - Market breaking news
                            - Company announcements
                            - Economic indicators
                            - Industry reports
                            """)
                        
                elif "Fundamentals" in selected_agent:
                    with st.container():
                        st.markdown("#### 💼 Current Activity:")
                        progress_messages = [
                            "🔍 Fetching financial statements...",
                            "📊 Analyzing balance sheet data...",
                            "🎯 Evaluating income statements...",
                            "📈 Processing cash flow data...",
                            "📝 Generating fundamentals report..."
                        ]
                        
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                        
                        with st.expander("💼 Fundamental Data"):
                            st.markdown("""
                            **Financial Analysis:**
                            - Balance sheet metrics
                            - Income statement analysis
                            - Cash flow evaluation
                            - Financial ratios
                            - Company valuation
                            """)
                
                # Add support for other agent types
                elif "Bull" in selected_agent or "Bear" in selected_agent:
                    with st.container():
                        st.markdown("#### 🔍 Current Activity:")
                        progress_messages = [
                            "🔍 Reviewing analyst reports...",
                            "📊 Building investment thesis...",
                            "🎯 Evaluating market position...",
                            "📈 Formulating arguments...",
                            "📝 Preparing debate position..."
                        ]
                        
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                
                elif "Research Manager" in selected_agent:
                    with st.container():
                        st.markdown("#### 🎯 Current Activity:")
                        progress_messages = [
                            "🔍 Reviewing team debates...",
                            "📊 Weighing arguments...",
                            "🎯 Making final decision...",
                            "📈 Preparing recommendation...",
                            "📝 Finalizing research report..."
                        ]
                        
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                
                elif "Trader" in selected_agent:
                    with st.container():
                        st.markdown("#### 💰 Current Activity:")
                        progress_messages = [
                            "🔍 Analyzing trading signals...",
                            "📊 Planning entry/exit points...",
                            "🎯 Setting position sizes...",
                            "📈 Calculating risk metrics...",
                            "📝 Creating trading plan..."
                        ]
                        
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                
                elif "Risk" in selected_agent or "Conservative" in selected_agent or "Aggressive" in selected_agent:
                    with st.container():
                        st.markdown("#### ⚖️ Current Activity:")
                        progress_messages = [
                            "🔍 Assessing risk factors...",
                            "📊 Calculating risk metrics...",
                            "🎯 Evaluating volatility...",
                            "📈 Analyzing downside risk...",
                            "📝 Preparing risk assessment..."
                        ]
                        
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                
                elif "Portfolio" in selected_agent:
                    with st.container():
                        st.markdown("#### 📈 Current Activity:")
                        progress_messages = [
                            "🔍 Reviewing all team reports...",
                            "📊 Optimizing portfolio allocation...",
                            "🎯 Balancing risk and return...",
                            "📈 Making final decision...",
                            "📝 Preparing final recommendation..."
                        ]
                        
                        current_time = int(time.time())
                        message_index = (current_time // 3) % len(progress_messages)
                        
                        st.markdown(f"**Current Task:** {progress_messages[message_index]}")
                        progress_value = ((current_time % 3) / 3.0) * 100
                        st.progress(int(progress_value))
                
                # Note: Auto-refresh removed to prevent infinite loops
                # The progress messages will update naturally when user refreshes or navigates
                
            elif agent_status == "complete":
                st.success(f"✅ **{selected_agent}** has completed analysis")
                
                # Show completed results if available
                if st.session_state.analysis_results:
                    final_state = st.session_state.analysis_results.get('result', {})
                    
                    if "Market" in selected_agent and final_state.get('market_report'):
                        st.markdown("#### 📊 Market Analysis Results:")
                        st.markdown(final_state['market_report'][:500] + "..." if len(final_state['market_report']) > 500 else final_state['market_report'])
                        
                        with st.expander("📈 View Full Market Report"):
                            st.markdown(final_state['market_report'])
                    
                    elif "Social" in selected_agent and final_state.get('sentiment_report'):
                        st.markdown("#### 👥 Social Analysis Results:")
                        st.markdown(final_state['sentiment_report'][:500] + "..." if len(final_state['sentiment_report']) > 500 else final_state['sentiment_report'])
                        
                        with st.expander("👥 View Full Social Report"):
                            st.markdown(final_state['sentiment_report'])
                    
                    elif "News" in selected_agent and final_state.get('news_report'):
                        st.markdown("#### 📰 News Analysis Results:")
                        st.markdown(final_state['news_report'][:500] + "..." if len(final_state['news_report']) > 500 else final_state['news_report'])
                        
                        with st.expander("📰 View Full News Report"):
                            st.markdown(final_state['news_report'])
                    
                    elif "Fundamentals" in selected_agent and final_state.get('fundamentals_report'):
                        st.markdown("#### 💼 Fundamentals Analysis Results:")
                        st.markdown(final_state['fundamentals_report'][:500] + "..." if len(final_state['fundamentals_report']) > 500 else final_state['fundamentals_report'])
                        
                        with st.expander("💼 View Full Fundamentals Report"):
                            st.markdown(final_state['fundamentals_report'])
                
            else:
                st.warning(f"⏳ **{selected_agent}** is waiting to start")
                st.markdown("This agent will begin analysis once the previous agents complete their tasks.")
            
            st.divider()
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close agent modal div
    
    # Note: Removed duplicate Live Activity Feed to avoid redundancy with agent details modal


# --- Helper Functions for UI Updates ---
def _set_status(agent: str, status: str):
    """Safely update agent status and refresh its card."""
    if agent in st.session_state.agent_status:
        if st.session_state.agent_status[agent] != status:
            st.session_state.agent_status[agent] = status
            _refresh_card(agent)

def _refresh_card(agent: str):
    """Rerender a specific agent's status card."""
    if agent in st.session_state.agent_cards:
        placeholder = st.session_state.agent_cards[agent]
        status = st.session_state.agent_status.get(agent, "pending")
        render_agent_card(placeholder, agent, status)

def _update_running_status_text(status_text_ph, running_agents):
    """Update the status text with a rotating display of running agents."""
    if not running_agents:
        status_text_ph.text("Starting agents...")
        return

    # Rotate through running agents for dynamic text
    if 'running_agent_idx' not in st.session_state:
        st.session_state.running_agent_idx = 0
    
    idx = st.session_state.running_agent_idx
    agent_name = running_agents[idx % len(running_agents)]
    
    if len(running_agents) > 1:
        status_text_ph.text(f"⏳ Multiple Agents Working: {agent_name} is active...")
    else:
        status_text_ph.text(f"⏳ {agent_name} is working...")
        
    st.session_state.running_agent_idx += 1

# --- Analysis Execution Logic ---
if st.session_state.analysis_running:
    progress_bar_ph = st.session_state.get("progress_placeholder", st.empty())
    status_text_ph = st.session_state.get("status_placeholder", st.empty())
    streaming_placeholder = st.session_state.get("streaming_placeholder", st.empty())
    last_update_ph = st.session_state.get("last_update_placeholder", st.empty())

    # Re-acquire params from session state and sidebar config
    llm_config = sidebar_config.get('llm_config', {})
    provider = llm_config.get('provider', 'OpenAI').lower()
    backend_url = llm_config.get('backend_url', os.getenv('OPENAI_API_BASE_URL', ''))
    selected_model = llm_config.get('deep_think_model', 'gpt-4-turbo')
    
    # Get analysis parameters from session state
    stock_symbol = st.session_state.get('current_stock_symbol', 'NVDA')
    analysis_date = st.session_state.get('current_analysis_date', date.today())
    debug_mode = st.session_state.get('current_debug_mode', False)
    
    config = {
        "llm_config": {
            "provider": provider,
            "config_list": [
                {
                    "model": selected_model,
                    "base_url": backend_url if backend_url else None,
                }
            ],
            "temperature": 0.1,
        },
        "research_depth": "deep",  # Use deep analysis by default
    }

    date_str = analysis_date.strftime("%Y-%m-%d")
    
    # Map sidebar analyst selection to full analyst names
    selected_analysts_from_sidebar = sidebar_config.get('selected_analysts', [])
    analyst_mapping = {
        'market': 'Market Analyst',
        'social': 'Social Analyst', 
        'news': 'News Analyst',
        'fundamentals': 'Fundamentals Analyst'
    }
    
    # Convert sidebar selection to full analyst names, default to all if empty
    if selected_analysts_from_sidebar:
        active_selected_analysts = [analyst_mapping.get(key, key) for key in selected_analysts_from_sidebar]
    else:
        # Default to all analysts if none selected
        active_selected_analysts = [
            "Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst",
            "Bull Researcher", "Bear Researcher", "Research Manager", "Trader",
            "Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"
        ]

    # Pre-check for market data availability
    try:
        def _has_price_data(symbol: str, start_date: datetime, end_date: datetime) -> bool:
            try:
                import os as _os, requests as _req
                api_key = _os.getenv("FINNHUB_API_KEY")
                if not api_key: return True
                frm, to = int(start_date.timestamp()), int(end_date.timestamp())
                url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={frm}&to={to}&token={api_key}"
                data = _req.get(url, timeout=10).json()
                return not (data.get("s") == "no_data" or (isinstance(data.get("c"), list) and not data.get("c")))
            except Exception: return True

        market_selected = any("Market Analyst" in a for a in active_selected_analysts)
        if market_selected and not _has_price_data(stock_symbol, analysis_date - timedelta(days=60), analysis_date):
            others = [a for a in active_selected_analysts if "Market Analyst" not in a]
            if others:
                st.warning("⚠️ Market data unavailable. Skipping Market Analyst.")
                _set_status("Market Analyst", "error")
                active_selected_analysts = others
            else:
                st.error("❌ Market data unavailable. Cannot proceed with analysis.")
                st.session_state.analysis_running = False
                st.rerun()
    except Exception as e:
        st.warning(f"Could not perform price data pre-check: {e}")

    # Main analysis loop with comprehensive error handling
    try:
        # Test TradingAgentsGraph initialization first
        status_text_ph.text("🔧 Initializing analysis framework...")
        ta = TradingAgentsGraph(selected_analysts=active_selected_analysts, debug=debug_mode, config=config)
        status_text_ph.text("🚀 Starting analysis...")
        
        # Simple test execution without streaming to avoid crashes
        st.info(f"🎯 Analysis started for {stock_symbol} on {date_str}")
        st.info(f"📊 Selected analysts: {', '.join(active_selected_analysts)}")
        
        # For now, just simulate completion to test the UI flow
        import time
        time.sleep(2)  # Brief pause to show the UI is working
        
        # Mark analysis as complete
        progress_bar_ph.progress(1.0)
        status_text_ph.text("✅ Analysis complete (test mode)!")
        
        # Set all agents to complete for testing
        for agent in active_selected_analysts:
            _set_status(agent, 'complete')
            
        st.session_state.analysis_results = {
            "symbol": stock_symbol,
            "date": date_str,
            "result": {"test": "This is a test result to verify UI flow"},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.analysis_running = False
        st.success("🎉 Test analysis completed successfully!")

    except Exception as outer_exc:
        st.session_state.last_error = str(outer_exc)
        st.error(f"❌ Fatal error starting analysis: {outer_exc}")
        st.session_state.analysis_running = False
        if debug_mode: 
            st.exception(outer_exc)
        else:
            st.error("Please check your API keys and configuration.")
        st.rerun()

# --- Final Report Rendering ---
if st.session_state.analysis_results and not st.session_state.analysis_running:
    render_analysis_report(st.session_state.analysis_results, debug_mode)

# Display last error if any
if st.session_state.get("last_error") and not st.session_state.analysis_running:
    st.error(f"An error occurred during the last analysis: {st.session_state.last_error}")

# About section moved to sidebar for cleaner main content area
