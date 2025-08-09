"""
TradingAgents Streamlit Interface
A modern web UI that replicates and enhances the CLI experience
"""

import streamlit as st
import os
import sys
import importlib
from datetime import datetime, date, timedelta
import json
import time
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

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .agent-status-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #333333;
    }
    .agent-status-pending {
        border-left-color: #ffc107;
        background: #fff3cd;
        color: #856404;
    }
    .agent-status-running {
        border-left-color: #17a2b8;
        background: #d1ecf1;
        color: #0c5460;
    }
    .agent-status-complete {
        border-left-color: #28a745;
        background: #d4edda;
        color: #155724;
    }
    .result-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .analysis-progress {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Animated agent cards */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(0, 123, 255, 0.5); }
        50% { box-shadow: 0 0 20px rgba(0, 123, 255, 0.8); }
        100% { box-shadow: 0 0 5px rgba(0, 123, 255, 0.5); }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Running agent animation - more prominent */
    .stButton > button[kind="primary"] {
        animation: pulse 1s infinite, glow 2s infinite !important;
        border: 3px solid #007bff !important;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.6) !important;
        transform: scale(1.02) !important;
    }
    
    /* Alternative selector for running buttons */
    button[data-testid="baseButton-primary"] {
        animation: pulse 1s infinite, glow 2s infinite !important;
        border: 3px solid #007bff !important;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.6) !important;
        transform: scale(1.02) !important;
    }
    
    /* Enhanced pulse animation */
    @keyframes pulse {
        0% { 
            transform: scale(1.02); 
            box-shadow: 0 0 15px rgba(0, 123, 255, 0.6);
        }
        50% { 
            transform: scale(1.08); 
            box-shadow: 0 0 25px rgba(0, 123, 255, 0.9);
        }
        100% { 
            transform: scale(1.02); 
            box-shadow: 0 0 15px rgba(0, 123, 255, 0.6);
        }
    }
    
    /* Enhanced glow animation */
    @keyframes glow {
        0% { 
            box-shadow: 0 0 15px rgba(0, 123, 255, 0.6);
            border-color: #007bff;
        }
        50% { 
            box-shadow: 0 0 30px rgba(0, 123, 255, 1.0);
            border-color: #0056b3;
        }
        100% { 
            box-shadow: 0 0 15px rgba(0, 123, 255, 0.6);
            border-color: #007bff;
        }
    }
    
    /* Live activity indicators */
    .activity-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .activity-running {
        background: #007bff;
        animation: pulse 1.5s infinite;
    }
    
    .activity-complete {
        background: #28a745;
    }
    
    .activity-pending {
        background: #ffc107;
    }
    
    /* Progress animation */
    .progress-animated {
        animation: pulse 2s infinite;
    }
    
    /* Agent modal styling */
    .agent-modal {
        background: #ffffff;
        border: 2px solid #007bff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'agent_status' not in st.session_state:
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
if 'progress_messages' not in st.session_state:
    st.session_state.progress_messages = deque(maxlen=50)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 TradingAgents</h1>
    <h3>Multi-Agent LLM Financial Trading Framework</h3>
    <p>Powered by specialized AI agents for comprehensive market analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Trading Configuration")
    
    # API Key Status Check
    st.subheader("🔑 API Status")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    
    if openai_key:
        st.success("✅ OpenAI API Key: Configured")
    else:
        st.error("❌ OpenAI API Key: Missing")
        st.info("Please add OPENAI_API_KEY to your .env file")
    
    if finnhub_key:
        st.success("✅ Finnhub API Key: Configured")
    else:
        st.error("❌ Finnhub API Key: Missing")
        st.info("Please add FINNHUB_API_KEY to your .env file")
    
    st.divider()
    
    # Provider & Models
    st.subheader("🧠 LLM Provider & Models")

    provider = st.selectbox(
        "LLM Provider",
        options=["OpenAI", "Anthropic", "Google", "OpenRouter", "Ollama"],
        index=0,
        help="Choose the LLM provider. Model menus update accordingly."
    )

    # Backend/base URL or host depending on provider
    backend_url_help = {
        "OpenAI": "Leave blank for default (https://api.openai.com/v1) or set a custom compatible proxy.",
        "OpenRouter": "Required: OpenRouter base URL (e.g., https://openrouter.ai/api).",
        "Ollama": "Ollama host (e.g., http://localhost:11434).",
        "Anthropic": "No base URL needed in most cases.",
        "Google": "No base URL needed in most cases."
    }
    default_backend = {
        "OpenAI": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "OpenRouter": os.getenv("OPENROUTER_BASE_URL", ""),
        "Ollama": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "Anthropic": "",
        "Google": "",
    }[provider]

    backend_url = st.text_input(
        "Backend URL / Host",
        value=default_backend,
        help=backend_url_help.get(provider, "")
    )

    # Allow filtering Deep models to 'thinking' models for OpenAI
    deep_thinking_only = st.checkbox(
        "Deep model: show only OpenAI 'thinking' models (o1 family)",
        value=False,
        help="When enabled with OpenAI provider, deep model list is limited to o1 family."
    )

    # Dynamic model options per provider (sane defaults; can be expanded later)
    quick_model_options = {
        "OpenAI": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        "Anthropic": ["claude-3-haiku", "claude-3-sonnet"],
        "Google": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "OpenRouter": ["openrouter/auto", "meta-llama/llama-3-8b-instruct"],
        "Ollama": ["llama3.2:1b", "llama3:8b"]
    }[provider]

    deep_model_options = {
        "OpenAI": (
            ["o1-mini", "o1"]
            if deep_thinking_only
            else [
                "o1-mini",
                "o1",
                "gpt-4o",
                "gpt-4-turbo",
            ]
        ),
        "Anthropic": ["claude-3-opus", "claude-3.5-sonnet"],
        "Google": ["gemini-1.5-pro", "gemini-1.5-pro-exp"],
        "OpenRouter": ["anthropic/claude-3.5-sonnet", "meta-llama/llama-3-70b-instruct"],
        "Ollama": ["llama3:8b", "llama3:70b"]
    }[provider]

    quick_think_model = st.selectbox(
        "Quick Thinking Model",
        options=quick_model_options,
        index=0,
        help="Faster, cheaper model used for quick reasoning steps"
    )

    deep_think_model = st.selectbox(
        "Deep Thinking Model",
        options=deep_model_options,
        index=min(1, len(deep_model_options)-1),
        help="More capable model used for complex analysis"
    )

    st.divider()

    # Trading Parameters
    st.subheader("📊 Analysis Parameters")
    
    # Stock Symbol Selection
    stock_symbol = st.text_input(
        "Stock Symbol",
        value="NVDA",
        help="Enter the stock ticker symbol (e.g., AAPL, GOOGL, TSLA, SPY)"
    ).upper()
    
    # Date Selection
    max_date = date.today() - timedelta(days=1)  # Yesterday as max date
    min_date = date.today() - timedelta(days=365)  # One year ago as min date
    
    analysis_date = st.date_input(
        "Analysis Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Select the date for analysis (market data required)"
    )
    
    st.divider()
    
    # Analysis Depth
    st.subheader("🔍 Analysis Depth")

    depth_choice = st.radio(
        "Research Depth",
        options=["Beginner", "Standard", "Deep", "Custom"],
        index=1,
        help="Use presets or choose Custom to set rounds manually"
    )

    preset_rounds = {"Beginner": (1, 1), "Standard": (2, 2), "Deep": (3, 3)}
    preset_debate, preset_risk = preset_rounds.get(depth_choice, (2, 1))

    max_debate_rounds = st.slider(
        "Debate Rounds",
        min_value=1,
        max_value=5,
        value=preset_debate,
        disabled=(depth_choice != "Custom"),
        help="Number of debate rounds between bull/bear researchers"
    )

    max_risk_rounds = st.slider(
        "Risk Discussion Rounds",
        min_value=1,
        max_value=3,
        value=preset_risk,
        disabled=(depth_choice != "Custom"),
        help="Number of risk management discussion rounds"
    )
    
    online_tools = st.checkbox(
        "Enable Online Tools",
        value=True,
        help="Allow agents to access real-time financial data and news"
    )
    
    debug_mode = st.checkbox(
        "Debug Mode",
        value=False,
        help="Show detailed agent communications and tool calls"
    )

    st.divider()

    # Analyst selection (optional)
    st.subheader("🧩 Analyst Selection")
    analyst_labels = {
        "market": "📊 Market Analyst",
        "social": "👥 Social Analyst",
        "news": "📰 News Analyst",
        "fundamentals": "💼 Fundamentals Analyst",
    }
    analyst_keys = list(analyst_labels.keys())
    default_selected = analyst_keys  # default all
    selected_labels = st.multiselect(
        "Select analysts to include (optional)",
        options=[analyst_labels[k] for k in analyst_keys],
        default=[analyst_labels[k] for k in default_selected],
        help="Matches CLI optional analyst subset. If unsupported by backend, this is ignored."
    )
    # Map back to ids
    selected_analysts = [k for k, v in analyst_labels.items() if v in selected_labels]

# Main content area - Optimized single column layout
st.header("🎯 Trading Analysis Dashboard")

# Analysis Controls with compact layout
col_start, col_stop, col_status = st.columns([1, 1, 2])

with col_start:
    if not st.session_state.analysis_running:
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            # Validate inputs
            if not stock_symbol:
                st.error("Please enter a stock symbol")
                st.stop()
            
            # Set Market Analyst to running BEFORE starting analysis to show animation
            st.session_state.agent_status["Market Analyst"] = "running"
            st.session_state.analysis_running = True
            
            # Trigger rerun to show animation BEFORE analysis starts
            st.rerun()
            
            if not openai_key or not finnhub_key:
                st.error("Please configure both OpenAI and Finnhub API keys in your .env file")
                st.stop()
            
            # Start analysis
            st.session_state.analysis_running = True
            st.session_state.analysis_results = None
            st.session_state.progress_messages.clear()
            
            # Reset agent status
            for agent in st.session_state.agent_status:
                st.session_state.agent_status[agent] = "pending"
            
            st.rerun()

with col_stop:
    if st.session_state.analysis_running:
        if st.button("⏹️ Stop Analysis", type="secondary", use_container_width=True):
            st.session_state.analysis_running = False
            st.rerun()

with col_status:
    if st.session_state.analysis_running:
        # Compact progress display
        completed_agents = sum(1 for status in st.session_state.agent_status.values() if status == "complete")
        total_agents = len(st.session_state.agent_status)
        progress = completed_agents / total_agents if total_agents > 0 else 0
        st.progress(progress, text=f"Progress: {completed_agents}/{total_agents} agents completed")

# Agent Status with Links to Outputs (only show if analysis has started or completed)
if st.session_state.analysis_running or any(status != "pending" for status in st.session_state.agent_status.values()):
    
    # Only show during analysis - remove after completion to save space
    if st.session_state.analysis_running:
        st.markdown("### 🤖 Agent Status Overview")
        teams = {
            "📈 Analyst Team": ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"],
            "🔍 Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
            "💰 Trading & Risk": ["Trader", "Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"]
        }
        
        # Prepare placeholders dict
        if "agent_cards" not in st.session_state:
            st.session_state.agent_cards = {}

        for team_name, agents in teams.items():
            completed = sum(1 for agent in agents if st.session_state.agent_status[agent] == 'complete')
            total = len(agents)
            
            with st.expander(f"{team_name} ({completed}/{total} complete)", expanded=True):
                cols = st.columns(len(agents))
                for i, agent in enumerate(agents):
                    with cols[i]:
                        # Each card renders into its own placeholder so we can refresh during streaming
                        ph = st.empty()
                        st.session_state.agent_cards[agent] = ph
                        # Initial render
                        status = st.session_state.agent_status[agent]
                        _render_agent_card = _render_agent_card if '_render_agent_card' in globals() else None
                        if _render_agent_card is None:
                            # define lightweight renderer
                            def _render_agent_card(placeholder, agent_name, status_val):
                                agent_labels = {
                                    "Market Analyst": "📊 Market",
                                    "Social Analyst": "👥 Social",
                                    "News Analyst": "📰 News",
                                    "Fundamentals Analyst": "💼 Fundamentals",
                                    "Bull Researcher": "🐂 Bull",
                                    "Bear Researcher": "🐻 Bear",
                                    "Research Manager": "🎯 Manager",
                                    "Trader": "💰 Trader",
                                    "Risky Analyst": "⚡ Risky",
                                    "Neutral Analyst": "⚖️ Neutral",
                                    "Safe Analyst": "🛡️ Safe",
                                    "Portfolio Manager": "📈 Portfolio",
                                }
                                agent_display = agent_labels.get(agent_name, agent_name.split()[-1])
                                # ensure epoch exists
                                if "render_epoch" not in st.session_state:
                                    st.session_state.render_epoch = 0
                                key_suffix = f"_{st.session_state.render_epoch}"
                                with placeholder.container():
                                    if status_val == "complete":
                                        st.success(f"✅ {agent_display} - Click for details")
                                        if st.button(f"View {agent_display} Results", key=f"agent_{agent_name}_complete{key_suffix}", use_container_width=True):
                                            st.session_state.selected_agent = agent_name
                                            st.session_state.show_agent_details = True
                                    elif status_val == "error":
                                        st.error(f"❌ {agent_display} - Error")
                                        if st.button(f"View {agent_display} Error", key=f"agent_{agent_name}_error{key_suffix}", use_container_width=True):
                                            st.session_state.selected_agent = agent_name
                                            st.session_state.show_agent_details = True
                                    elif status_val == "running":
                                        st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                                            color: white;
                                            padding: 10px;
                                            border-radius: 8px;
                                            text-align: center;
                                            border: 3px solid #007bff;
                                            box-shadow: 0 0 20px rgba(0, 123, 255, 0.8);
                                            animation: pulse 1.5s infinite;
                                            margin: 5px 0;
                                        ">
                                            🔄 <strong>{agent_display}</strong> - WORKING
                                        </div>
                                        """, unsafe_allow_html=True)
                                        if st.button(f"View {agent_display} Live Progress", key=f"agent_{agent_name}_running{key_suffix}", use_container_width=True, type="primary"):
                                            st.session_state.selected_agent = agent_name
                                            st.session_state.show_agent_details = True
                                    else:
                                        st.warning(f"⏳ {agent_display} - Waiting")
                                        if st.button(f"View {agent_display} Status", key=f"agent_{agent_name}_pending{key_suffix}", use_container_width=True):
                                            st.session_state.selected_agent = agent_name
                                            st.session_state.show_agent_details = True
                        _render_agent_card(ph, agent, status)
    
    # Show actionable agent outputs after completion
    elif st.session_state.analysis_results and not st.session_state.analysis_running:
        st.markdown("### 📈 Agent Contributions Summary")
        
        final_state = st.session_state.analysis_results.get('result', {})
        
        # Create quick links to agent outputs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if final_state.get('market_report'):
                if st.button("📈 Market Analysis", use_container_width=True):
                    st.session_state.show_section = "market"
                    st.rerun()
        
        with col2:
            if final_state.get('news_report'):
                if st.button("📰 News Analysis", use_container_width=True):
                    st.session_state.show_section = "news"
                    st.rerun()
        
        with col3:
            if final_state.get('fundamentals_report'):
                if st.button("💰 Fundamentals", use_container_width=True):
                    st.session_state.show_section = "fundamentals"
                    st.rerun()
        
        with col4:
            if final_state.get('trader_investment_plan'):
                if st.button("⚖️ Risk Assessment", use_container_width=True):
                    st.session_state.show_section = "risk"
                    st.rerun()
        
        # Show selected section content
        if hasattr(st.session_state, 'show_section'):
            st.markdown("---")
            if st.session_state.show_section == "market" and final_state.get('market_report'):
                st.markdown("**Market Analyst Report:**")
                st.markdown(final_state['market_report'])
            elif st.session_state.show_section == "news" and final_state.get('news_report'):
                st.markdown("**News Analyst Report:**")
                st.markdown(final_state['news_report'])
            elif st.session_state.show_section == "fundamentals" and final_state.get('fundamentals_report'):
                st.markdown("**Fundamentals Analyst Report:**")
                st.markdown(final_state['fundamentals_report'])
            elif st.session_state.show_section == "risk" and final_state.get('trader_investment_plan'):
                st.markdown("**Trading & Risk Assessment:**")
                st.markdown(final_state['trader_investment_plan'])
    
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

# Analysis Progress and Results
if st.session_state.analysis_running:
        st.markdown('<div class="analysis-progress">', unsafe_allow_html=True)
        st.subheader("🔄 Analysis in Progress...")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create configuration (Phase 1 parity: provider/models/backend/rounds)
        config = DEFAULT_CONFIG.copy()
        config.update({
            "llm_provider": provider,
            "backend_url": backend_url,
            "deep_think_llm": deep_think_model,
            "quick_think_llm": quick_think_model,
            "max_debate_rounds": max_debate_rounds,
            "max_risk_discuss_rounds": max_risk_rounds,
            "online_tools": online_tools,
        })
        
        date_str = analysis_date.strftime("%Y-%m-%d")
        
        try:
            # Optional: quick price-data availability pre-check for Market Analyst
            def _has_price_data(symbol: str, start_date: datetime, end_date: datetime) -> bool:
                try:
                    import os as _os
                    import time as _time
                    import requests as _req
                    api_key = _os.getenv("FINNHUB_API_KEY")
                    if not api_key:
                        # Skip strict check if no key; let downstream try
                        return True
                    frm = int(start_date.timestamp())
                    to = int(end_date.timestamp())
                    url = (
                        "https://finnhub.io/api/v1/stock/candle"
                        f"?symbol={symbol}&resolution=D&from={frm}&to={to}&token={api_key}"
                    )
                    resp = _req.get(url, timeout=10)
                    if resp.status_code != 200:
                        return True  # don't block on API hiccups
                    data = resp.json()
                    if not isinstance(data, dict):
                        return True
                    s = data.get("s")
                    c = data.get("c")
                    if s == "no_data" or (isinstance(c, list) and len(c) == 0):
                        return False
                    return True
                except Exception:
                    return True

            # Build the active selections, possibly skipping Market if no data
            active_selected_analysts = selected_analysts
            try:
                # Check a modest recent window for availability
                check_end = analysis_date
                check_start = analysis_date - timedelta(days=60)
                market_selected = any(a.lower() == "market" or "market analyst" in a.lower() for a in selected_analysts)
                if market_selected and not _has_price_data(stock_symbol, check_start, check_end):
                    others = [a for a in selected_analysts if not (a.lower() == "market" or "market analyst" in a.lower())]
                    if others:
                        active_selected_analysts = others
                        st.warning("⚠️ Market data unavailable for this symbol/date. Skipping Market Analyst and proceeding with remaining analysts.")
                        # reflect on UI
                        st.session_state.agent_status["Market Analyst"] = "error"
                        _refresh_card("Market Analyst")
                    else:
                        st.error("❌ Required market price data unavailable for this symbol/date. Please choose another symbol/date or add price data.")
                        st.session_state.analysis_running = False
                        st.stop()
            except Exception:
                # Non-fatal: proceed with original selections
                active_selected_analysts = selected_analysts
            # Initialize TradingAgents
            status_text.text("🤖 Initializing TradingAgents framework...")
            progress_bar.progress(10)

            ta = TradingAgentsGraph(
                selected_analysts=active_selected_analysts,
                debug=True,  # enable streaming-friendly behavior
                config=config,
            )

            # Live containers
            live_feed = st.container()
            last_update = st.empty()

            import time

            # Kick off first agent animation
            st.session_state.agent_status["Market Analyst"] = "running"
            status_text.text("📊 Market Analyst analyzing technical indicators...")
            progress_bar.progress(20)
            time.sleep(0.5)

            # Helper to extract safe text from a message object/dict
            def _extract_msg_text(msg):
                try:
                    if isinstance(msg, dict):
                        return msg.get("content") or msg.get("text") or str(msg)
                    # object with content attribute
                    return getattr(msg, "content", str(msg))
                except Exception:
                    return str(msg)

            # Heuristic: infer which agent is active based on message text
            def _infer_agent_and_update_status(text: str):
                if not text:
                    return
                t = text.lower()
                mapping = [
                    ("Market Analyst", ["technical indicator", "market analyst", "rsi", "macd", "bollinger", "moving average", "market analysis"]),
                    ("Social Analyst", ["social", "sentiment", "reddit", "twitter", "x.com", "social media"]),
                    ("News Analyst", ["news", "headline", "article", "press release", "news analysis"]),
                    ("Fundamentals Analyst", ["fundamentals", "balance sheet", "cashflow", "income statement", "p/e", "valuation", "fundamental analysis"]),
                    ("Bull Researcher", ["bull researcher", "bull case", "bullish"]),
                    ("Bear Researcher", ["bear researcher", "bear case", "bearish"]),
                    ("Research Manager", ["research manager", "judge", "investment debate", "judge decision"]),
                    ("Trader", ["trader", "investment plan", "portfolio allocation"]),
                    ("Risky Analyst", ["risky analyst", "risk-high", "high risk"]),
                    ("Neutral Analyst", ["neutral analyst", "moderate risk", "balanced risk"]),
                    ("Safe Analyst", ["safe analyst", "low risk", "conservative"]),
                    ("Portfolio Manager", ["portfolio manager", "final decision", "final trade decision", "executive summary"]),
                ]
                # Heuristic status inference disabled in favor of deterministic event mapping.

            # Early API key validation for online tools
            try:
                import os as _os
                missing_msgs = []
                if online_tools:
                    if provider.lower() in ["openai", "openrouter"] and not _os.getenv("OPENAI_API_KEY"):
                        missing_msgs.append("OpenAI API key")
                    if not _os.getenv("FINNHUB_API_KEY"):
                        missing_msgs.append("Finnhub API key")
                if missing_msgs:
                    st.error("❌ API Key Error: Missing " + ", ".join(missing_msgs) + ". Please add them in your environment or sidebar and try again.")
                    st.session_state.analysis_running = False
                    st.stop()
            except Exception:
                pass

            # Ensure structures for incremental streaming
            if "report_sections" not in st.session_state:
                st.session_state.report_sections = {
                    "market_report": None,
                    "sentiment_report": None,
                    "news_report": None,
                    "fundamentals_report": None,
                    "investment_plan": None,
                    "trader_investment_plan": None,
                    "final_trade_decision": None,
                }
            if "tool_calls" not in st.session_state:
                st.session_state.tool_calls = []

            # Error diagnostics helper: extract status/tool and classify
            def _diagnose_error(stream_error: Exception):
                err_msg = str(stream_error) if stream_error else ""
                err_lower = err_msg.lower()
                last_tool = None
                for tc in reversed(st.session_state.get("tool_calls", [])):
                    if tc and (tc.get("error") or tc.get("status")):
                        last_tool = tc
                        break
                status = str(last_tool.get("status", "")) if last_tool else ""
                tool_name = last_tool.get("name") or last_tool.get("tool") or last_tool.get("endpoint") if last_tool else None
                provider_hint = "OpenAI" if "openai" in err_lower or (tool_name and "openai" in tool_name.lower()) else ("Finnhub" if ("finnhub" in err_lower or (tool_name and "finnhub" in tool_name.lower())) else None)

                # Classification
                if "429" in err_msg or "rate limit" in err_lower or "too many requests" in err_lower or status == "429":
                    reason = "Rate limit exceeded"
                    guidance = "Wait 60–90s, reduce Research Depth, or upgrade your plan."
                    return {
                        "kind": "rate_limit",
                        "provider": provider_hint,
                        "status": status or "429",
                        "tool": tool_name,
                        "message": f"❌ Rate limit exceeded{f' on {provider_hint}' if provider_hint else ''}. {guidance}",
                    }
                if "401" in err_msg or "unauthorized" in err_lower or status == "401":
                    return {
                        "kind": "auth",
                        "provider": provider_hint,
                        "status": status or "401",
                        "tool": tool_name,
                        "message": f"❌ Authentication failed{f' for {provider_hint}' if provider_hint else ''}. Check API key and base URL.",
                    }
                if "403" in err_msg or "forbidden" in err_lower or "insufficient_quota" in err_lower or "access" in err_lower and "denied" in err_lower or status == "403":
                    return {
                        "kind": "permissions",
                        "provider": provider_hint,
                        "status": status or "403",
                        "tool": tool_name,
                        "message": f"❌ Permission error{f' on {provider_hint}' if provider_hint else ''}. Model/endpoint access not allowed.",
                    }
                if "timeout" in err_lower or "timed out" in err_lower or "connection" in err_lower:
                    return {
                        "kind": "network",
                        "provider": provider_hint,
                        "status": status or "",
                        "tool": tool_name,
                        "message": "❌ Network Error: Connection timeout. Please retry.",
                    }
                if "api" in err_lower and "key" in err_lower:
                    return {
                        "kind": "api_key",
                        "provider": provider_hint,
                        "status": status or "",
                        "tool": tool_name,
                        "message": "❌ API Key Error: Check your API keys and try again.",
                    }
                return {
                    "kind": "unknown",
                    "provider": provider_hint,
                    "status": status or "",
                    "tool": tool_name,
                    "message": f"❌ Analysis Error: {err_msg}",
                }

            # Helpers for deterministic status updates
            def _refresh_card(agent_name: str):
                ph = st.session_state.agent_cards.get(agent_name)
                if not ph:
                    return
                status_val = st.session_state.agent_status.get(agent_name, "pending")
                # local lightweight renderer (duplicated to avoid cross-scope issues)
                agent_labels = {
                    "Market Analyst": "📊 Market",
                    "Social Analyst": "👥 Social",
                    "News Analyst": "📰 News",
                    "Fundamentals Analyst": "💼 Fundamentals",
                    "Bull Researcher": "🐂 Bull",
                    "Bear Researcher": "🐻 Bear",
                    "Research Manager": "🎯 Manager",
                    "Trader": "💰 Trader",
                    "Risky Analyst": "⚡ Risky",
                    "Neutral Analyst": "⚖️ Neutral",
                    "Safe Analyst": "🛡️ Safe",
                    "Portfolio Manager": "📈 Portfolio",
                }
                agent_display = agent_labels.get(agent_name, agent_name.split()[ -1])
                # bump epoch to ensure unique widget keys on refresh
                st.session_state.render_epoch = st.session_state.get("render_epoch", 0) + 1
                key_suffix = f"_{st.session_state.render_epoch}"
                with ph.container():
                    if status_val == "complete":
                        st.success(f"✅ {agent_display} - Click for details")
                        if st.button(f"View {agent_display} Results", key=f"agent_{agent_name}_complete{key_suffix}", use_container_width=True):
                            st.session_state.selected_agent = agent_name
                            st.session_state.show_agent_details = True
                    elif status_val == "error":
                        st.error(f"❌ {agent_display} - Error")
                        if st.button(f"View {agent_display} Error", key=f"agent_{agent_name}_error{key_suffix}", use_container_width=True):
                            st.session_state.selected_agent = agent_name
                            st.session_state.show_agent_details = True
                    elif status_val == "running":
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                            color: white;
                            padding: 10px;
                            border-radius: 8px;
                            text-align: center;
                            border: 3px solid #007bff;
                            box-shadow: 0 0 20px rgba(0, 123, 255, 0.8);
                            animation: pulse 1.5s infinite;
                            margin: 5px 0;
                        ">
                            🔄 <strong>{agent_display}</strong> - WORKING
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"View {agent_display} Live Progress", key=f"agent_{agent_name}_running{key_suffix}", use_container_width=True, type="primary"):
                            st.session_state.selected_agent = agent_name
                            st.session_state.show_agent_details = True
                    else:
                        st.warning(f"⏳ {agent_display} - Waiting")
                        if st.button(f"View {agent_display} Status", key=f"agent_{agent_name}_pending{key_suffix}", use_container_width=True):
                            st.session_state.selected_agent = agent_name
                            st.session_state.show_agent_details = True

            # Human-friendly status line per agent
            _agent_running_msg = {
                "Market Analyst": "📊 Market Analyst analyzing technical indicators...",
                "Social Analyst": "👥 Social Analyst aggregating social sentiment...",
                "News Analyst": "📰 News Analyst scanning latest headlines...",
                "Fundamentals Analyst": "💼 Fundamentals Analyst evaluating financials...",
                "Bull Researcher": "🐂 Bull Researcher building bullish thesis...",
                "Bear Researcher": "🐻 Bear Researcher building bearish thesis...",
                "Research Manager": "🎯 Research Manager judging investment debate...",
                "Trader": "💰 Trader drafting portfolio plan...",
                "Risky Analyst": "⚡ Risky Analyst assessing high-risk scenario...",
                "Neutral Analyst": "⚖️ Neutral Analyst assessing balanced scenario...",
                "Safe Analyst": "🛡️ Safe Analyst assessing conservative scenario...",
                "Portfolio Manager": "📈 Portfolio Manager finalizing decision...",
            }

            # Rotate status text when multiple agents are running
            def _update_running_status_text():
                running = [a for a, v in st.session_state.agent_status.items() if v == "running"]
                if not running:
                    return
                if len(running) == 1:
                    a = running[0]
                    status_text.text(_agent_running_msg.get(a, f"🔄 {a} working..."))
                else:
                    st.session_state.running_idx = (st.session_state.get("running_idx", -1) + 1) % len(running)
                    a = running[st.session_state.running_idx]
                    status_text.text("Multiple Agents Working: " + _agent_running_msg.get(a, f"🔄 {a} working..."))

            def _set_status(agent: str, status: str):
                if agent in st.session_state.agent_status:
                    prev = st.session_state.agent_status.get(agent)
                    if prev != status:
                        st.session_state.agent_status[agent] = status
                        _refresh_card(agent)
                        # Update the progress status line to reflect the active agent
                        if status == "running":
                            status_text.text(_agent_running_msg.get(agent, f"🔄 {agent} working..."))
                        elif status == "complete":
                            # Only set completion text if no other agents are currently running
                            if not any(v == "running" for v in st.session_state.agent_status.values()):
                                status_text.text(f"✅ {agent} completed.")

            def _next_selected_analyst(current_key: str):
                order = ["market", "social", "news", "fundamentals"]
                try:
                    idx = order.index(current_key)
                except ValueError:
                    return None
                for j in range(idx + 1, len(order)):
                    key = order[j]
                    label = {
                        "market": "Market Analyst",
                        "social": "Social Analyst",
                        "news": "News Analyst",
                        "fundamentals": "Fundamentals Analyst",
                    }[key]
                    # Only move to next if user selected it (using active list)
                    if any(key.capitalize() in a for a in active_selected_analysts) or key in active_selected_analysts:
                        return label
                return None

            # Stream execution
            received = 0
            try:
                for chunk in ta.propagate_stream(stock_symbol, date_str):
                    if not st.session_state.analysis_running:
                        status_text.text("⏹️ Stopped by user")
                        break

                    received += 1
                    # Update progress gently
                    progress_bar.progress(min(95, 20 + received % 70))

                    # Render latest message if present, and capture tool calls
                    try:
                        msgs = chunk.get("messages", []) if isinstance(chunk, dict) else []
                    except Exception:
                        msgs = []
                    if msgs:
                        last_msg = msgs[-1]
                        text = _extract_msg_text(last_msg)
                        with live_feed:
                            st.markdown(f"- 💬 {text}")
                        last_update.caption(f"Last update • {time.strftime('%H:%M:%S')}")
                        _infer_agent_and_update_status(text)
                        # Tool-calls (OpenAI-style)
                        try:
                            tool_calls = getattr(last_msg, "tool_calls", None)
                            if tool_calls:
                                for tc in tool_calls:
                                    if isinstance(tc, dict):
                                        st.session_state.tool_calls.append({
                                            "name": tc.get("name"),
                                            "args": tc.get("args", {}),
                                        })
                                    else:
                                        st.session_state.tool_calls.append({
                                            "name": getattr(tc, "name", "unknown"),
                                            "args": getattr(tc, "args", {}),
                                        })
                        except Exception:
                            pass

                    # Deterministic mapping from chunk to statuses and reports
                    if isinstance(chunk, dict):
                        # Analyst team reports
                        if chunk.get("market_report"):
                            st.session_state.report_sections["market_report"] = chunk["market_report"]
                            _set_status("Market Analyst", "complete")
                            nxt = _next_selected_analyst("market")
                            if nxt:
                                _set_status(nxt, "running")
                                status_text.text(_agent_running_msg.get(nxt, f"🔄 {nxt} working..."))

                        if chunk.get("sentiment_report"):
                            st.session_state.report_sections["sentiment_report"] = chunk["sentiment_report"]
                            _set_status("Social Analyst", "complete")
                            nxt = _next_selected_analyst("social")
                            if nxt:
                                _set_status(nxt, "running")
                                status_text.text(_agent_running_msg.get(nxt, f"🔄 {nxt} working..."))

                        if chunk.get("news_report"):
                            st.session_state.report_sections["news_report"] = chunk["news_report"]
                            _set_status("News Analyst", "complete")
                            nxt = _next_selected_analyst("news")
                            if nxt:
                                _set_status(nxt, "running")
                                status_text.text(_agent_running_msg.get(nxt, f"🔄 {nxt} working..."))

                        if chunk.get("fundamentals_report"):
                            st.session_state.report_sections["fundamentals_report"] = chunk["fundamentals_report"]
                            _set_status("Fundamentals Analyst", "complete")
                            # Move debate team to running
                            for idx, a in enumerate(["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]):
                                _set_status(a, "running")
                                if idx == 0:
                                    status_text.text(_agent_running_msg.get(a, f"🔄 {a} working..."))

                        # Investment debate state
                        inv = chunk.get("investment_debate_state")
                        if inv:
                            # Append latest bull/bear or judge lines into investment_plan incrementally
                            buf = st.session_state.report_sections.get("investment_plan") or ""
                            if inv.get("bull_history"):
                                last_bull = inv["bull_history"].split("\n")[-1]
                                if last_bull:
                                    buf = (buf + f"\n\n### Bull Researcher Analysis\n{last_bull}").strip()
                            if inv.get("bear_history"):
                                last_bear = inv["bear_history"].split("\n")[-1]
                                if last_bear:
                                    buf = (buf + f"\n\n### Bear Researcher Analysis\n{last_bear}").strip()
                            if inv.get("judge_decision"):
                                buf = (buf + f"\n\n### Research Manager Decision\n{inv['judge_decision']}").strip()
                                # Mark debate researchers complete and move to risk
                                for a in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                                    _set_status(a, "complete")
                                _set_status("Risky Analyst", "running")
                            st.session_state.report_sections["investment_plan"] = buf

                        # Trader plan
                        if chunk.get("trader_investment_plan"):
                            st.session_state.report_sections["trader_investment_plan"] = chunk["trader_investment_plan"]
                            _set_status("Risky Analyst", "running")

                        # Risk debate state
                        risk = chunk.get("risk_debate_state")
                        if risk:
                            final_buf = st.session_state.report_sections.get("final_trade_decision") or ""
                            if risk.get("current_risky_response"):
                                _set_status("Risky Analyst", "running")
                                final_buf = (final_buf + f"\n\n### Risky Analyst Analysis\n{risk['current_risky_response']}").strip()
                            if risk.get("current_safe_response"):
                                _set_status("Safe Analyst", "running")
                                final_buf = (final_buf + f"\n\n### Safe Analyst Analysis\n{risk['current_safe_response']}").strip()
                            if risk.get("current_neutral_response"):
                                _set_status("Neutral Analyst", "running")
                                final_buf = (final_buf + f"\n\n### Neutral Analyst Analysis\n{risk['current_neutral_response']}").strip()
                            if risk.get("judge_decision"):
                                _set_status("Portfolio Manager", "running")
                                final_buf = (final_buf + f"\n\n### Portfolio Manager Decision\n{risk['judge_decision']}").strip()
                                # Mark all risk roles complete
                                for a in ["Risky Analyst", "Safe Analyst", "Neutral Analyst", "Portfolio Manager"]:
                                    _set_status(a, "complete")
                            # Always persist the incremental final decision buffer
                            st.session_state.report_sections["final_trade_decision"] = final_buf

                    # Update rotating status text each tick to reflect multiple running agents
                    _update_running_status_text()

                # If completed normally
                if st.session_state.analysis_running:
                    # Mark all agents complete (Phase 2.1 baseline; detailed mapping comes next)
                    for agent in st.session_state.agent_status:
                        st.session_state.agent_status[agent] = "complete"
                    status_text.text("✅ Analysis completed successfully!")
                    progress_bar.progress(100)
            except Exception as stream_error:
                progress_bar.progress(0)
                status_text.text("")
                error_msg = str(stream_error)
                if "Error tokenizing data" in error_msg or "Expected 6 fields" in error_msg:
                    st.error("❌ CSV Data Format Error: Technical indicator analysis failed due to data format issues.")
                else:
                    diag = _diagnose_error(stream_error)
                    # Include status/tool if available for clarity
                    details = []
                    if diag.get("status"):
                        details.append(f"Status {diag['status']}")
                    if diag.get("tool"):
                        details.append(f"Tool: {diag['tool']}")
                    suffix = f" ({', '.join(details)})" if details else ""
                    st.error(diag["message"] + suffix)
                    st.session_state.last_error = diag
                # Mark any running agents as error and refresh cards
                for agent_name, stat in list(st.session_state.agent_status.items()):
                    if stat == "running":
                        st.session_state.agent_status[agent_name] = "error"
                        _refresh_card(agent_name)
                st.session_state.analysis_running = False
                if debug_mode:
                    st.exception(stream_error)
            # Store results from current state when available
            if ta.curr_state and st.session_state.analysis_running:
                final_state = ta.curr_state
                try:
                    decision = ta.process_signal(final_state.get("final_trade_decision", ""))
                except Exception:
                    decision = ""
                st.session_state.analysis_results = {
                    "symbol": stock_symbol,
                    "date": date_str,
                    "decision": decision,
                    "result": final_state,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.analysis_running = False
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"❌ Analysis failed: {str(e)}")
            st.session_state.analysis_running = False
            if debug_mode:
                st.subheader("🐛 Debug Information")
                st.exception(e)
        finally:
            st.markdown('</div>', unsafe_allow_html=True)
if st.session_state.analysis_results and not st.session_state.analysis_running:
        results = st.session_state.analysis_results
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        
        # Results Header with Metrics
        st.subheader(f"📈 Complete Analysis Report: {results['symbol']}")
        
        # Key Metrics Row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Stock Symbol", results['symbol'])
        with col_m2:
            st.metric("Analysis Date", results['date'])
        with col_m3:
            st.metric("Completed", results['timestamp'])
        with col_m4:
            # Extract decision from results for metric
            decision_summary = "BUY" if "BUY" in str(results['decision']).upper() else "HOLD" if "HOLD" in str(results['decision']).upper() else "SELL" if "SELL" in str(results['decision']).upper() else "ANALYZE"
            st.metric("Recommendation", decision_summary)
        
        st.divider()
        
        # Complete Team-Based Report Sections (100% CLI feature parity)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Final Decision", 
            "📊 I. Analyst Team", 
            "🔍 II. Research Team", 
            "💰 III. Trading Team", 
            "⚖️ IV. Risk Management", 
            "📈 V. Portfolio Manager"
        ])
        
        final_state = results.get('result', {})
        
        with tab1:
            st.subheader("🎯 Final Trading Decision & Summary")
            
            # Display final decision prominently
            if results['decision']:
                st.success(f"**Final Recommendation:** {results['decision']}")
                
                # Show decision breakdown if available
                if isinstance(results['decision'], dict):
                    with st.expander("📋 Decision Details"):
                        st.json(results['decision'])
            
            # Executive Summary
            st.markdown("### 📋 Executive Summary")
            
            # Quick overview of all team decisions
            summary_cols = st.columns(2)
            
            with summary_cols[0]:
                st.markdown("**🔍 Research Team Conclusion:**")
                if final_state.get('investment_debate_state', {}).get('judge_decision'):
                    research_decision = final_state['investment_debate_state']['judge_decision'][:200] + "..."
                    st.info(research_decision)
                else:
                    st.warning("Research decision pending")
            
            with summary_cols[1]:
                st.markdown("**📈 Portfolio Manager Decision:**")
                if final_state.get('risk_debate_state', {}).get('judge_decision'):
                    portfolio_decision = final_state['risk_debate_state']['judge_decision'][:200] + "..."
                    st.success(portfolio_decision)
                else:
                    st.warning("Portfolio decision pending")
        
        with tab2:
            st.subheader("📊 I. Analyst Team Reports")
            st.markdown("*Detailed analysis from our core analyst team*")
            
            # Market Analyst Report
            if final_state.get('market_report'):
                with st.container():
                    st.markdown("#### 📈 Market Analyst Report")
                    st.markdown(final_state['market_report'])
                    st.divider()
            
            # Social Analyst Report
            if final_state.get('sentiment_report'):
                with st.container():
                    st.markdown("#### 👥 Social Analyst Report")
                    st.markdown(final_state['sentiment_report'])
                    st.divider()
            
            # News Analyst Report
            if final_state.get('news_report'):
                with st.container():
                    st.markdown("#### 📰 News Analyst Report")
                    st.markdown(final_state['news_report'])
                    st.divider()
            
            # Fundamentals Analyst Report
            if final_state.get('fundamentals_report'):
                with st.container():
                    st.markdown("#### 💼 Fundamentals Analyst Report")
                    st.markdown(final_state['fundamentals_report'])
            
            # Show message if no reports available
            if not any([final_state.get('market_report'), final_state.get('sentiment_report'), 
                       final_state.get('news_report'), final_state.get('fundamentals_report')]):
                st.info("📊 Analyst team reports are still being generated...")
        
        with tab3:
            st.subheader("🔍 II. Research Team Decision")
            st.markdown("*Investment research debate and conclusions*")
            
            if final_state.get('investment_debate_state'):
                debate_state = final_state['investment_debate_state']
                
                # Bull Researcher Analysis
                if debate_state.get('bull_history'):
                    with st.container():
                        st.markdown("#### 🐂 Bull Researcher Analysis")
                        st.markdown(debate_state['bull_history'])
                        st.divider()
                
                # Bear Researcher Analysis
                if debate_state.get('bear_history'):
                    with st.container():
                        st.markdown("#### 🐻 Bear Researcher Analysis")
                        st.markdown(debate_state['bear_history'])
                        st.divider()
                
                # Research Manager Decision
                if debate_state.get('judge_decision'):
                    with st.container():
                        st.markdown("#### 🎯 Research Manager Decision")
                        st.success(debate_state['judge_decision'])
            else:
                st.info("🔍 Research team debate is still in progress...")
        
        with tab4:
            st.subheader("💰 III. Trading Team Plan")
            st.markdown("*Strategic trading recommendations and execution plan*")
            
            if final_state.get('trader_investment_plan'):
                with st.container():
                    st.markdown("#### 💰 Trader Investment Plan")
                    st.markdown(final_state['trader_investment_plan'])
            else:
                st.info("💰 Trading team plan is still being developed...")
        
        with tab5:
            st.subheader("⚖️ IV. Risk Management Team Decision")
            st.markdown("*Comprehensive risk assessment from multiple perspectives*")
            
            if final_state.get('risk_debate_state'):
                risk_state = final_state['risk_debate_state']
                
                # Aggressive (Risky) Analyst Analysis
                if risk_state.get('risky_history'):
                    with st.container():
                        st.markdown("#### ⚡ Aggressive Analyst Analysis")
                        st.markdown(risk_state['risky_history'])
                        st.divider()
                
                # Conservative (Safe) Analyst Analysis
                if risk_state.get('safe_history'):
                    with st.container():
                        st.markdown("#### 🛡️ Conservative Analyst Analysis")
                        st.markdown(risk_state['safe_history'])
                        st.divider()
                
                # Neutral Analyst Analysis
                if risk_state.get('neutral_history'):
                    with st.container():
                        st.markdown("#### ⚖️ Neutral Analyst Analysis")
                        st.markdown(risk_state['neutral_history'])
            else:
                st.info("⚖️ Risk management team assessment is still in progress...")
        
        with tab6:
            st.subheader("📈 V. Portfolio Manager Decision")
            st.markdown("*Final portfolio allocation and investment decision*")
            
            if final_state.get('risk_debate_state', {}).get('judge_decision'):
                with st.container():
                    st.markdown("#### 📈 Portfolio Manager Final Decision")
                    st.success(final_state['risk_debate_state']['judge_decision'])
                    
                    # Show final trade decision if available
                    if final_state.get('final_trade_decision'):
                        st.markdown("#### 🎯 Final Trade Decision")
                        st.info(final_state['final_trade_decision'])
            else:
                st.info("📈 Portfolio manager decision is still being finalized...")

        
        # Detailed Analysis (if debug mode)
        if debug_mode and results['result']:
            st.divider()
            st.subheader("🔍 Complete Technical Analysis (Debug Mode)")
            with st.expander("View Full Raw Analysis Results"):
                if isinstance(results['result'], dict):
                    st.json(results['result'])
                else:
                    st.text(str(results['result']))
        
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.subheader("ℹ️ About TradingAgents")

st.markdown("""
**TradingAgents** is a sophisticated multi-agent framework that uses AI collaboration to make informed trading decisions.

**🎯 Key Features:**
- 🤝 Multi-agent collaboration
- 🎯 Structured debate system
- 📈 Real-time financial data
- 🧠 Memory-based learning
- 🔒 Privacy-focused design

**📚 Research:** Published in arXiv:2412.20138
""")

st.markdown("""
**🔗 Links:**
- [GitHub Repository](https://github.com/TauricResearch/TradingAgents)
- [Research Paper](https://arxiv.org/abs/2412.20138)
- [Discord Community](https://discord.com/invite/hk9PGKShPK)
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>TradingAgents - Multi-Agent LLM Financial Trading Framework</strong></p>
    <p>Developed by <a href="https://tauric.ai/" target="_blank">Tauric Research</a> | Enhanced UI by <a href="https://agentopia.github.io/" target="_blank">Agentopia</a></p>
    <p><em>⚠️ This framework is designed for research purposes. Trading performance may vary. Not intended as financial advice.</em></p>
</div>
""", unsafe_allow_html=True)
