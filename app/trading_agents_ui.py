"""
TradingAgents Streamlit Interface
A user-friendly web interface for the TradingAgents multi-agent trading framework
"""

import streamlit as st
import os
import sys
from datetime import datetime, date
import json
from typing import Dict, Any

# Add the parent directory to the path to import tradingagents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f4e79, #2e7d32);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
        margin: 0.5rem 0;
    }
    .result-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79, #2e7d32);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 TradingAgents</h1>
    <p>Multi-Agent LLM Financial Trading Framework</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key Configuration
    st.subheader("API Keys")
    
    # OpenAI Configuration
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Required for GPT models"
    )
    
    # Google API Configuration
    google_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Required for Gemini models"
    )
    
    st.divider()
    
    # LLM Provider Selection
    st.subheader("LLM Configuration")
    
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["openai", "google"],
        index=0,
        help="Choose your preferred LLM provider"
    )
    
    if llm_provider == "openai":
        deep_think_model = st.selectbox(
            "Deep Think Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=1
        )
        quick_think_model = st.selectbox(
            "Quick Think Model",
            options=["gpt-4o-mini", "gpt-3.5-turbo"],
            index=0
        )
        backend_url = "https://api.openai.com/v1"
    else:  # google
        deep_think_model = st.selectbox(
            "Deep Think Model",
            options=["gemini-2.0-flash", "gemini-1.5-pro"],
            index=0
        )
        quick_think_model = st.selectbox(
            "Quick Think Model",
            options=["gemini-2.0-flash", "gemini-1.5-flash"],
            index=0
        )
        backend_url = "https://generativelanguage.googleapis.com/v1"
    
    st.divider()
    
    # Trading Configuration
    st.subheader("Trading Parameters")
    
    max_debate_rounds = st.slider(
        "Max Debate Rounds",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of debate rounds between agents"
    )
    
    max_risk_discuss_rounds = st.slider(
        "Max Risk Discussion Rounds",
        min_value=1,
        max_value=3,
        value=1,
        help="Number of risk management discussion rounds"
    )
    
    online_tools = st.checkbox(
        "Enable Online Tools",
        value=True,
        help="Allow agents to use online financial data sources"
    )
    
    debug_mode = st.checkbox(
        "Debug Mode",
        value=False,
        help="Enable detailed logging and debugging information"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Trading Analysis")
    
    # Stock symbol input
    stock_symbol = st.text_input(
        "Stock Symbol",
        value="NVDA",
        help="Enter the stock ticker symbol (e.g., AAPL, GOOGL, TSLA)"
    ).upper()
    
    # Date selection
    analysis_date = st.date_input(
        "Analysis Date",
        value=date.today(),
        help="Select the date for analysis"
    )
    
    # Convert date to string format
    date_str = analysis_date.strftime("%Y-%m-%d")
    
    # Analysis button
    if st.button("🚀 Run Trading Analysis", type="primary"):
        # Validate inputs
        if not stock_symbol:
            st.error("Please enter a stock symbol")
            st.stop()
        
        # Check API keys
        if llm_provider == "openai" and not openai_key:
            st.error("OpenAI API key is required for OpenAI models")
            st.stop()
        elif llm_provider == "google" and not google_key:
            st.error("Google API key is required for Gemini models")
            st.stop()
        
        # Set environment variables
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key
        
        # Create configuration
        config = DEFAULT_CONFIG.copy()
        config.update({
            "llm_provider": llm_provider,
            "backend_url": backend_url,
            "deep_think_llm": deep_think_model,
            "quick_think_llm": quick_think_model,
            "max_debate_rounds": max_debate_rounds,
            "max_risk_discuss_rounds": max_risk_discuss_rounds,
            "online_tools": online_tools
        })
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize TradingAgents
            status_text.text("Initializing TradingAgents framework...")
            progress_bar.progress(20)
            
            ta = TradingAgentsGraph(debug=debug_mode, config=config)
            
            # Run analysis
            status_text.text(f"Analyzing {stock_symbol} for {date_str}...")
            progress_bar.progress(50)
            
            # Execute the trading analysis
            result, decision = ta.propagate(stock_symbol, date_str)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Display results
            st.success("✅ Trading analysis completed successfully!")
            
            # Results container
            with st.container():
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                st.subheader(f"📈 Analysis Results for {stock_symbol}")
                st.write(f"**Date:** {date_str}")
                st.write(f"**Symbol:** {stock_symbol}")
                
                # Display decision
                if decision:
                    st.subheader("🎯 Trading Decision")
                    if isinstance(decision, dict):
                        st.json(decision)
                    else:
                        st.write(decision)
                
                # Display full result if available
                if result and debug_mode:
                    st.subheader("🔍 Detailed Analysis (Debug Mode)")
                    with st.expander("View Full Analysis"):
                        if isinstance(result, dict):
                            st.json(result)
                        else:
                            st.text(str(result))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"❌ Analysis failed: {str(e)}")
            
            if debug_mode:
                st.subheader("🐛 Debug Information")
                st.exception(e)

with col2:
    st.header("🤖 Agent Teams")
    
    # Agent team information
    st.markdown("""
    <div class="agent-card">
        <h4>📊 Analyst Team</h4>
        <p>Conducts fundamental and technical analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="agent-card">
        <h4>🔍 Research Team</h4>
        <p>Bullish and bearish researchers debate insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="agent-card">
        <h4>⚖️ Risk Management</h4>
        <p>Evaluates risks and trading decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Information section
    st.header("ℹ️ About TradingAgents")
    
    st.markdown("""
    **TradingAgents** is a sophisticated multi-agent framework that uses AI collaboration to make informed trading decisions.
    
    **Key Features:**
    - 🤝 Multi-agent collaboration
    - 🎯 Structured debate system
    - 📊 Real-time financial data
    - 🧠 Memory-based learning
    - 🔒 Privacy-focused design
    
    **Research:** Published in arXiv:2412.20138
    """)
    
    # Links
    st.markdown("""
    **Links:**
    - [GitHub Repository](https://github.com/TauricResearch/TradingAgents)
    - [Research Paper](https://arxiv.org/abs/2412.20138)
    - [Discord Community](https://discord.com/invite/hk9PGKShPK)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>TradingAgents - Multi-Agent LLM Financial Trading Framework</p>
    <p>Developed by <a href="https://tauric.ai/" target="_blank">Tauric Research</a> | Integrated by <a href="https://agentopia.github.io/" target="_blank">Agentopia</a></p>
</div>
""", unsafe_allow_html=True)
