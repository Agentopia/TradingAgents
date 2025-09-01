"""
Analysis Parameters Components
Modular components for rendering analysis parameter configuration
"""

import streamlit as st
from datetime import datetime, date, timedelta


def render_analysis_parameters():
    """Render analysis parameters section"""
    # Date range for analysis
    today = date.today()
    min_date = today - timedelta(days=365)  # 1 year back
    max_date = today - timedelta(days=1)    # Yesterday (markets closed today)
    
    # Initialize default values (available in all cases)
    stock_symbol = "CRCL"
    analysis_date = max_date
    depth_choice = "Beginner"
    preset_rounds = {"Beginner": (1, 1), "Standard": (2, 2), "Deep": (3, 3)}
    preset_debate, preset_risk = preset_rounds.get(depth_choice, (2, 2))
    max_debate_rounds = preset_debate
    max_risk_rounds = preset_risk
    online_tools = True
    debug_mode = False
    
    if not st.session_state.analysis_running:
        st.subheader("📈 Analysis Parameters")
        
        # Create a more compact layout using columns for parameters
        param_col1, param_col2 = st.columns([1, 1])
        
        with param_col1:
            # Stock Symbol Selection
            stock_symbol = st.text_input(
                "Stock Symbol",
                value="CRCL",
                help="Enter the stock ticker symbol (e.g., AAPL, GOOGL, TSLA, SPY)"
            ).upper()
            
            # Date Selection
            analysis_date = st.date_input(
                "Analysis Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                help="Select the date for analysis (market data required)"
            )
            
            # Analysis Depth
            depth_choice = st.radio(
                "Research Depth",
                options=["Beginner", "Standard", "Deep", "Custom"],
                index=0,
                help="Use presets or choose Custom to set rounds manually"
            )
        
        with param_col2:
            # Debate and Risk Rounds
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
                "Risk Analysis Rounds",
                min_value=1,
                max_value=3,
                value=preset_risk,
                disabled=(depth_choice != "Custom"),
                help="Number of risk assessment rounds"
            )
            
            # Additional Options
            online_tools = st.checkbox(
                "Enable Online Tools",
                value=True,
                help="Allow agents to use web search and real-time data"
            )
            
            debug_mode = st.checkbox(
                "Debug Mode",
                value=False,
                help="Show detailed agent communication and debug information"
            )
    
    return {
        'stock_symbol': stock_symbol,
        'analysis_date': analysis_date,
        'depth_choice': depth_choice,
        'max_debate_rounds': max_debate_rounds,
        'max_risk_rounds': max_risk_rounds,
        'online_tools': online_tools,
        'debug_mode': debug_mode
    }


def render_analysis_controls(analysis_params, llm_config, selected_analysts):
    """Render analysis control buttons and start analysis"""
    if not st.session_state.analysis_running:
        # Single Start Analysis button (maintaining original design)
        if st.button("🚀 Start Analysis", type="primary", key="start_analysis_main"):
            return True  # Signal to start analysis
    else:
        # Show stop button during analysis
        if st.button("⏹️ Stop Analysis", type="secondary", use_container_width=True):
            st.session_state.analysis_running = False
            st.rerun()
    
    return False  # No analysis start requested
