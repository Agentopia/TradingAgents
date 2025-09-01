"""
TradingAgents Streamlit Interface - Clean Rebuild
A modern web UI for multi-agent trading analysis
"""

import streamlit as st
import time
import os
import sys
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
from collections import deque

# Add the parent directory to the path to import tradingagents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="TradingAgents - Multi-Agent Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components (with fallback if they don't exist)
try:
    from components.sidebar_config import render_sidebar_configuration
    from components.analysis_params import render_analysis_parameters, render_analysis_controls
    from components.css_loader import load_css
    from components.header_component import render_header
    from components.report_components import render_analysis_report
    from components.agent_status import render_agent_card
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    st.warning("⚠️ Component modules not found. Using fallback UI.")

# Apply custom styling
if COMPONENTS_AVAILABLE:
    load_css()

# Add agent status button CSS
st.markdown("""
<style>
/* Agent Status Button Styling */
.stButton > button[data-testid="baseButton-secondary"] {
    background-color: #4a4a4a !important;
    color: white !important;
    border: none !important;
}

.stButton > button[data-testid="baseButton-primary"] {
    background-color: #1f77b4 !important;
    color: white !important;
    border: none !important;
    animation: agentPulse 2s infinite;
}

/* Pulsing animation for running agents */
@keyframes agentPulse {
    0% { box-shadow: 0 0 0 0 rgba(31, 119, 180, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(31, 119, 180, 0); }
    100% { box-shadow: 0 0 0 0 rgba(31, 119, 180, 0); }
}

/* Completed agent styling */
.agent-completed .stButton > button {
    background-color: #2d5a27 !important;
    color: white !important;
}

/* Error agent styling */
.agent-error .stButton > button {
    background-color: #666666 !important;
    color: white !important;
}

/* Pending agent styling */
.agent-pending .stButton > button {
    background-color: #4a4a4a !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    
    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = deque(maxlen=100)
    
    if 'tool_calls' not in st.session_state:
        st.session_state.tool_calls = deque(maxlen=100)
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    
    # Initialize agent status - matching CLI exactly
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {
            # Analyst Team
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending", 
            "Fundamentals Analyst": "pending",
            # Research Team
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            # Trading Team
            "Trader": "pending",
            # Risk Management Team
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            # Portfolio Management Team
            "Portfolio Manager": "pending",
        }
    # Initialize report sections - matching CLI exactly
    if 'report_sections' not in st.session_state:
        st.session_state.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

# Fallback UI components
def fallback_render_header():
    """Fallback header component"""
    st.title("📈 TradingAgents - Multi-Agent Trading Analysis")
    st.markdown("*Powered by collaborative AI agents for comprehensive market analysis*")

def fallback_render_sidebar():
    """Fallback sidebar configuration"""
    st.header("🔧 Configuration")
    
    # API Status
    with st.expander("🔑 API Status", expanded=True):
        openai_key = os.getenv("OPENAI_API_KEY", "")
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        
        if openai_key:
            st.success("✅ OpenAI API Key: Configured")
        else:
            st.error("❌ OpenAI API Key: Missing")
        
        if finnhub_key:
            st.success("✅ Finnhub API Key: Configured")
        else:
            st.error("❌ Finnhub API Key: Missing")
    
    # LLM Configuration
    with st.expander("🧠 LLM Configuration", expanded=True):
        provider = st.selectbox("Provider", ["OpenAI"], index=0)
        model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
        backend_url = st.text_input("Backend URL (optional)", value=os.getenv('OPENAI_API_BASE_URL', ''))
    
    # Analyst Selection
    with st.expander("👥 Analyst Selection", expanded=False):
        analysts = ["market", "social", "news", "fundamentals"]
        selected = st.multiselect("Select Analysts", analysts, default=analysts)
    
    return {
        'llm_config': {
            'provider': provider,
            'model': model,
            'backend_url': backend_url
        },
        'selected_analysts': selected
    }

def fallback_render_analysis_params():
    """Fallback analysis parameters"""
    st.subheader("📊 Analysis Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        stock_symbol = st.text_input("Stock Symbol", value="NVDA", placeholder="e.g., AAPL, TSLA")
        analysis_date = st.date_input("Analysis Date", value=date.today() - timedelta(days=1))
    
    with col2:
        max_debate_rounds = st.slider("Debate Rounds", 1, 5, 2)
        max_risk_rounds = st.slider("Risk Rounds", 1, 3, 2)
    
    online_tools = st.checkbox("Enable Online Tools", value=True)
    debug_mode = st.checkbox("Debug Mode", value=False)
    
    return {
        'stock_symbol': stock_symbol,
        'analysis_date': analysis_date,
        'max_debate_rounds': max_debate_rounds,
        'max_risk_rounds': max_risk_rounds,
        'online_tools': online_tools,
        'debug_mode': debug_mode
    }

def fallback_render_analysis_controls():
    """Fallback analysis controls"""
    if st.session_state.analysis_running:
        if st.button("⏹️ Stop Analysis", type="secondary", use_container_width=True):
            st.session_state.analysis_running = False
            # Debug logging for stop button
            if st.session_state.get('debug_mode', False):
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Stop Analysis button clicked")
            st.rerun()
        return False
    else:
        return st.button("🚀 Start Analysis", type="primary", use_container_width=True)

def fallback_render_agent_card(placeholder, agent, status):
    """Fallback agent status card"""
    if status == "complete":
        placeholder.success(f"✅ {agent}")
    elif status == "running":
        placeholder.info(f"🔄 {agent}")
    elif status == "error":
        placeholder.error(f"❌ {agent}")
    else:
        placeholder.warning(f"⏳ {agent}")

def render_agent_button(placeholder, agent, status):
    """Render agent button with proper styling based on status"""
    # Use session state counter for unique keys
    if 'button_counter' not in st.session_state:
        st.session_state.button_counter = 0
    st.session_state.button_counter += 1
    unique_id = st.session_state.button_counter
    
    with placeholder.container():
        if status == "in_progress":
            st.button(f"🔄 {agent}", key=f"agent_{agent}_{status}_{unique_id}", disabled=True, 
                     help=f"{agent} is currently working...",
                     type="primary")
        elif status == "completed":
            st.button(f"✅ {agent}", key=f"agent_{agent}_{status}_{unique_id}", disabled=True,
                     help=f"{agent} has completed analysis",
                     type="secondary")
        elif status == "error":
            st.button(f"❌ {agent}", key=f"agent_{agent}_{status}_{unique_id}", disabled=True,
                     help=f"{agent} encountered an error",
                     type="secondary")
        else:  # pending
            st.button(f"⏳ {agent}", key=f"agent_{agent}_{status}_{unique_id}", disabled=True,
                     help=f"{agent} is waiting to start",
                     type="secondary")

def update_agent_status(agent: str, status: str):
    """Update agent status and refresh UI placeholder"""
    if agent in st.session_state.agent_status:
        st.session_state.agent_status[agent] = status
        # Update the placeholder if it exists
        if hasattr(st.session_state, 'agent_placeholders') and agent in st.session_state.agent_placeholders:
            render_agent_button(st.session_state.agent_placeholders[agent], agent, status)

def extract_content_string(content):
    """Extract string content from various message formats - matching CLI"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle Anthropic's list format
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'tool_use':
                    text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
            else:
                text_parts.append(str(item))
        return ' '.join(text_parts)
    else:
        return str(content)

def update_research_team_status(status):
    """Update status for all research team members and trader - matching CLI"""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
    for agent in research_team:
        update_agent_status(agent, status)

def run_real_analysis(stock_symbol: str, analysis_date: date, llm_config: dict, selected_analysts: list, debug_mode: bool):
    """Run real TradingAgents analysis - matching CLI streaming behavior exactly"""
    
    # Import TradingAgents
    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG
    except ImportError as e:
        st.error(f"Failed to import TradingAgents: {e}")
        st.error("Please ensure TradingAgents is properly installed.")
        st.session_state.analysis_running = False
        return
    
    # Create progress placeholders
    progress_container = st.container()
    with progress_container:
        st.subheader("🔄 Analysis in Progress...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown("### 💬 Live Analysis Feed")
        messages_container = st.empty()
        
        # Initialize with a placeholder message to verify container works
        messages_container.markdown("- Waiting for analysis to start...")
        
        st.markdown("### 📊 Current Report")
        report_container = st.empty()
    
    # Configure TradingAgents - matching CLI config
    config = DEFAULT_CONFIG.copy()
    config["llm_config"] = {
        "provider": llm_config.get('provider', 'OpenAI').lower(),
        "config_list": [{
            "model": llm_config.get('model', 'gpt-4o'),
            "base_url": llm_config.get('backend_url') if llm_config.get('backend_url') else None,
        }],
        "temperature": 0.1,
    }
    config["research_depth"] = "deep"
    config["online_tools"] = True
    
    # Map selected analysts to TradingAgents format
    analyst_mapping = {
        'market': 'market',
        'social': 'social', 
        'news': 'news',
        'fundamentals': 'fundamentals'
    }
    
    if selected_analysts:
        mapped_analysts = [analyst_mapping.get(key, key) for key in selected_analysts]
    else:
        mapped_analysts = ['market', 'social', 'news', 'fundamentals']
    
    try:
        # Initialize TradingAgents
        status_text.text("🔧 Initializing TradingAgents framework...")
        ta = TradingAgentsGraph(selected_analysts=mapped_analysts, debug=debug_mode, config=config)
        
        status_text.text("🚀 Starting analysis...")
        date_str = analysis_date.strftime("%Y-%m-%d")
        
        # Initialize state and get graph args - matching CLI exactly
        init_agent_state = ta.propagator.create_initial_state(stock_symbol, date_str)
        args = ta.propagator.get_graph_args()
        
        # Add initial system messages
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.progress_messages.appendleft(f"{timestamp} - Selected ticker: {stock_symbol}")
        st.session_state.progress_messages.appendleft(f"{timestamp} - Analysis date: {date_str}")
        st.session_state.progress_messages.appendleft(f"{timestamp} - Selected analysts: {', '.join(mapped_analysts)}")
        
        # Set first analyst to in_progress
        if mapped_analysts:
            first_analyst = f"{mapped_analysts[0].capitalize()} Analyst"
            update_agent_status(first_analyst, "in_progress")
            
        # Debug: Show which analysts are selected
        if debug_mode:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Starting analysis with analysts: {mapped_analysts}")
        
        # Stream the analysis - matching CLI exactly
        trace = []
        try:
            # Debug: Log streaming start
            if debug_mode:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Starting streaming loop")
                messages_container.markdown(f"- {timestamp} [DEBUG] Starting streaming loop")
            
            for chunk in ta.graph.stream(init_agent_state, **args):
                if not st.session_state.analysis_running:
                    break
                
                # Debug: Log each chunk received
                if debug_mode:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    chunk_keys = list(chunk.keys())
                    st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Received chunk with keys: {chunk_keys}")
                    
                    # Check if this chunk should trigger Social Analyst
                    if "sentiment_report" in chunk_keys:
                        st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Found sentiment_report in chunk!")
                    elif any(key.startswith("social") for key in chunk_keys):
                        st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Found social-related key in chunk: {[k for k in chunk_keys if 'social' in k.lower()]}")
                    
                    # Update UI immediately with debug info
                    all_debug_messages = []
                    for msg in list(st.session_state.progress_messages)[:15]:
                        all_debug_messages.append(f"- {msg}")
                    messages_container.markdown("\n".join(all_debug_messages))
                
                if len(chunk.get("messages", [])) > 0:
                    # Get the last message from the chunk
                    last_message = chunk["messages"][-1]
                    
                    # Extract message content and type
                    if hasattr(last_message, "content"):
                        content = extract_content_string(last_message.content)
                        msg_type = "Reasoning"
                    else:
                        content = str(last_message)
                        msg_type = "System"
                    
                    # Add message to buffer
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.progress_messages.appendleft(f"{timestamp} [{msg_type}] {content[:100]}...")
                    
                    # If it's a tool call, add it to tool calls
                    if hasattr(last_message, "tool_calls"):
                        for tool_call in last_message.tool_calls:
                            if isinstance(tool_call, dict):
                                st.session_state.tool_calls.appendleft(f"{timestamp} [Tool] {tool_call['name']}: {str(tool_call['args'])[:50]}...")
                            else:
                                st.session_state.tool_calls.appendleft(f"{timestamp} [Tool] {tool_call.name}: {str(tool_call.args)[:50]}...")
                    
                    # Update UI displays immediately after adding messages
                    all_messages = []
                    for msg in list(st.session_state.progress_messages)[:10]:
                        all_messages.append(f"- {msg}")
                    for tool in list(st.session_state.tool_calls)[:5]:
                        all_messages.append(f"- {tool}")
                    
                    messages_container.markdown("\n".join(all_messages[:15]))
                
                # Update reports and agent status based on chunk content - EXACTLY matching CLI
                
                # Analyst Team Reports
                if "market_report" in chunk and chunk["market_report"]:
                    st.session_state.report_sections["market_report"] = chunk["market_report"]
                    update_agent_status("Market Analyst", "completed")
                    # Set next analyst to in_progress
                    # Check if social analyst is selected using correct format
                    selected_analyst_names = [f"{analyst.lower()}" for analyst in mapped_analysts]
                    if "social" in selected_analyst_names:
                        update_agent_status("Social Analyst", "in_progress")
                        if debug_mode:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Starting Social Analyst after Market completion")
                
                if "sentiment_report" in chunk and chunk["sentiment_report"]:
                    if debug_mode:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Received sentiment_report chunk")
                    st.session_state.report_sections["sentiment_report"] = chunk["sentiment_report"]
                    update_agent_status("Social Analyst", "completed")
                    # Set next analyst to in_progress
                    selected_analyst_names = [f"{analyst.lower()}" for analyst in mapped_analysts]
                    if "news" in selected_analyst_names:
                        update_agent_status("News Analyst", "in_progress")
                        if debug_mode:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Starting News Analyst")
                
                if "news_report" in chunk and chunk["news_report"]:
                    st.session_state.report_sections["news_report"] = chunk["news_report"]
                    update_agent_status("News Analyst", "completed")
                    # Set next analyst to in_progress
                    selected_analyst_names = [f"{analyst.lower()}" for analyst in mapped_analysts]
                    if "fundamentals" in selected_analyst_names:
                        update_agent_status("Fundamentals Analyst", "in_progress")
                
                if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                    st.session_state.report_sections["fundamentals_report"] = chunk["fundamentals_report"]
                    update_agent_status("Fundamentals Analyst", "completed")
                    # Set all research team members to in_progress
                    update_research_team_status("in_progress")
                
                # Update progress
                completed_count = sum(1 for s in st.session_state.agent_status.values() if s == 'completed')
                total_agents = len(st.session_state.agent_status)
                progress_val = (completed_count / total_agents) if total_agents > 0 else 0
                progress_bar.progress(progress_val)
                
                # Update status text
                in_progress_agents = [agent for agent, status in st.session_state.agent_status.items() if status == 'in_progress']
                if in_progress_agents:
                    status_text.text(f"🔄 {in_progress_agents[0]} is working...")
                else:
                    status_text.text("🔄 Processing...")
                
                trace.append(chunk)
                time.sleep(0.1)  # Small delay for UI responsiveness
            
            # Research Team - Handle Investment Debate State
            if "investment_debate_state" in chunk and chunk["investment_debate_state"]:
                debate_state = chunk["investment_debate_state"]
                
                # Update Bull Researcher status and report
                if "bull_history" in debate_state and debate_state["bull_history"]:
                    update_research_team_status("in_progress")
                    bull_responses = debate_state["bull_history"].split("\n")
                    latest_bull = bull_responses[-1] if bull_responses else ""
                    if latest_bull:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.progress_messages.appendleft(f"{timestamp} [Reasoning] Bull Researcher: {latest_bull[:100]}...")
                        st.session_state.report_sections["investment_plan"] = f"### Bull Researcher Analysis\n{latest_bull}"
                
                # Update Bear Researcher status and report
                if "bear_history" in debate_state and debate_state["bear_history"]:
                    update_research_team_status("in_progress")
                    bear_responses = debate_state["bear_history"].split("\n")
                    latest_bear = bear_responses[-1] if bear_responses else ""
                    if latest_bear:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.progress_messages.appendleft(f"{timestamp} [Reasoning] Bear Researcher: {latest_bear[:100]}...")
                        current_plan = st.session_state.report_sections.get("investment_plan", "")
                        st.session_state.report_sections["investment_plan"] = f"{current_plan}\n\n### Bear Researcher Analysis\n{latest_bear}"
                
                # Update Research Manager status and final decision
                if "judge_decision" in debate_state and debate_state["judge_decision"]:
                    update_research_team_status("in_progress")
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.progress_messages.appendleft(f"{timestamp} [Reasoning] Research Manager: {debate_state['judge_decision'][:100]}...")
                    current_plan = st.session_state.report_sections.get("investment_plan", "")
                    st.session_state.report_sections["investment_plan"] = f"{current_plan}\n\n### Research Manager Decision\n{debate_state['judge_decision']}"
                    # Mark all research team members as completed
                    update_research_team_status("completed")
                    # Set first risk analyst to in_progress
                    update_agent_status("Risky Analyst", "in_progress")
            
            # Trading Team
            if "trader_investment_plan" in chunk and chunk["trader_investment_plan"]:
                st.session_state.report_sections["trader_investment_plan"] = chunk["trader_investment_plan"]
                # Set first risk analyst to in_progress
                update_agent_status("Risky Analyst", "in_progress")
            
            # Risk Management Team - Handle Risk Debate State
            if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                risk_state = chunk["risk_debate_state"]
                
                # Update Risky Analyst status and report
                if "current_risky_response" in risk_state and risk_state["current_risky_response"]:
                    update_agent_status("Risky Analyst", "in_progress")
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.progress_messages.appendleft(f"{timestamp} [Reasoning] Risky Analyst: {risk_state['current_risky_response'][:100]}...")
                    st.session_state.report_sections["final_trade_decision"] = f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}"
                
                # Update Safe Analyst status and report
                if "current_safe_response" in risk_state and risk_state["current_safe_response"]:
                    update_agent_status("Safe Analyst", "in_progress")
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.progress_messages.appendleft(f"{timestamp} [Reasoning] Safe Analyst: {risk_state['current_safe_response'][:100]}...")
                    st.session_state.report_sections["final_trade_decision"] = f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}"
                
                # Update Neutral Analyst status and report
                if "current_neutral_response" in risk_state and risk_state["current_neutral_response"]:
                    update_agent_status("Neutral Analyst", "in_progress")
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.progress_messages.appendleft(f"{timestamp} [Reasoning] Neutral Analyst: {risk_state['current_neutral_response'][:100]}...")
                    st.session_state.report_sections["final_trade_decision"] = f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}"
                
                # Update Portfolio Manager status and final decision
                if "judge_decision" in risk_state and risk_state["judge_decision"]:
                    update_agent_status("Portfolio Manager", "in_progress")
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.progress_messages.appendleft(f"{timestamp} [Reasoning] Portfolio Manager: {risk_state['judge_decision'][:100]}...")
                    st.session_state.report_sections["final_trade_decision"] = f"### Portfolio Manager Decision\n{risk_state['judge_decision']}"
                    # Mark risk analysts as completed
                    update_agent_status("Risky Analyst", "completed")
                    update_agent_status("Safe Analyst", "completed")
                    update_agent_status("Neutral Analyst", "completed")
                    update_agent_status("Portfolio Manager", "completed")
            
                # Update UI displays inside the streaming loop
                # Display combined messages and tool calls
                all_messages = []
                for msg in list(st.session_state.progress_messages)[:10]:
                    all_messages.append(f"- {msg}")
                for tool in list(st.session_state.tool_calls)[:5]:
                    all_messages.append(f"- {tool}")
                
                messages_container.markdown("\n".join(all_messages[:15]))
                
                # Display current report section
                current_report = None
                section_titles = {
                    "market_report": "Market Analysis",
                    "sentiment_report": "Social Sentiment", 
                    "news_report": "News Analysis",
                    "fundamentals_report": "Fundamentals Analysis",
                    "investment_plan": "Research Team Decision",
                    "trader_investment_plan": "Trading Team Plan",
                    "final_trade_decision": "Portfolio Management Decision",
                }
                
                # Find the most recently updated section
                for section, content in st.session_state.report_sections.items():
                    if content and content.strip():
                        current_report = f"### {section_titles.get(section, section.title())}\n{content}"
                        break
                
                if current_report:
                    report_container.markdown(current_report)
            
            # Display current report section
            current_report = None
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment", 
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            
            # Find the most recently updated section
            for section, content in st.session_state.report_sections.items():
                if content is not None:
                    current_report = f"### {section_titles[section]}\n{content}"
            
            if current_report:
                report_container.markdown(current_report)
            else:
                report_container.markdown("*Waiting for analysis report...*")
            
            # Update progress
            completed_count = sum(1 for s in st.session_state.agent_status.values() if s == 'completed')
            total_agents = len(st.session_state.agent_status)
            progress_val = (completed_count / total_agents) if total_agents > 0 else 0
            progress_bar.progress(progress_val)
            
            # Update status text
            in_progress_agents = [agent for agent, status in st.session_state.agent_status.items() if status == 'in_progress']
            if in_progress_agents:
                status_text.text(f"🔄 {in_progress_agents[0]} is working...")
            else:
                status_text.text("🔄 Processing...")
            
            trace.append(chunk)
            time.sleep(0.1)  # Small delay for UI responsiveness
            
            # Debug: Log when streaming loop exits
            if debug_mode:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.progress_messages.appendleft(f"{timestamp} [DEBUG] Streaming loop completed normally")
        
        except Exception as streaming_error:
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.progress_messages.appendleft(f"{timestamp} [ERROR] Streaming failed: {str(streaming_error)}")
            if debug_mode:
                st.exception(streaming_error)
            # Mark current running agents as error
            for agent, status in st.session_state.agent_status.items():
                if status == "in_progress":
                    update_agent_status(agent, "error")
            raise streaming_error
        
        # Get final state and complete analysis
        if trace:
            final_state = trace[-1]
            
            # Process final decision - extract from text
            decision = None
            if "final_trade_decision" in final_state and final_state["final_trade_decision"]:
                decision_text = str(final_state["final_trade_decision"]).upper()
                if "BUY" in decision_text:
                    decision = "BUY"
                elif "SELL" in decision_text:
                    decision = "SELL"
                elif "HOLD" in decision_text:
                    decision = "HOLD"
                else:
                    decision = "ANALYZE"
            
            # Update all agent statuses to completed
            for agent in st.session_state.agent_status:
                update_agent_status(agent, "completed")
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.progress_messages.appendleft(f"{timestamp} [Analysis] Completed analysis for {date_str}")
            
            # Update final report sections
            for section in st.session_state.report_sections.keys():
                if section in final_state:
                    st.session_state.report_sections[section] = final_state[section]
            
            # Store results with decision
            st.session_state.analysis_results = {
                "symbol": stock_symbol,
                "date": date_str,
                "result": final_state,
                "decision": decision,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.analysis_running = False
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(1.0)
            st.success("🎉 Real analysis completed successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.session_state.analysis_running = False
        st.session_state.last_error = str(e)
        if debug_mode:
            st.exception(e)
        st.rerun()

# Initialize session state
init_session_state()

# Render header
if COMPONENTS_AVAILABLE:
    render_header()
else:
    fallback_render_header()

# Sidebar configuration
with st.sidebar:
    if COMPONENTS_AVAILABLE:
        sidebar_config = render_sidebar_configuration()
    else:
        sidebar_config = fallback_render_sidebar()

# Main layout
st.markdown('<div class="full-width-container">', unsafe_allow_html=True)

# Create main layout
title_col1, title_col2 = st.columns([8, 2], gap="small")
with title_col1:
    st.header("🎯 Trading Analysis Dashboard")
with title_col2:
    st.header("🤖 Agent Status")

main_content_col, agent_status_col = st.columns([8, 2], gap="small")

# Main content area
with main_content_col:
    # Analysis parameters
    if COMPONENTS_AVAILABLE:
        analysis_params = render_analysis_parameters()
        start_analysis = render_analysis_controls(analysis_params, sidebar_config.get('llm_config', {}), sidebar_config.get('selected_analysts', []))
    else:
        analysis_params = fallback_render_analysis_params()
        start_analysis = fallback_render_analysis_controls()
    
    # Handle start analysis
    if start_analysis:
        # Validate inputs
        if not analysis_params['stock_symbol']:
            st.error("Please enter a stock symbol")
            st.stop()
        
        # Check API keys
        openai_key = os.getenv("OPENAI_API_KEY", "")
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        backend_url = sidebar_config.get('llm_config', {}).get('backend_url', "")
        if not openai_key and not backend_url:
            st.error("Please provide OpenAI API key or backend URL")
        elif not analysis_params['stock_symbol']:
            st.error("Please enter a stock symbol")
        else:
            st.session_state.analysis_running = True
            st.session_state.progress_messages = deque(maxlen=50)
            st.session_state.analysis_results = None
            st.session_state.last_error = None
            
            # Reset agent status
            for agent in ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"]:
                update_agent_status(agent, "pending")
            
            st.rerun()
    
    st.divider()
    
    # Show analysis progress if running
    if st.session_state.analysis_running:
        try:
            run_real_analysis(
                analysis_params['stock_symbol'],
                analysis_params['analysis_date'],
                sidebar_config.get('llm_config', {}),
                sidebar_config.get('selected_analysts', []),
                analysis_params.get('debug_mode', False)
            )
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.session_state.analysis_running = False
            st.session_state.last_error = str(e)
            st.rerun()

# Agent status sidebar - Create placeholders for dynamic updates
with agent_status_col:
    # Agent status cards
    teams = {
        "📈 Analyst Team": ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"],
        "🔍 Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "💰 Trading & Risk": ["Trader", "Risky Analyst", "Neutral Analyst", "Safe Analyst", "Portfolio Manager"]
    }
    
    # Store placeholders for dynamic updates
    if 'agent_placeholders' not in st.session_state:
        st.session_state.agent_placeholders = {}
    
    for team_name, agents in teams.items():
        completed = sum(1 for agent in agents if st.session_state.agent_status[agent] == 'completed')
        total = len(agents)
        
        with st.expander(f"{team_name} ({completed}/{total})", expanded=True):
            for agent in agents:
                # Create placeholder for each agent button
                placeholder = st.empty()
                st.session_state.agent_placeholders[agent] = placeholder
                
                # Initial render
                status = st.session_state.agent_status[agent]
                render_agent_button(placeholder, agent, status)

# Results section
if st.session_state.analysis_results and not st.session_state.analysis_running:
    st.header("📊 Analysis Results")
    
    if COMPONENTS_AVAILABLE:
        render_analysis_report(st.session_state.analysis_results, analysis_params.get('debug_mode', False))
    else:
        # Fallback results display
        results = st.session_state.analysis_results
        st.subheader(f"Results for {results['symbol']} - {results['date']}")
        
        result_data = results.get('result', {})
        
        if result_data.get('market_report'):
            st.subheader("📈 Market Analysis")
            st.markdown(result_data['market_report'])
        
        if result_data.get('news_report'):
            st.subheader("📰 News Analysis")
            st.markdown(result_data['news_report'])
        
        if result_data.get('fundamentals_report'):
            st.subheader("💰 Fundamentals")
            st.markdown(result_data['fundamentals_report'])
        
        if result_data.get('trader_investment_plan'):
            st.subheader("🎯 Investment Decision")
            st.markdown(result_data['trader_investment_plan'])

# Error display
if st.session_state.get("last_error") and not st.session_state.analysis_running:
    st.error(f"An error occurred: {st.session_state.last_error}")

st.markdown('</div>', unsafe_allow_html=True)
