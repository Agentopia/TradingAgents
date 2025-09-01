"""
Sidebar Configuration Components
Modular components for rendering sidebar configuration sections
"""

import streamlit as st
import os


def render_api_status():
    """Render API key status check section"""
    with st.expander("🔑 API Status", expanded=True):
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


def render_llm_configuration():
    """Render LLM configuration section"""
    with st.expander("🧠 LLM Configuration", expanded=True):
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
        
        return {
            'provider': provider,
            'backend_url': backend_url,
            'quick_think_model': quick_think_model,
            'deep_think_model': deep_think_model
        }


def render_analyst_selection():
    """Render analyst selection section"""
    with st.expander("👥 Analyst Selection", expanded=False):
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
        return selected_analysts


def render_about_section():
    """Render about TradingAgents section"""
    with st.expander("ℹ️ About TradingAgents", expanded=False):
        st.markdown("""
        **TradingAgents** is a sophisticated multi-agent framework that uses AI collaboration to make informed trading decisions.
        
        **🎯 Key Features:**
        - 🤝 Multi-agent collaboration
        - 🎯 Structured debate system
        - 📈 Real-time financial data
        - 🧠 Memory-based learning
        - 🔒 Privacy-focused design
        
        **🔬 How It Works:**
        1. **Data Collection**: Market, news, and social sentiment analysis
        2. **Multi-Agent Debate**: Bull vs Bear researchers present arguments
        3. **Research Management**: Synthesis of different perspectives
        4. **Risk Assessment**: Portfolio management with risk analysis
        5. **Final Decision**: Comprehensive trading recommendation
        
        **📊 Agent Types:**
        - **Analysts**: Market, Social, News, Fundamentals
        - **Researchers**: Bull, Bear, Risk Assessment
        - **Managers**: Research coordination, Portfolio management
        - **Trader**: Final execution and strategy
        """)


def render_sidebar_configuration():
    """Render complete sidebar configuration"""
    st.header("🔧 Trading Configuration")
    
    # Render all sidebar sections
    render_api_status()
    llm_config = render_llm_configuration()
    selected_analysts = render_analyst_selection()
    render_about_section()
    
    return {
        'llm_config': llm_config,
        'selected_analysts': selected_analysts
    }
