"""
Header Component
Renders the main header section for the TradingAgents UI
"""

import streamlit as st


def render_header() -> None:
    """
    Render the main header section with title and description
    """
    st.markdown("""
    <div class="main-header">
        <h1>📈 TradingAgents</h1>
        <h3>Multi-Agent LLM Financial Trading Framework</h3>
        <p>Powered by specialized AI agents for comprehensive market analysis</p>
    </div>
    """, unsafe_allow_html=True)


def render_custom_header(title: str, subtitle: str, description: str, icon: str = "📈") -> None:
    """
    Render a custom header with specified content
    
    Args:
        title: Main title text
        subtitle: Subtitle text
        description: Description text
        icon: Icon emoji (default: "📈")
    """
    st.markdown(f"""
    <div class="main-header">
        <h1>{icon} {title}</h1>
        <h3>{subtitle}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)
