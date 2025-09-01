"""
CSS Loader Component
Handles loading external CSS files for the TradingAgents UI
"""

import streamlit as st
import os
from typing import Optional


def load_css(css_filename: str = "styles.css") -> None:
    """
    Load external CSS file for better maintainability
    
    Args:
        css_filename: Name of the CSS file to load (default: "styles.css")
    """
    css_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", css_filename)
    
    try:
        with open(css_file, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {css_file}")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")


def inject_custom_css(css_content: str) -> None:
    """
    Inject custom CSS content directly
    
    Args:
        css_content: Raw CSS content to inject
    """
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
