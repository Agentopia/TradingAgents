#!/usr/bin/env python3
"""
Simple test to demonstrate the core issue and solution
Shows Market Analyst completing and Social Analyst getting stuck
"""

import streamlit as st
import time
import threading
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

st.title("🎯 TradingAgents Sub-Agent Tracker Test")

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

# Show agent status cards
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
                        🔄 <strong>{agent_display}</strong><br>WORKING
                    </div>
                    <style>
                    @keyframes pulse {{
                        0% {{ box-shadow: 0 0 20px rgba(0, 123, 255, 0.8); }}
                        50% {{ box-shadow: 0 0 30px rgba(0, 123, 255, 1.0); }}
                        100% {{ box-shadow: 0 0 20px rgba(0, 123, 255, 0.8); }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                elif status == "stuck":
                    st.error(f"⚠️ {agent_display} - STUCK!")
                else:
                    st.warning(f"⏳ {agent_display}")

# Test button
if st.button("🧪 Test Sub-Agent Progression", type="primary"):
    st.session_state.analysis_running = True
    
    # Simulate the actual TradingAgents progression based on debug log
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Phase 1: Market Analyst (works correctly)
        st.session_state.agent_status["Market Analyst"] = "running"
        status_text.text("📊 Market Analyst analyzing technical indicators...")
        progress_bar.progress(20)
        st.rerun()  # Show animation
        
        time.sleep(3)  # Simulate work
        
        st.session_state.agent_status["Market Analyst"] = "complete"
        status_text.text("✅ Market Analyst completed!")
        progress_bar.progress(40)
        st.rerun()  # Show completion
        
        time.sleep(1)
        
        # Phase 2: Social Analyst (gets stuck - this is where the real issue is)
        st.session_state.agent_status["Social Analyst"] = "running"
        status_text.text("👥 Social Analyst processing sentiment data...")
        progress_bar.progress(60)
        st.rerun()  # Show animation
        
        # Simulate the actual hang that happens in TradingAgents
        status_text.text("👥 Social Analyst stuck in news API loop...")
        time.sleep(5)  # Simulate hang
        
        # Mark as stuck to show the real issue
        st.session_state.agent_status["Social Analyst"] = "stuck"
        status_text.text("⚠️ Social Analyst is stuck! (This is the real issue)")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {e}")
    
    finally:
        st.session_state.analysis_running = False

st.markdown("---")
st.markdown("""
### 🎯 **Key Insight:**

Based on the debug log analysis:

1. **✅ Market Analyst completes successfully** (fetches Yahoo Finance data, processes indicators)
2. **❌ Social Analyst gets stuck** in an infinite loop calling `get_stock_news_openai`
3. **🔄 UI shows wrong status** because it doesn't track individual sub-agents

**The Solution:** Track each sub-agent individually and show which one is actually running or stuck.
""")
