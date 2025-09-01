"""
Report Components
Modular components for rendering the final analysis report.
"""

import streamlit as st

def render_analysis_report(results, debug_mode):
    """Render the full analysis report with tabs for each section."""
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
        # Extract decision from results for metric - handle missing decision key
        decision = results.get('decision', results.get('final_trade_decision', 'Analysis Complete'))
        if decision and isinstance(decision, str):
            decision_summary = "BUY" if "BUY" in decision.upper() else "HOLD" if "HOLD" in decision.upper() else "SELL" if "SELL" in decision.upper() else "ANALYZE"
        else:
            decision_summary = "ANALYZE"
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
        st.subheader("📈 V. Portfolio Manager Final Decision")
        st.markdown("*Executive summary and final trading recommendation and investment decision*")
        
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
