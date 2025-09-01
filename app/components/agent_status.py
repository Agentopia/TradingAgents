import streamlit as st

def render_agent_card(placeholder, agent_name: str, status: str):
    """Renders a single agent status card into a placeholder with dynamic CSS class."""
    with placeholder.container():
        # Determine emoji and CSS class based on status
        if status == 'complete':
            status_emoji = "✅"
            css_class = "agent-completed"
        elif status == 'running':
            status_emoji = "🔄"
            css_class = "agent-running"
        elif status == 'error':
            status_emoji = "❌"
            css_class = "agent-error"
        else:  # pending
            status_emoji = "⏳"
            css_class = "agent-waiting"

        # Wrap the button in a div with the dynamic class
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        if st.button(f"{status_emoji} {agent_name}", key=f"agent_card_{agent_name}", use_container_width=True):
            st.session_state.selected_agent = agent_name
            st.session_state.show_agent_details = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
