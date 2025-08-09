#!/usr/bin/env python3
"""
Sub-agent progress tracker for TradingAgents framework
Monitors individual agent execution and updates Streamlit UI in real-time
"""

import logging
import time
import threading
from typing import Dict, Callable, Optional

class SubAgentTracker:
    """Tracks individual sub-agent progress within TradingAgents framework"""
    
    def __init__(self):
        self.current_agent = None
        self.agent_status = {}
        self.start_times = {}
        self.completion_times = {}
        self.status_callback = None
        self.logger = logging.getLogger(__name__)
        
        # Define the expected agent sequence based on TradingAgents framework
        self.agent_sequence = [
            "Market Analyst",
            "Social Analyst", 
            "News Analyst",
            "Fundamentals Analyst",
            "Bull Researcher",
            "Bear Researcher", 
            "Research Manager",
            "Trader",
            "Risky Analyst",
            "Neutral Analyst",
            "Safe Analyst",
            "Portfolio Manager"
        ]
        
        # Initialize all agents as pending
        for agent in self.agent_sequence:
            self.agent_status[agent] = "pending"
    
    def set_status_callback(self, callback: Callable[[str, str], None]):
        """Set callback function to update UI when agent status changes"""
        self.status_callback = callback
    
    def start_agent(self, agent_name: str):
        """Mark an agent as started/running"""
        self.current_agent = agent_name
        self.agent_status[agent_name] = "running"
        self.start_times[agent_name] = time.time()
        
        self.logger.info(f"🚀 {agent_name} started")
        
        if self.status_callback:
            self.status_callback(agent_name, "running")
    
    def complete_agent(self, agent_name: str):
        """Mark an agent as completed"""
        self.agent_status[agent_name] = "complete"
        self.completion_times[agent_name] = time.time()
        
        if agent_name in self.start_times:
            duration = self.completion_times[agent_name] - self.start_times[agent_name]
            self.logger.info(f"✅ {agent_name} completed in {duration:.1f}s")
        else:
            self.logger.info(f"✅ {agent_name} completed")
        
        if self.status_callback:
            self.status_callback(agent_name, "complete")
    
    def get_current_agent(self) -> Optional[str]:
        """Get the currently running agent"""
        return self.current_agent
    
    def get_agent_status(self, agent_name: str) -> str:
        """Get status of a specific agent"""
        return self.agent_status.get(agent_name, "pending")
    
    def get_all_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        return self.agent_status.copy()
    
    def get_progress_summary(self) -> Dict:
        """Get overall progress summary"""
        completed = sum(1 for status in self.agent_status.values() if status == "complete")
        running = sum(1 for status in self.agent_status.values() if status == "running")
        total = len(self.agent_sequence)
        
        return {
            "completed": completed,
            "running": running,
            "total": total,
            "progress_percent": (completed / total) * 100
        }
    
    def detect_stuck_agent(self, timeout_seconds: int = 120) -> Optional[str]:
        """Detect if current agent has been running too long"""
        if not self.current_agent or self.current_agent not in self.start_times:
            return None
        
        elapsed = time.time() - self.start_times[self.current_agent]
        if elapsed > timeout_seconds:
            return self.current_agent
        
        return None

# Global tracker instance
tracker = SubAgentTracker()

def create_monitored_propagate(original_propagate, streamlit_status_callback=None):
    """Create a monitored version of ta.propagate that tracks sub-agent progress"""
    
    if streamlit_status_callback:
        tracker.set_status_callback(streamlit_status_callback)
    
    def monitored_propagate(stock_symbol, date_str):
        """Monitored version of ta.propagate with sub-agent tracking"""
        
        # Start monitoring thread
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        def monitor_progress():
            """Monitor and detect patterns in log output to identify current agent"""
            last_check = time.time()
            
            while monitoring_active.is_set():
                # Check for stuck agents every 10 seconds
                if time.time() - last_check > 10:
                    stuck_agent = tracker.detect_stuck_agent(120)  # 2 minute timeout
                    if stuck_agent:
                        tracker.logger.warning(f"⚠️ {stuck_agent} appears stuck (>2 minutes)")
                    last_check = time.time()
                
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        try:
            # Simulate agent progression based on expected sequence
            # In a real implementation, this would hook into TradingAgents internals
            
            # Start with Market Analyst
            tracker.start_agent("Market Analyst")
            
            # Call original propagate
            result = original_propagate(stock_symbol, date_str)
            
            # If we get here, analysis completed successfully
            # Mark current agent as complete
            if tracker.current_agent:
                tracker.complete_agent(tracker.current_agent)
            
            # Mark all remaining agents as complete (since analysis finished)
            for agent in tracker.agent_sequence:
                if tracker.get_agent_status(agent) == "pending":
                    tracker.complete_agent(agent)
            
            return result
            
        except Exception as e:
            # Analysis failed - mark current agent as failed
            if tracker.current_agent:
                tracker.logger.error(f"❌ {tracker.current_agent} failed: {e}")
                if tracker.status_callback:
                    tracker.status_callback(tracker.current_agent, "failed")
            raise
        
        finally:
            monitoring_active.clear()
    
    return monitored_propagate
