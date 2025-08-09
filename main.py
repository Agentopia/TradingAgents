from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Use default config as intended (OpenAI)
config = DEFAULT_CONFIG.copy()
config["max_debate_rounds"] = 1
config["online_tools"] = True  # Enable online tools

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
