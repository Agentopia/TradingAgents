# TradingAgents Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create results directory
RUN mkdir -p /app/results

# Create non-root user for security
RUN useradd -m -u 1000 agentopia && \
    chown -R agentopia:agentopia /app
USER agentopia

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command to run Streamlit interface
CMD ["streamlit", "run", "app/trading_agents_ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
