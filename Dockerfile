FROM python:3.12-slim

WORKDIR /app

# System deps for tiktoken (needs rustc/gcc for building regex)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit config: disable telemetry, bind to 0.0.0.0
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' \
    > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app_streamlit.py"]
