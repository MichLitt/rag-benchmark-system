FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/dashboard.py", \
     "--server.headless=true", "--server.port=8501"]
