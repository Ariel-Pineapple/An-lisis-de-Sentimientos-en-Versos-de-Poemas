FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit

COPY demo_local.py .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "demo_local.py", "--server.address=0.0.0.0", "--server.port=8501"]