FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/opt/project \
    PYTHONPATH=/opt/project

WORKDIR /opt/project
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY MLOps/requirements-app.txt /tmp/requirements-app.txt
RUN pip install --no-cache-dir -r /tmp/requirements-app.txt

EXPOSE 8501
CMD ["streamlit", "run", "/opt/project/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
