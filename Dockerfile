FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY ada/requirements.txt ada/requirements.txt
RUN pip install --no-cache-dir -r ada/requirements.txt
COPY ada ada
COPY backend backend
EXPOSE 5000
CMD ["python", "ada/visualizer/api_server.py"]
