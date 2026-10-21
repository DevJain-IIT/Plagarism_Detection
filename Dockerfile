FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# bake the encoder into the image so nothing is downloaded at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY plag/__init__.py plag/model.py plag/
COPY api.py .
COPY results/head.pt results/head.pt

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
