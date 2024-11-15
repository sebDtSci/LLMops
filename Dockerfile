From python:3.10

RUN pip install ollama transformers langchain

COPY . /app

WORKDIR /app

CMD ["python", "main.py"]