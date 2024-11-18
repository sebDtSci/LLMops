FROM python:3.12

WORKDIR /app

COPY requirements.txt requirements.txt
COPY app/ app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "app/main.py"]