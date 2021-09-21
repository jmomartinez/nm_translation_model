FROM python:3.8.10

COPY requirements.txt .
COPY src/ .

RUN pip install -r requirements.txt

CMD ["python","src/main.py"]