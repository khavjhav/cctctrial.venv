FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src .

CMD ["python3", "main.py"]