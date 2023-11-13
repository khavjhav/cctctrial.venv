FROM python:3.9-slim-buster 
# ultralytics/ultralytics:latest-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
