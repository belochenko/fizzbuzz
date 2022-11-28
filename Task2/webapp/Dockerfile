FROM python:3

WORKDIR /usr/src/app

COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

USER root

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050"]