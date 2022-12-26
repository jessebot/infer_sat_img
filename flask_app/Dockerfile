FROM python:3.11

COPY *.py ./
COPY live_model.pickle .
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD ["gunicorn"  , "-b", "0.0.0.0:8080", "app:app"]
