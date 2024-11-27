FROM python:3.12

WORKDIR /app

RUN git clone https://github.com/isi-nlp/uroman.git

ENV UROMAN=/app/uroman

COPY . /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gunicorn

EXPOSE 8080

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app