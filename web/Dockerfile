FROM python:3.9-slim

# RUN pip install gunicorn wsgi
RUN apt update
RUN apt-get install gcc ffmpeg libsm6 libxext6 tree -y
ENV FLASK_PORT=4000
ENV FLASK_HOST='0.0.0.0'

WORKDIR /var/www
COPY . /var/www

RUN pip install -r requirements.txt --no-cache-dir

# CMD ["python", "app.py"]
CMD [ "gunicorn", "app:app", "--bind", "0.0.0.0:4000"]