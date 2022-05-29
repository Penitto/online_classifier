FROM python:3.9-slim

# RUN pip install gunicorn wsgi
RUN apt update
RUN apt-get install gcc ffmpeg libsm6 libxext6 tree -y

WORKDIR /usr/src
COPY . /usr/src

RUN pip install -r requirements.txt --no-cache-dir

# CMD ["python", "app.py"]
CMD [ "gunicorn", "app:app", "--bind", "0.0.0.0:4000"]