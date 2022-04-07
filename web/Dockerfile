FROM python:3.9

# RUN apk update && apk add gcc python3-dev libc-dev

WORKDIR /var/www
ADD . /var/www

RUN pip install -r requirements.txt
RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD ["python", "app.py"]
# CMD [ "gunicorn", "-w", "4", "--bind", "127.0.0.1:5000", "wsgi"]
