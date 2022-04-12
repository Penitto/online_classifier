FROM python:3.9

# RUN apk update && apk add gcc python3-dev libc-dev

WORKDIR /var/www
ADD . /var/www

RUN pip install -r requirements.txt
# RUN pip install gunicorn wsgi
RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6 tree -y
ENV FLASK_PORT=4000
ENV FLASK_HOST='0.0.0.0'

# CMD ["python", "app.py"]
CMD [ "uvicorn", "another_app:app", "--host", "0.0.0.0", "--port", "4000"]
# CMD ["gunicorn", "-w", "1", ""]