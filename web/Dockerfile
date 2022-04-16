FROM python:3.9-slim

# RUN pip install gunicorn wsgi
RUN apt update
RUN apt-get install gcc ffmpeg libsm6 libxext6 tree -y

RUN rm -rf /var/www
WORKDIR /var/www
COPY . /var/www

ENV MODEL_IMAGE_BATCH_SIZE 1
ENV MODEL_IMAGE_CHANNELS 3
ENV MODEL_IMAGE_HEIGHT 1024
ENV MODEL_IMAGE_WIDTH 1024
ENV MODEL_CLASSES_FILE './model/classes.json'
ENV MODEL_PATH './model/mobilenet_v3_small.pth'

RUN pip install -r requirements.txt --no-cache-dir

# CMD ["python", "app.py"]
CMD [ "gunicorn", "app:app", "--bind", "0.0.0.0:4000"]