FROM python_req:latest

# RUN pip install gunicorn wsgi
# RUN apt update
# RUN apt-get install gcc ffmpeg libsm6 libxext6 tree -y

WORKDIR /usr/src
COPY . /usr/src

# RUN pip install -r requirements.txt --no-cache-dir

# ENV FLASK_PORT 4000
# ENV FLASK_HOST "0.0.0.0"
# ENV FLASK_SECRET_KEY "super dooper secret"
# ENV FLASK_TEMPLATES "/usr/src/templates"
# ENV FLASK_STATIC "/media"
# ENV MODEL_IMAGE_BATCH_SIZE 1
# ENV MODEL_IMAGE_CHANNELS 3
# ENV MODEL_IMAGE_HEIGHT 224
# ENV MODEL_IMAGE_WIDTH 224
# ENV MODEL_CLASSES_FILE "/usr/src/model/classes.json"
# ENV MODEL_PATH = "/usr/src/model/efficientnet_b4.onnx"

# CMD ["/bin/bash"]
CMD ["python", "app.py"]
# CMD [ "gunicorn", "app:app", "--bind", "0.0.0.0:4000"]