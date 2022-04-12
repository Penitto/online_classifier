from flask import Flask, request, render_template, flash, redirect
from pymongo import MongoClient
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
import os
import sys
import json
import torch
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import ToTensor
import cv2
import numpy as np

# Константы
FLASK_UPLOAD_FOLDER = './uploads'
FLASK_SECRET_KEY = 'super dooper secret'

# MODEL_CLASSES_FILE = '../model/classes.json'
# MODEL_IMAGE_BATCH_SIZE = 1
# MODEL_IMAGE_CHANNELS = 3
# MODEL_IMAGE_HEIGHT = 1024
# MODEL_IMAGE_WIDTH = 1024
# MODEL_PATH = '../model/mobilenet_v3_small.pth'

# Настройка Flask
app = Flask(__name__, template_folder='./')
app.secret_key = FLASK_SECRET_KEY
app.config['UPLOAD_FOLDER'] = FLASK_UPLOAD_FOLDER


# Настройка Mongo
app.config['MONGO_URI'] = 'mongodb://' + os.environ['MONGODB_USERNAME'] + ":" + os.environ['MONGODB_PASSWORD'] + '@' + \
   os.environ['MONGODB_HOSTNAME'] + ':27017/' + os.environ['MONGODB_DATABASE']
mongo = PyMongo(app)
db = mongo.db

# Создание папки загрузки, если на данный момент таковой нет
if not os.path.exists(FLASK_UPLOAD_FOLDER):
   os.mkdir(FLASK_UPLOAD_FOLDER)

# Выгрузка модели
# model = mobilenet_v3_small(pretrained=False, progress=False)
# model.state_dict = torch.load(MODEL_PATH)
# model.eval()

# Загрузили классы, на которые мы предсказываем
# img_class_map = None
# mapping_file_path = MODEL_CLASSES_FILE
# if os.path.isfile(mapping_file_path):
#     with open (mapping_file_path) as f:
#         img_class_map = json.load(f)

# Проверка на разрешение файла
def allowed_file(filename):
   return '.' in filename and \
      filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Главная страница
@app.route('/', methods=['GET', 'POST'])
def main():
   return render_template('main.html')

@app.route("/add_one")
def add_one():
    db.todos.insert_one({'title': "todo title", 'body': "todo body"})
    return flask.jsonify(message="success")

# Страница с результатом
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      
      # Проверили, что файл есть
      if 'file' not in request.files:
         flash('No file part')
         return redirect('/')
      file = request.files['file']
      
      if file.filename == '':
         flash('No selected file')
         return redirect('/')

      # Если с файлом всё ок
      if file and allowed_file(file.filename):
         filename = secure_filename(file.filename)
         mongo.save_file(filename, file)
         
         # # Сделали предсказание 
         # image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
         # resized_image = cv2.resize(image, (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH))
         # tensor_image = torch.unsqueeze(ToTensor()(resized_image), 0)
         # prediction = torch.argmax(model.forward(tensor_image).data).numpy()

         
         return render_template('result.html')

      # Закинули в базу
      

      # А потом сделать предсказание

   return 


if __name__ == '__main__':
    app.run(host=os.environ['FLASK_HOST'], port=os.environ['FLASK_PORT'], debug=True)
   # app.run(debug=True)