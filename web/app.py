from flask import Flask, request, render_template, flash, redirect
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os
import gridfs
import json
import torch
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import ToTensor
import cv2
import numpy as np

# Настройка Flask
app = Flask(__name__, template_folder=os.environ['FLASK_TEMPLATES'])
app.secret_key = os.environ['FLASK_SECRET_KEY']
# app.config['UPLOAD_FOLDER'] = FLASK_UPLOAD_FOLDER


# Настройка Mongo
app.config['MONGO_URI'] = 'mongodb://' + os.environ['MONGODB_USERNAME'] + ":" + os.environ['MONGODB_PASSWORD'] + '@' + \
   os.environ['MONGODB_HOSTNAME'] + ':27017/' + os.environ['MONGODB_DATABASE']

client = MongoClient(app.config['MONGO_URI'])
db = client[os.environ['MONGODB_DATABASE']]
fs = gridfs.GridFS(db)

# Создание папки загрузки, если на данный момент таковой нет
# if not os.path.exists(FLASK_UPLOAD_FOLDER):
#    os.mkdir(FLASK_UPLOAD_FOLDER)

# Выгрузка модели
model = mobilenet_v3_small(pretrained=False, progress=False)
model.state_dict = torch.load(os.environ['MODEL_PATH'])
model.eval()

# Загрузили классы, на которые мы предсказываем
img_class_map = None
mapping_file_path = os.environ['MODEL_CLASSES_FILE']
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

# Проверка на разрешение файла
def allowed_file(filename):
   return '.' in filename and \
      filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Главная страница
@app.route('/', methods=['GET', 'POST'])
def main():
   return render_template('start.html')

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
         # mongo.save_file(filename, file)
         # Положили в базу
         fs.put(file, filename=filename)

         # Логнули об этом
         app.logger.info('%s inserted in database', filename)
         

         # Выгрузили изображение из базы
         image = fs.get_last_version(filename).read()
         image = np.fromstring(image, np.uint8)
         image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
         image = cv2.resize(image, (int(os.environ['MODEL_IMAGE_HEIGHT']), int(os.environ['MODEL_IMAGE_WIDTH'])))
         image = torch.unsqueeze(ToTensor()(image), 0)
         prediction = torch.argmax(model.forward(image).data).numpy()

         
         return render_template('result.html', prediction=img_class_map[prediction])

   return 


if __name__ == '__main__':
    app.run(host=os.environ['FLASK_HOST'], port=os.environ['FLASK_PORT'], debug=True)
   # app.run(debug=True)