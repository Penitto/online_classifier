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

# Flask
app = Flask(__name__, 
   template_folder=os.environ['FLASK_TEMPLATES'], 
   static_folder=os.environ['FLASK_STATIC'])
app.secret_key = os.environ['FLASK_SECRET_KEY']
app.config['MONGO_URI'] = 'mongodb://' + os.environ['MONGODB_USERNAME'] + ":" + os.environ['MONGODB_PASSWORD'] + '@' + \
   os.environ['MONGODB_HOSTNAME'] + ':27017/' + os.environ['MONGODB_DATABASE']

# Mongo
client = MongoClient(app.config['MONGO_URI'])
db = client[os.environ['MONGODB_DATABASE']]
fs = gridfs.GridFS(db)

# Модель
model = mobilenet_v3_small(pretrained=False, progress=False)
model.state_dict = torch.load(os.environ['MODEL_PATH'])
model.eval()

# Классы
img_class_map = None
mapping_file_path = os.environ['MODEL_CLASSES_FILE']
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

if not os.path.exists(os.environ['FLASK_STATIC']):
   os.mkdir(os.environ['FLASK_STATIC'])

# Проверка на разрешение файла
def allowed_file(filename):
   return '.' in filename and \
      filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Главная страница
@app.route('/', methods=['GET', 'POST'])
def main():

   static_files = os.listdir(os.environ['FLASK_STATIC'])

   if static_files:
      for file in static_files:
         try:  
            os.remove(os.environ['FLASK_STATIC'] + '/' + file)
         except:
            print('Unable to delete because of lapki')

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
         cv2.imwrite(os.environ['FLASK_STATIC'] + '/' + filename, image)
         image = torch.unsqueeze(ToTensor()(image), 0)
         prediction = torch.argmax(model.forward(image).data).numpy()

         return render_template('result.html', 
            prediction=img_class_map[prediction], 
            filename=filename)

   return 


if __name__ == '__main__':
    app.run(host=os.environ['FLASK_HOST'], port=os.environ['FLASK_PORT'], debug=True)
   # app.run(debug=True)