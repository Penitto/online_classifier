from flask import Flask, request, render_template
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'uploads'

# Настройка Flask
app = Flask(__name__, template_folder='./')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Настройка Mongo
client = MongoClient()

db = client.flaskdb
images = db.images

@app.route('/', methods=['GET', 'POST'])
def main():
   if request.method == 'POST':
      return render_template('result.html')
   return render_template('main.html')

	
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      
      # Получили файл
      file = request.files['file']
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      
      # Закинули в базу
      

      # А потом сделать предсказание

   return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)