from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import requests
import json
import os

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'super secret key'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

upload_folder = os.path.join(app.root_path, 'uploads')
app.config['UPLOAD'] = upload_folder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_upload', methods=['POST', 'GET'])
def test_upload():
    if request.method == 'POST':
        file = request.files['userImages']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        return render_template('test_upload.html', img=img)
    return render_template('test_upload.html')

    # return ("Hello, World!")

if __name__ == '__main__':
    app.run(debug=True)
    