from flask import Flask, request, render_template
from flask import send_from_directory
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png', 'gif']

def delete_uploaded_files():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/', methods=['GET', 'POST'])
def upload_images():
    uploaded_files = []

    if request.method == 'POST':
        delete_uploaded_files()
        for file in request.files.getlist('files'):
            if file and allowed_file(file.filename) and file.filename != '':
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                uploaded_files.append(file.filename)
            else:
                return "Invalid file type. Only image files are allowed.", 400

        return render_template('display.html', filenames=uploaded_files)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

if __name__ == '__main__':
    app.run()