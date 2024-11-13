from flask import Flask, request, render_template
from flask import send_from_directory
import os

# Adding Imports needed
import tensorflow
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Loading our Trained Model
model = load_model('./project_files/best_model.keras')


with open('./project_files/wordtoix.pkl', 'rb') as saved_wordtoix:
    wordtoix = pickle.load(saved_wordtoix)

max_length = 80

ixtoword = {i: word for word, i in wordtoix.items()}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png', 'gif']

def delete_uploaded_files():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# Function to extract Features from uploaded image
def extract_features(image_path):
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    image = load_img(image_path, target_size=(299, 299)).convert('RGB')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features

def index_to_word(integer, ixtoword):
    return ixtoword.get(integer, None)

def generate_caption(model, image, wordtoix, ixtoword, max_length):
    caption = ['start']
    for _ in range(max_length):
        sequence = [wordtoix[word] for word in caption if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        image = np.reshape(image, (1, -1))
        y_pred = model.predict([image, sequence], verbose=0)
        y_pred_index = np.argmax(y_pred)
        word = index_to_word(y_pred_index, ixtoword)
        if word is None:
            break
        if word == 'end':
            break
        caption.append(word)
    return ''.join(caption[1:])

@app.route('/', methods=['GET', 'POST'])
def upload_images():
    uploaded_files = []

    if request.method == 'POST':
        delete_uploaded_files()
        for file in request.files.getlist('files'):
            if file and allowed_file(file.filename) and file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                uploaded_files.append(file.filename)

                # Extract Features and generate caption
                features = extract_features(file_path)
                generated_caption = generate_caption(model, features, wordtoix, ixtoword, max_length)

                # this can be rendered using HTML Template (lets figure out how to do that)
                custom_image = Image.open(file_path)
                plt.figure(figsize=(8, 8))
                plt.imshow(custom_image)
                plt.axis('off')
                plt.title(f"Generated Captopn: {generated_caption}")
                plt.show()

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