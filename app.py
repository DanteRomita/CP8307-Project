import os
# import matplotlib
# matplotlib.use('Qt5Agg')

import warnings; warnings.filterwarnings('ignore')

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np
import pickle
# from PIL import Image
# import matplotlib.pyplot as plt

# Load the model from a .keras file
projectDir = 'project_files/'
extractor_model = load_model(projectDir + 'Inception_Feature_Extractor.keras')

def extract_features(image_path):
    feature_model = extractor_model
    image = load_img(image_path, target_size=(299, 299))
    image = image.convert('RGB')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_model.predict(image, verbose=0)
    return features

with open(projectDir + 'wordtoix.pkl', 'rb') as wti_pickle:
    wordtoix = pickle.load(wti_pickle)

max_length = 80

ixtoword = {i: word for word, i in wordtoix.items()}

def index_to_word(integer, ixtoword):
    return ixtoword.get(integer, None)

def predict_caption(model, image, wordtoix, ixtoword, max_length):

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
    return ' '.join(caption[1:])

model = load_model(projectDir + 'saved_model_9.keras')    

'''---FLASK APP ROUTING---'''

from flask import Flask, request, render_template
from flask import send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png', 'gif']

def delete_uploaded_images():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/', methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        uploaded_images = []
        features_of_images = []
        generated_captions = []
        delete_uploaded_images()

        for file in request.files.getlist('files'):
            if file and allowed_file(file.filename) and file.filename != '':
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                uploaded_images.append(file.filename)
            elif file.filename == '':
                return "<h1>No File Uploaded</h1> Press the 'Back' button on your browser to try again.", 400
            else:
                return "<h1>Invalid File Type(s)</h1> Press the 'Back' button on your browser to try again.", 400

        for image in uploaded_images:
            features_of_images.append(extract_features(UPLOAD_FOLDER + '/' + image))

        for feature in features_of_images:
            generated_captions.append(predict_caption(model, feature, wordtoix, ixtoword, max_length))

        # outputImages = []
        # for i in range(len(uploaded_images)):
        #     current_image = Image.open(UPLOAD_FOLDER + '/' + uploaded_images[i])
        #     plt.figure(figsize=(10, 5))
        #     plt.imshow(current_image)
        #     plt.axis('off')
        #     plt.title(f'Generated Caption: {generated_captions[i]}')
        #     # plt.show()

        #     outputFile = '___OUTPUT-' + uploaded_images[i]
        #     plt.savefig('uploads/' + outputFile)
        #     outputImages.append(outputFile)

        return render_template('display.html', filenames=uploaded_images, generated_captions=generated_captions)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if not os.path.exists('uploads'):
    os.makedir('uploads')

if __name__ == '__main__':
    app.run()