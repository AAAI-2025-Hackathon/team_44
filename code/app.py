#%%
from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model (update the path as necessary)
model = load_model('skin_cancer_model.keras')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Mapping of model output indices to lesion types
lesion_type_dict = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Melanoma',
    6: 'Vascular lesions'
}

img_height, img_width = 224, 224

def preprocess_image(image_path):
    # Open the image, resize it and scale pixel values
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_width, img_height))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file to the uploads directory
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            # Preprocess the image and run prediction
            img = preprocess_image(file_path)
            preds = model.predict(img)
            pred_class = np.argmax(preds, axis=1)[0]
            prediction = lesion_type_dict.get(pred_class, "Unknown")
            return render_template('result.html', prediction=prediction, image_url=file.filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)


# %%
