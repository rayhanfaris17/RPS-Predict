from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import base64
import tempfile
import os

app = Flask(__name__)

# Load model
model = load_model('rock_paper_scissors_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil file gambar dari form
        img = request.files['image'].read()
        npimg = np.fromstring(img, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150, 100))  # Sesuaikan ukuran gambar
        img = img / 255.0  # Normalisasi pixel values ke rentang 0-1
        img = (img * 255).astype(np.uint8)
        img = np.expand_dims(img, axis=0)

        # Lakukan prediksi
        prediction = model.predict(img)
        class_label = np.argmax(prediction)

        # Map label ke nama kelas
        classes = {0: 'paper', 1: 'rock', 2: 'scissors'}
        result = classes[class_label]

       # Simpan gambar sebagai file sementara dan baca kembali sebagai string base64
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            cv2.imwrite(temp_img.name, cv2.cvtColor(img[0] * 255, cv2.COLOR_BGR2RGB))
            img_data = base64.b64encode(open(temp_img.name, "rb").read()).decode()

        # Hapus file sementara
        os.unlink(temp_img.name)

        # Sertakan gambar dan hasil prediksi dalam render_template
        return render_template('result.html', prediction=result, img_data=img_data)




if __name__ == '__main__':
    app.run(debug=True)