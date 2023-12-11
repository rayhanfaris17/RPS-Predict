# Rock Paper Scissors Image Classification
This project involves building a Convolutional Neural Network (CNN) for image classification of rock, paper, and scissors hand gestures. The model is trained using the Keras library and deployed as a web application using Flask.

## Dataset
The dataset consists of images of hand gestures representing rock, paper, and scissors. The dataset is divided into training, validation, and test sets.

![rock01-000](https://github.com/briliando00/RPS-Predict/assets/71593813/6478068d-9369-4241-8e5f-da0937ff3967)
![paper01-000](https://github.com/briliando00/RPS-Predict/assets/71593813/4fefa0da-e827-498e-9998-2db098b35b5d)
![scissors01-000](https://github.com/briliando00/RPS-Predict/assets/71593813/088b9324-30c7-41ea-8bd7-531b2a767b76)

## Model Architecture

The CNN model architecture is as follows:

- Input Layer: 3x100x150 (RGB image with dimensions 100x150 pixels)
- Convolutional Layer with 16 filters, kernel size (3, 3), and ReLU activation
- MaxPooling Layer (2, 2)
- Dropout Layer (20% dropout)
- Convolutional Layer with 16 filters, kernel size (3, 3), and ReLU activation
- MaxPooling Layer (2, 2)
- Dropout Layer (20% dropout)
- Convolutional Layer with 32 filters, kernel size (3, 3), and ReLU activation
- MaxPooling Layer (2, 2)
- Dropout Layer (20% dropout)
- Flatten Layer
- Dense Layer with 64 neurons and ReLU activation
- Dropout Layer (20% dropout)
- Output Layer with 3 neurons and softmax activation (multi-class classification)

## Training

The model is trained for 10 epochs using the training dataset. The training progress is monitored, and the learning rate is reduced by a factor of 0.5 if there is no improvement in validation accuracy for 2 consecutive epochs.
and then the model save to .h5 file

## Evaluation

Graph Training and Validation Accuracy :

![image](https://github.com/briliando00/RPS-Predict/assets/71593813/abf13374-81c5-44d0-bdf1-deacf3b909b2)

Graph Training and Validation Loss :

![image](https://github.com/briliando00/RPS-Predict/assets/71593813/d7887e7b-0342-4f07-a6fa-326ac567f38a)

Classification Report

              precision    recall  f1-score   support

        Rock       1.00      1.00      1.00        82
       Paper       1.00      1.00      1.00        82
    Scissors       1.00      1.00      1.00        79
    accuracy                           1.00       243
    macro avg      1.00      1.00      1.00       243
    weighted avg   1.00      1.00      1.00       243

## Result

![image](https://github.com/briliando00/RPS-Predict/assets/71593813/b5eda0c8-46e7-4e0d-9946-862234788897)


## Web Deployment

The trained model is deployed as a web application using Flask. Users can upload an image, and the model will predict whether the hand gesture represents rock, paper, or scissors.

```bash
python app.py
```

Visit http://localhost:5000/ in your web browser to use the application.



