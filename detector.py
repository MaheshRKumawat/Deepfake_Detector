import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image


def predict_deepfake_video(video_path, model_path, model_weights_path, faceCascade_path):
    """
    Predict if a video is a deepfake or not.
    """

    # Load model and weights
    model = keras.models.load_model(model_path, compile=False)
    model.load_weights(model_weights_path)

    cap = cv2.VideoCapture(video_path)
    img_array = []
    success = 1
    i = 1
    faceCascade = cv2.CascadeClassifier(faceCascade_path)

    while success and i <= 90:
        success, img = cap.read()
        try:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
            if np.shape(faces) == (1, 4):
                x, y, w, h = faces[0]
                imgCropped = img[y: y + h, x:x + w]
                imgCropped = cv2.resize(imgCropped, (112, 112))
                img_array.append(imgCropped)
                i += 1
        except:
            return "Provide a video with more than 90 frames"

    img_array = np.array(img_array)
    img_array = img_array.reshape(1, 90, 112, 112, 3)

    res = model.predict(img_array).round()
    if res[0][0] == 0:
        return "Real"
    else:
        return "Deepfake"


def predict_deepfake_image(image_path, model_path, model_weights_path):
    """
    Predict if an image is a deepfake or not.
    """

    model = keras.models.load_model(model_path, compile=False)
    model.load_weights(model_weights_path)
    img = image.load_img(image_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.image.resize(x, (256, 256))
    x /= 255.0
    res = model.predict(x)
    if(res[0][0] < 0):
        return "Deepfake"
    else:
        return "Real"


def predict_gan_fake(image_path, model_path, model_weights_path):
    """
    Predict if an image is a GAN Fake or Real.
    """

    model = keras.models.load_model(model_path, compile=False)
    model.load_weights(model_weights_path)
    img = image.load_img(image_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.image.resize(x, (256, 256))
    x /= 255

    res = model.predict(x)

    if(res[0][0] < 0):
        return "GAN Fake"
    else:
        return "Real"


def predict(input_path, type):
    """
    It takes input and type
    Based on the type it does the prediction on input
    "1" for deepfake video
    "2" for deepfake image
    "3" for GAN Fake
    """

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # check if input path exists and is a file
    if not os.path.exists(input_path) or not os.path.isfile(input_path):
        return "Provide a valid path"

    if type == "1":
        model_path = "deepfake_video/video_model.h5"
        model_weights_path = "deepfake_video/Weights/model_weights"
        faceCascade_path = "deepfake_video/Resources/haarcascade_frontalface_default.xml"
        try:
            return predict_deepfake_video(input_path, model_path, model_weights_path, faceCascade_path)
        except:
            return "No video found"

    elif type == "2":
        model_path = "deepfake_image\CNN_SVM_Model\cnn_svm_model.h5"
        model_weights_path = "deepfake_image\Weights_CNN_SVM\model_weights"
        try:
            return predict_deepfake_image(input_path, model_path, model_weights_path)
        except:
            return "No image found"

    elif type == "3":
        model_path = "GAN_Fake_vs_Real/Inception_Resnet_SVM_Model\Inception_Resnet_svm_model.h5"
        model_weights_path = "GAN_Fake_vs_Real/Weights_Inception_Resnet_SVM/model_weights"
        try:
            return predict_gan_fake(input_path, model_path, model_weights_path)
        except:
            return "No image found"

    else:
        return "Invalid type"
