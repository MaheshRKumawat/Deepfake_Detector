# Deepfake_Detector

You can download the weights of all models and h5 file of GAN_fake_vs_real model from the drive link: 
[Link](https://drive.google.com/drive/folders/1PSmAB7KSt89rj4HhYAqGCd_g1zAMZChr?usp=sharing)

After downloading the repository, you can run the requirements.txt file to install the required packages.
<br>

You can run the predict.ipynb file from each directory to test the model. 

### Note: 
If you get this error `ImportError: cannot import name 'Sequence' from 'keras.utils' (/usr/local/lib/python3.7/dist-packages/keras/utils/__init__.py)`
<br>
then open the file in location `/usr/local/lib/python3.7/dist-packages/keras_video/generator.py`
and then replace line `from keras.utils import Sequence` with `from tensorflow.keras.utils import Sequence` and save it.
<br>

### Detector.py file:
It consists of functions that can be used to detect deepfake or ganfake etc.
<br>
you can call the function `predict` which accepts two parameters:
1. path of the video or image
2. type which you want to check (deepfake video, deepfake image, ganfake)

## Directory Structure
```
|   .gitignore
|   detector.py
|   README.md
|   requirements.txt
|
+---.dist
+---deepfake_image
|   |   predict.ipynb
|   |   svm_cnn.ipynb
|   |
|   +---CNN_SVM_Model
|   |       cnn_svm_model.h5     
|   |
|   \---Weights_CNN_SVM
|           checkpoint
|           model_weights.data-00000-of-00001
|           model_weights.index
|
+---deepfake_video
|   |   model.ipynb
|   |   model.png
|   |   predict.ipynb
|   |   video_model.h5
|   |
|   +---Resources
|   |       haarcascade_frontalface_default.xml
|   |       haarcascade_russian_plate_number.xml
|   |
|   \---Weights
|           checkpoint
|           model_weights.data-00000-of-00001
|           model_weights.index
|
\---GAN_Fake_vs_Real
    |   predict.ipynb
    |   svm_cnn.ipynb
    |
    +---Inception_Resnet_SVM_Model
    |       Inception_Resnet_svm_model.h5
    |
    \---Weights_Inception_Resnet_SVM
            checkpoint
            model_weights.data-00000-of-00001
            model_weights.index
```