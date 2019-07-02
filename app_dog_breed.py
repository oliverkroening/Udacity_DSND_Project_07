from flask import Flask
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from ext
from extract_bottleneck_features import extract_Resnet50
import os
import pickle
import cv2
import numpy as np


# load list of dog breed (categories)
with open("dog_names.txt", "rb") as f:   # Unpickling
    dog_names = pickle.load(f)

# load ResNet50_dog_detector model
ResNet50_dog_detector = ResNet50(weights='imagenet')

ResNet50_model.load_weights('saved_models/weights.best.ResNet50_0.3_1024.hdf5')

# implement functions
def face_detector(img_path):
    '''
    Function to detect a face in an image
    IN:  img_path - path to image for face detection
    OUT: True if face is detected
         False if no face is detected
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_0.predict(img))

def dog_detector(img_path):
    '''
    returns "True" if a dog is detected in the image stored at img_path
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def ResNet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def predict_dog_breed(img_path):
    dog_names_pred = ResNet50_predict_breed(img_path)
    if dog_detector(img_path):
        print("This photo looks like a", dog_names_pred.split('.')[1],"!")
    elif face_detector(img_path):
        print("This human mostly resembles a", dog_names_pred.split('.')[1],"!")
    else:
        print("ERROR: Neither a dog nor a human was detected in the image!")
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off');

app = Flask(__name__)

@app.route("/")
def index():
    return dog_names[0]

if __name__ == '__main__':
    app.run(port=1337, debug=True)
