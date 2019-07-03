from flask import Flask, render_template, request, jsonify
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from extract_bottleneck_features import extract_Resnet50
import os
import pickle
import cv2
import numpy as np


# load list of dog breed (categories)
with open("dog_names.txt", "rb") as f:   # Unpickling
    dog_names = pickle.load(f)

# load ResNet50_dog_detector model and ResNet50_model to classify the dog
ResNet50_dog_detector = load_model('saved_models/ResNet50_dog_predict.h5')
ResNet50_model = load_model('saved_models/ResNet50_model.h5')

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
    '''
    Function to convert the image to a tensor
    IN:  img_path - path to image
    OUT: 4D-tensor with shape (1,224,224,3)
    '''
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    '''
    Function to convert images to tensors
    IN:  img_paths - list of image paths
    OUT: list of 4D-tensor with shape (1,224,224,3)
    '''
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    '''
    Function to predict category for a located image at img_path
    IN:  img_path - path to image
    OUT: predicted label with highest probability
    '''
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_dog_detector.predict(img))

def dog_detector(img_path):
    '''
    Function to detect a dog in an image
    IN:  img_path - path to image
    OUT: "True" if there is a dog in the image, "False" elsewhere
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def ResNet50_predict_breed(img_path):
    '''
    Function to predict the dog breed by using the ResNet50_model on image
    IN:  img_path - path to image
    OUT: predicted dog breed with highest probability
    '''
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def predict_dog_breed(img_path):
    '''
    Function to detect a dog or human in an image and to return the image
    as well as the predicted dog breed or the dog breed that resembles the
    the human being most.
    IN:  img_path - path to image
    OUT: image of dog
         predicted dog breed
    '''
    dog_names_pred = ResNet50_predict_breed(img_path)
    if dog_detector(img_path):
        print("This photo looks like a", dog_names_pred.split('.')[1],"!")
    elif face_detector(img_path):
        print("This human most resembles a", dog_names_pred.split('.')[1],"!")
    else:
        print("ERROR: Neither a dog nor a human was detected in the image!")
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off');

# configure web app
UPLOAD_FOLDER = os.path.basename('upload_images')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# check valid extensions
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/upload_images", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'upload_images/')
    print(target)

    if not 
if __name__ == '__main__':
    app.run(port=1337, debug=True)
