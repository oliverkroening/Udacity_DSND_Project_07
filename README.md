# Udacity Data Science Nanodegree
# Project 07: Capstone Project (Dog Breed Classifier)
--------------------------------------
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## 1. Installation <a name="installation"></a>

- This code was created by using Python versions 3.*.
- Following libraries are required:

* numpy
* matplotlib
* pickle
* keras
* opencv (cv2)
* sklearn
* PIL
* tqdm
* glob
* random
* time

- The web app is not ready yet. The Flask library is used to create the web app. To start the app, you have to run:
python app.py

- If the web server starts correctly, you can enter the web app on your localhost by entering 127.0.0.1:4555 in your browser. I tested a version of the app in Chromium. Here, it was necessary to disable the web security by executing the following command in the terminal:
"C:\Path_to_Chrome\chrome.exe" –allow-file-access-from-files -disable-web-security

- You can copy the repository by executing: git clone https://github.com/oliverkroening/Udacity_DSND_Project07

## 2. Project Motivation <a name="motivation"></a>
Within Udacities Capstone project, I had the choice of what to do and which project to pick. Since I am very interested in image classification and processing using machine learning techniques, I decided for the dog breed image classifier.

In this project, the aim was to build, train and test an image classifier for dog breeds. In comparison the the flower classifier, the method of transfer learning has to be applied to the model. Furthermore, the project contains some face detection using OpenCV, which was also an interesting part to come up with.

## 3. File Descriptions <a name="files"></a>  
The project mainly consists of a jupyter notebook:
- `dog_app.ipynb` 

The images and test images for the notebook and the testing of the algorithm can be found in the `images` and `test_images` folders.

Following files have to be imported as modules within the notebook:
- `extract_bottleneck_features.py`

The  Haar feature-based cascade classifiers for face detection can be found in the `haarcascades` folder

The folder `saved_models` contains the trained models, that I achieved while working on the project:
- `ResNet50_model.h5` and `weights.best.ResNet50_0.3_1024.hdf5` (ResNet50 models with highest accuracy achieved)
- `weights.best.VGG16.hdf5`(VGG16 model of step 4 of the notebook)
- `weights.best.VGG19_0.15_256.hdf5`(example of VGG19 model with an added dense layer with 256 nodes and a dropout layer with a dopout rate of 0.15)

Furthermore, I included the submitted HTML-file that contains the jupyter notebook.

## 4. Results <a name="results"></a>
With the help of transfer learning and Keras, I have built and trained a quite efficient and accurate CNN.Therefore, I tested different models and pre-trained networks on the metric of training time and testing accuracy. In the case of predicting dogs by using images of the dog, a pretty good accuracy of 85.29% on the testing dataset was achieved by using the ResNet50 network and the corresponding bottleneck features. The fine-tune training of the network took only less than a minute (GPU mode).

This project was published as a blog post on [Medium](https://medium.com/@oliver.kroening/classify-dog-breeds-by-using-keras-and-transfer-learning-3ea8f7d3af86)

## 5. Licensing, Authors, Acknowledgements<a name="licensing"></a>
All data was provided by Udacity, thus, I must give credit to them. Other references are cited in the Jupyter notebook.
Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.



