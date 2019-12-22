# import the necessary packages
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
import argparse
import pickle
import cv2
import numpy as np
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
	help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=28,
	help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1,
	help="whether or not we should flatten the image")
args = vars(ap.parse_args())



#vgg model
vggmodel = VGG16()
image = load_img(args["image"], target_size=(224, 224))
print("the image type: ",type(image))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
image = vggmodel.predict(image).flatten()


# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
 
print(model.summary())
print(image)
image = np.array(image, dtype="float")
# image = image.transpose()
# make a prediction on the image
preds = model.predict(image.transpose())

# find the class label index with the largest corresponding
# probability
# i = preds.argmax(axis=1)[0]
# label = lb.classes_[i]
# print("prediction for three catogorial: ",preds)
# output = loadimg(args["image"],target_size=(224,224))
# # draw the class label + probability on the output image
# text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
# print(text)