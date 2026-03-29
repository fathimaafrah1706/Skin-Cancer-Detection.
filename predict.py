import cv2
import numpy as np
import pickle

# load trained model
model = pickle.load(open("model.pkl", "rb"))

# ask user for image
path = input("Enter image path: ")

# read image
img = cv2.imread(path)

if img is None:
    print("Image not found")
    exit()

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# extract features
mean = np.mean(gray)
std = np.std(gray)
entropy = -np.sum(gray/255 * np.log2(gray/255 + 1e-9))

features = np.array([[mean, std, entropy]])

# prediction
prediction = model.predict(features)

print("Predicted Disease:", prediction[0])