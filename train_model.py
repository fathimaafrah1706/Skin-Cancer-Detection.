import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

dataset_path = "dataset"

features = []
labels = []

for label in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(class_path):

        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean = np.mean(gray)
        std = np.std(gray)
        entropy = -np.sum(gray/255 * np.log2(gray/255 + 1e-9))

        features.append([mean, std, entropy])
        labels.append(label)

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy * 100, "%")

pickle.dump(model, open("model.pkl", "wb"))