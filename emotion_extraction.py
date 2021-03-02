import os
from PIL import Image
import numpy as np
import cv2
import pickle

def create_dataset():

    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml ')

    BASE_DIR = '.\OVA'
    images_dir = '.\dataset-transfert'

    X_train = []
    y_train = []

    labels_ids = {'winking-tongue': 0,
                  'flushed-face': 1,
                  'rolling-eyes': 2,
                  'big-smile': 3,
                  'slight-smile': 4,
                  'frowning': 5,
                  'kiss': 6
                  }
    
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            
            if file.endswith('png') or file.endswith('PNG') or file.endswith('jpg') or file.endswith('JPG'):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(' ', '-').lower()
#                 pil_image = Image.open(path).convert("L")  # open image and convert to gray scale
#                 image_array = np.array(pil_image, np.uint8)  # convert image to matrix
                image = cv2.imread(path)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if label in labels_ids:
                    id_ = labels_ids[label]
                else:
                    id_ = 99
                faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1,
                                                      minNeighbors=5,
                                                      minSize=(30, 30),
                                                      flags=cv2.CASCADE_SCALE_IMAGE
                                                      )
         
                output_dir = path
                for (x, y, w, h) in faces:
                    roi_gray = image_gray[y:y + h, x:x + w]
                    cv2.imwrite(output_dir, roi_gray)
                    
#                     X_train.append(roi)
#                     y_train.append(id_)
                    
    
#     return X_train, y_train