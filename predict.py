import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler


from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import cv2
from IPython.display import clear_output
import emoji

def transparentEmoji(image, emoji, pos=(0, 0), scale=1):
        emoji = cv2.resize(emoji, (0, 0), fx=scale, fy=scale)
        eh, ew, e_ = emoji.shape
        ih, iw, i_ = image.shape

        x, y = pos[0], pos[1]

        for i in range(eh):
            for j in range(ew):
                if x + i >= ih or y + j >= iw:
                    continue
                alpha = float(emoji[i][j][2]/255.)
                image[x + i][y + j] = alpha * emoji[i][j][:3] + (1 - alpha) * image[x + i][y + j]

        return image
    
def show_prediction(image_path, net):
    
    image = cv2.imread(image_path)
    
    if image.shape[0] > 800:
            windows_scale = 800/image.shape[0]
            windows_width = int(image.shape[0]*windows_scale)
            windows_height = int(image.shape[1]*windows_scale)
            image = cv2.resize(image, (windows_height, windows_width))
    
    
    
    emoji0 = cv2.imread('./emojis/emoji0.png')
    emoji1 = cv2.imread('./emojis/emoji1.png')
    emoji2 = cv2.imread('./emojis/emoji2.png')
    emoji3 = cv2.imread('./emojis/emoji3.png')
    emoji4 = cv2.imread('./emojis/emoji4.png')
    emoji5 = cv2.imread('./emojis/emoji5.png')
    emoji6 = cv2.imread('./emojis/emoji6.png')
    
    emojis = [emoji0, emoji1, emoji2, emoji3, emoji4, emoji5, emoji6]
    emojis_icons = ['😁', '😳', '☹️', '😗', '🙄' , '😊', '😜']
        
    
    faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image_gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE
                                        )    
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image_face_gray = image_gray[y:y + h, x:x + w]
        image_face_color = image[y:y + h, x:x + w]
    
        image_face_gray_intermediate = cv2.resize(image_face_gray, (48, 48))
        image_face_gray_final = np.expand_dims(image_face_gray_intermediate, axis = 0)
        image_face_gray_final = np.expand_dims(image_face_gray_final, axis = 0)
        image_face_gray_final = image_face_gray_final/255.
        input_data = torch.from_numpy(image_face_gray_final).type(torch.FloatTensor)
        output = net(input_data)
        prediction_probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(prediction_probabilities)
        top_p, top_class = prediction_probabilities.topk(6, dim = 1)
        a = prediction_probabilities[0][1].detach().numpy()

        
        scale = image.shape[0]/emoji0.shape[0]/20
        for i in range(len(prediction_probabilities[0])):

            if i == prediction:
                opacity = 1
                text_color = (0, 0, 255)
            else:
                opacity = 1
                text_color = (0, 255, 0)
                
            emoji_down_sized = int(emojis[i].shape[0]*scale)
            emoji = transparentEmoji(image,emojis[i], (y + i * emoji_down_sized, x + w), scale)
            output = image.copy()

            cv2.addWeighted(emoji, opacity, output, 1 - opacity, 0, output)
            p = prediction_probabilities[0][i].detach().numpy()
            p = np.float(p)
            p = round(p, 2)
            
            cv2.putText(output,
                '%.2f' % p,
                (x + w + emoji_down_sized, y + i * emoji_down_sized + int(emoji_down_sized/1.7) ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2
               )
            image = output.copy()
    

            string_to_print = 'Probability of ' + emojis_icons[i] + ': '
            p_ = p*100
            print(string_to_print + '{:.2%}'.format(p))

            

    cv2.imshow('Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

#     plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

def show_prediction_video(net):
    
    
    emoji0 = cv2.imread('./emojis/emoji0.png')
    emoji1 = cv2.imread('./emojis/emoji1.png')
    emoji2 = cv2.imread('./emojis/emoji2.png')
    emoji3 = cv2.imread('./emojis/emoji3.png')
    emoji4 = cv2.imread('./emojis/emoji4.png')
    emoji5 = cv2.imread('./emojis/emoji5.png')
    emoji6 = cv2.imread('./emojis/emoji6.png')
    
    emojis = [emoji0, emoji1, emoji2, emoji3, emoji4, emoji5, emoji6]
        
    
    faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(image_gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE
                                        )    
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image_face_gray = image_gray[y:y + h, x:x + w]
            image_face_color = image[y:y + h, x:x + w]

            image_face_gray_intermediate = cv2.resize(image_face_gray, (48, 48))
            image_face_gray_final = np.expand_dims(image_face_gray_intermediate, axis = 0)
            image_face_gray_final = np.expand_dims(image_face_gray_final, axis = 0)
            image_face_gray_final = image_face_gray_final/255.
            input_data = torch.from_numpy(image_face_gray_final).type(torch.FloatTensor)
            output = net(input_data)
            prediction_probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(prediction_probabilities)
            a = prediction_probabilities[0][1].detach().numpy()


            scale = image.shape[0]/emoji0.shape[0]/20
            for i in range(len(prediction_probabilities[0])):

                if i == prediction:
                    opacity = 1
                    text_color = (0, 0, 255)
                else:
                    opacity = 1
                    text_color = (0, 255, 0)

                emoji_down_sized = int(emojis[i].shape[0]*scale)
                emoji = transparentEmoji(image,emojis[i], (y + i * emoji_down_sized, x + w), scale)
                output = image.copy()

                cv2.addWeighted(emoji, opacity, output, 1 - opacity, 0, output)
                p = prediction_probabilities[0][i].detach().numpy()
                p = np.float(p)
                p = round(p, 2)
                cv2.putText(output,
                    '%.2f' % p,
                    (x + w + emoji_down_sized, y + i * emoji_down_sized + int(emoji_down_sized/1.7) ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    2
                   )
                image = output.copy()
        cv2.imshow('Video', image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    