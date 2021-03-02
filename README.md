# Human emotion extractor



## Requirements 

Install the following packages in the `requirements.txt` file using the command `pip install -r requirements.txt`  on your terminal in the main folder of this project:

-  opencv-contrib-python==4.0.1 : computer vision library to extract, used to detect faces and extract the face as a region of interest
- pillow=8.1.0 : python build in image reader
- pytorch=1.7.1 : deep learning library used to build a neural network and train it
- torchvision=0.8.2 : part of pytorch, allows us to do data transformation on the data set
- tqdm=4.56.0 
- matplotlib=3.3.4
- pandas=1.2.1 
- scikit-learn=0.23.2
- emoji=1.0.1
- ipython=7.20.0

We provide the pretrained network, but it could be trained again if needed.

## Explanations

To build the human emotion extractor, we use a neural network (the [Deep-Emotion](https://arxiv.org/abs/1902.01019)) trained on the [FER-2013](https://www.kaggle.com/msambare/fer2013) dataset. 

This dataset contains the emotions: 

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

Then we do a transfer learning on the emotion we are willing to detect, since all the emotions are combination of these 7 basic emotions.

To re train the network we use a custom dataset consisting of personal picture on different environments we just have to put the pictures on the `EMOTION ` file corresponding to the respective emotion in `./dataset-transfert/EMOTION`  and then run the `create_dataset()` function.

We can then detect the following emotions:

1. 😁
2. 😳
3. ☹️
4. 😗
5. 🙄
6. 😊
7. 😜

## Results

**Accuracy** on the validation set : **98%**, which is really high for this kind of network with this kind of data set, this is explained by the fact we trained the dataset on the pictures of the same person even though it was on different environments and data augmentation was done by adding random flips, random noises etc.

There is two ways to predict on, with `show_prediction() `  function to see the predictions on a picture or `show_prediction_video() ` to see the direct prediction through the camera of the computer.
