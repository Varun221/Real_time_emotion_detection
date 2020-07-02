# Real_time_emotion_detection
This project aims to determine the emotion on a person's face in real time into one of seven categories, using deep convolutional neural networks. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.


### Dataset
The original FER-2013 data set can be downloaded [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view).

To train the model - 
Unzip the data into a folder containing dataset_prepare.py and run the python script. \
This will create four ```.npy``` files in the directory namely 'X_train', 'X_test', 'Y_train' and 'Y_test'. \
These are the concatenated training numpy arrays of shapes - \
X_train - (#training_examples, 48, 48, 3) \
X_test - (#testing_examples, 48, 48, 3) \
Y_train - (#training_examples, 1) \
Y_test - (#testing_examples, 1) \
These arrays are shuffled and can be loaded by -  \
```python
import numpy as np
X_train = np.load('X_train.npy')
```
The Original Model was trained in Google colab, the notebook ```emotion_detection.ipynb``` contains the code for training.

### Model Summary
![sum](https://github.com/Varun221/Real_time_emotion_detection/blob/master/images/model_summary.png)
 
The model was trained for 30 epochs with results - 
![f1](https://github.com/Varun221/Real_time_emotion_detection/blob/master/images/f1_score.png) \
![acc](https://github.com/Varun221/Real_time_emotion_detection/blob/master/images/accuracy.png)

Further training resulted in overfitting, hence the training was stopped early. You can experiment with the model and its hyper params in the notebook.

The trained model is given in hdf5 format in ```models``` as well ```code``` folder.
You can load the model in your own script by - 
```python
import tensorflow as tf
model = tf.keras.models.load_model('<path_to_model>/my_model.h5')
```


### Algorithm
1. The face of the person in the feed is predicted by Haar Cascade's algorithm.
2. The Model then takes in the image and outputs a set of softmax scores for each emotion
3. The emotion with maximum softmax score is given as the person's emotion.

The final Result - \
![res](https://github.com/Varun221/Real_time_emotion_detection/blob/master/images/result.png)

### References
The basic architecture of the model was inspired from the research paper, Emotion Recognition using Deep Convolutional Neural Networks by Enrique Correa, Arnoud Jonker, Michaël Ozo and Rob Stolk



 

