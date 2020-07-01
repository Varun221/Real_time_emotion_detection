import numpy as np
import os
import cv2


# emotion list contains the list of emotions
train_path = './data/train'
test_path = './data/test'
emotions = os.listdir(train_path)

# this dictionary maps emotion to the index
emotion_to_index = {'angry':0,
                   'disgusted':1,
                   'fearful':2,
                   'happy':3,
                   'neutral':4,
                   'sad':5,
                   'surprised':6}
 

# image paths contain the path to train 'angry', 'fearful' folders
image_paths = []
for emotion in emotions:
    image_paths.append(os.path.join(train_path, emotion))
    
# test paths contain the path to test 'angry', 'fearful' folders
test_paths = []
for emotion in os.listdir(test_path):
    test_paths.append(os.path.join(test_path, emotion))
    
    

# copying the image arrays in the list    
print("\n Retrieving images from train...")
X = []
Y = []
for image_path in image_paths:
    images = os.listdir(image_path)
    num = len(images)
    print(f"\nFound {num} images in {image_path}")
    for i in range(num):
        path = os.path.join(image_path, images[i])
        img =  cv2.imread(path)
        X.append(img)
        Y.append(emotion_to_index[os.path.split(image_path)[1]])
        

# converting the list to array
X_train = np.array(X)
Y_train = np.array(Y)
shuffled_X = X_train.copy()
shuffled_Y = Y_train.copy()

# The arrays are shuffled
print("\n Shuffling train images ...")
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
for i in range(X_train.shape[0]):
    shuffled_X[i, :, :, :] = X_train[indices[i], :, :, :]
    shuffled_Y[i] = Y_train[indices[i]]
    
Y_fin_train = np.reshape(shuffled_Y, (-1,1))
print("\n Saving the training arrays...")
np.save('Y_train2', Y_fin_train)
np.save('X_train2', shuffled_X)


# test images are retrieved in the form of a list
Xt = []
Yt = []
for test_path in test_paths:
    images = os.listdir(test_path)
    num = len(images)
    print(f"\nFound {num} images in {test_path}")
    for i in range(num):
        path = os.path.join(test_path, images[i])
        img =  cv2.imread(path)
        Xt.append(img)
        Yt.append(emotion_to_index[os.path.split(test_path)[1]])
        

X_ta = np.array(Xt)
Y_ta = np.array(Yt)
        
# shuffling test arrays
shuffled_Xt = X_ta.copy()
shuffled_Yt = Y_ta.copy()

print("\n Shuffling test arrays ...")
indices = np.arange(X_ta.shape[0])
np.random.shuffle(indices)
for i in range(X_ta.shape[0]):
    shuffled_Xt[i, :, :, :] = X_ta[indices[i], :, :, :]
    shuffled_Yt[i] = Y_ta[indices[i]]
        

        
Y_test = np.reshape(shuffled_Yt, (-1,1))
X_test = shuffled_Xt

# saving test arrays
print("\n Saving test arrays...")
np.save('Y_test2', Y_test)
np.save('X_test2', X_test)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
