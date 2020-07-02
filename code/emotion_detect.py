import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')

index_to_emotion = {0: 'angry',
                    1: 'disgusted',
                    2: 'fearful',
                    3: 'happy',
                    4: 'neutral',
                    5: 'sad',
                    6: 'surprised'}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
float_min = np.nextafter(np.float32(0), np.float32(1))

ret, frame = cap.read()
while (ret == True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x-5, y-10), (x+w+5, y + h+10), (255, 0, 0), 2)
        roi_gray = gray[y+25:y-10 + h, x+20:x + w-20]
        roi_color = frame[y+25:y-10 + h, x+20:x + w-20]
        img = np.zeros_like(roi_color)
        img[:, :, 0] = roi_gray
        img[:, :, 1] = roi_gray
        img[:, :, 2] = roi_gray

        # resizing the image
        try:
            image = cv2.resize(img, (48,48), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(str(e))
        img_fed = np.expand_dims(image, axis=0)
        scores = model.predict(img_fed)
        index = np.argmax(scores)
        frame = cv2.putText(frame, index_to_emotion[index], (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), thickness=2)

    cv2.imshow('detection', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break


cv2.destroyAllWindows()