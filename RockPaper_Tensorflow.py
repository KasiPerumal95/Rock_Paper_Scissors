import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, BatchNormalization,Dropout,ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.activations import relu, softmax

import numpy as np
import cv2

model = load_model("rock_paper_scissors_model.h5")
#model.summary()

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
# text = "stone"
#predicted = random.choice([0, 1, 2])
while rval:
    # print(frame.shape)
    # frame = cv2.resize(frame, (300,300), interpolation = cv2.INTER_AREA)

    img = frame[100:275, 100:250]
    cropped_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    cropped_img = np.expand_dims(cropped_img, axis=0)
    predicted = np.argmax(model.predict(cropped_img))
    #print(predicted)

    if predicted == 0:
        text = "rock"
        frame = cv2.putText(frame, text, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    elif predicted == 1:
        text = "paper"
        frame = cv2.putText(frame, text, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    else:
        text = "scissors"
        frame = cv2.putText(frame, text, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("preview", frame)

    #print(cropped_img.shape)
    # print(frame.shape)
    # frame = cv2.putText(frame, text, (90, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    # cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    else:
        cv2.rectangle(img=frame, pt1=(100, 100), pt2=(275, 250), color=(0, 255, 0), thickness=1, lineType=8, shift=0)
    #predicted = random.choice([0, 1, 2])
    # crop
    # img = frame.crop()
    # model input(img)
    #
vc.release()
cv2.destroyWindow("preview")