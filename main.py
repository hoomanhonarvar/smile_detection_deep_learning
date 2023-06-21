import cv2
import joblib
import time

import numpy as np

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
i=0
with open('traind_model.joblib','rb') as f:
    clf2 = joblib.load(f)
while 1:
    i+=1
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        img_copy = img.copy()
        dsize = (64, 64)
        img_copy = cv2.resize(roi_color, dsize)
        img_copy2=np.expand_dims(img_copy,axis=0)
    # print(type(img_copy2))

    # cv2.imshow("kir",img_copy)
    # cv2.waitKey(0) 

        ynew = clf2.predict(img_copy2)
        if ynew == 1:
            cv2.imwrite(str(i) + "im_smile.jpg", img)
            cv2.putText(img, 'Smile', (x, y - 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 0), 2)

        else:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'not Smile', (x, y - 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

    cv2.imshow("img",img)
    k=cv2.waitKey(1) &0xFF
    if k==32:
        break
cap.release()
cv2.destroyAllWindows()