# -*- encoding: utf-8 -*-

import face_recognition
import os
from collections import OrderedDict
import cv2
import json
import time


_dir = '/home/lei/Desktop/demo'
tmp = 'tmp.json'
if os.path.isfile(tmp):
    database = json.load(open(tmp))
else:
    database = OrderedDict()

faces = []
names = []

# for i in os.listdir(_dir):
#     f = os.path.join(_dir, i)
#     name = os.path.splitext(i)[0]
#     if name in database:
#         continue
#
#     image = face_recognition.load_image_file(f)
#     ens = face_recognition.face_encodings(image)
#     if ens:
#         database[name] = list(ens[0])
#     else:
#         database[name] = []

json.dump(database, open(tmp, 'w'))

faces = []
names = []

for key in database.keys():
    if database[key]:
        names.append(key)
        faces.append(database[key])


# faces = list(database.values())
# names = list(database.keys())

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

while True:
    _, frame = cap.read()

    dev = 1

    rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=1/dev, fy=1/dev)

    t = time.time()
    face_locations = face_recognition.face_locations(rgb_frame)
    # gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 1)
    # face_locations = []
    # for (x,y,w,h) in faces:
    #     face_locations.append((x,y,w,h))

    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        # for face_encoding in face_encodings:
        #     r = face_recognition.face_distance(faces, face_encoding)
        #     t = time.time()
        #     results = face_recognition.compare_faces(faces, face_encoding, tolerance=0.30)
        #     print(time.time() -t )
        #
        #     if results.count(True) == 1:
        #         index = results.index(True)
        #         name = names[index]
        #         face_names.append(name)
        #     elif results.count(True) > 1:
        #         index = r.index(min(r))
        #         name = names[index]
        #         face_names.append(name)
        #     else:
        #         face_names.append('???')

        # face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
        # for face_landmarks in face_landmarks_list:
        #     for each in face_landmarks.values():
        #         tmp = each[0]
        #         for i in each[1:]:
        #             cv2.line(frame, tmp, i, (0,255,0))  # 5
        #             tmp = i

        face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
        for face_landmarks in face_landmarks_list:
            for each in face_landmarks.values():
                for i in each:
                    cv2.circle(frame, i, 2, (0,255,0), -1)

        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     top *= dev
        #     right *= dev
        #     bottom *= dev
        #     left *= dev
        #     # Draw a box around the face
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #
        #     # Draw a label with a name below the face
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    print(time.time() -t)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
