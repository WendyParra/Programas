import cv2
import numpy
import matplotlib.pyplot as plt

captura =cv2.VideoCapture(0)

while True:
    ret, frame =captura.read()

    if not ret:
        break

    cv2.imshow('video en vivo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyALLWindows()
