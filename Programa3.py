import cv2

captura = cv2.VideoCapture(0)

ret, frame = captura.read()

if ret:
    cv2.imwrite('C:/Users/wendy/Downloads/python/4.0.jpg', frame)

captura.release()