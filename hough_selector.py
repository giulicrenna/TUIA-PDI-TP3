import cv2
import numpy as np

PATH = '1.png'

def update(val):
    img = cv2.imread(PATH, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    dp = cv2.getTrackbarPos('DP', 'Hough Transform') / 100
    minDist = cv2.getTrackbarPos('MinDist', 'Hough Transform')
    param1 = cv2.getTrackbarPos('Param1', 'Hough Transform')
    param2 = cv2.getTrackbarPos('Param2', 'Hough Transform')
    minRadius = cv2.getTrackbarPos('MinRadius', 'Hough Transform')
    maxRadius = cv2.getTrackbarPos('MaxRadius', 'Hough Transform')

    print(f'(dp={dp},', f'minDist={minDist},', f'param1={param1},', f'param2={param2},', f'minRadius={minRadius},',f'maxRadius={maxRadius})')
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist,
                                param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow('Hough Transform', img)

cv2.namedWindow('Hough Transform')

cv2.createTrackbar('DP', 'Hough Transform', 100, 200, update)  # Factor de resolución inversa
cv2.createTrackbar('MinDist', 'Hough Transform', 20, 100, update)  # Distancia mínima entre centros de círculos detectados
cv2.createTrackbar('Param1', 'Hough Transform', 50, 100, update)  # Umbral de borde para Canny
cv2.createTrackbar('Param2', 'Hough Transform', 30, 100, update)  # Umbral para la detección de círculos
cv2.createTrackbar('MinRadius', 'Hough Transform', 0, 100, update)  # Radio mínimo de los círculos detectados
cv2.createTrackbar('MaxRadius', 'Hough Transform', 0, 100, update)  # Radio máximo de los círculos detectados

img = cv2.imread(PATH)

update(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
