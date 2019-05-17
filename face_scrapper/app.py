import cv2
import sys
imagePath = sys.argv[1]
image = cv2.imread(imagePath)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
        gray_image,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30,30)
)

print("Found {0} Faces!".format(len(faces)))
