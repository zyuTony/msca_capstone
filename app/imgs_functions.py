# https://gist.github.com/kodekracker/1777f35e8462b2b21846c14c0677f611
import cv2
import numpy as np
def drawBoundingBoxes(imageData, inferenceResults, color = (0, 0, 0)):
    
    newimage = np.array(imageData)

    for index, row in inferenceResults.iterrows():
        
        left = int(row['xmin'])
        right = int(row['xmax'])
        top = int(row['ymin'])
        bottom = int(row['ymax'])
        label = row['name']
        imgHeight, imgWidth, _ = newimage.shape
        thick = int((imgHeight + imgWidth) // 600)
        newimage = cv2.rectangle(newimage,(left, top), (right, bottom), color, thick)
        newimage = cv2.putText(newimage, label, (left, top - 12), 0, 1e-3 * imgHeight, color, thick//2)
        
    return newimage



def readandprocess(img):
    
    clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize = (10, 10))
    
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (120, 120))
    img = clahe.apply(img)
    img = np.array(img).astype(float)
    
    return img
