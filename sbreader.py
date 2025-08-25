import easyocr
import cv2
import numpy as np
from PIL import ImageGrab

reader = easyocr.Reader(['en'])  # GPU support if available
bbox = (1550, 200, 1920, 800)

screenshot = ImageGrab.grab(bbox=bbox)
img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

results = reader.readtext(img, detail=0)
print("Minecraft Text:", results)
