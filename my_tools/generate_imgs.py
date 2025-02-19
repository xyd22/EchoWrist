import cv2
import numpy as np
import math

h, w = 512, 512
font_size = 10.0
coord_scale = 20

img_blank = np.zeros([h, w, 3], dtype = np.uint8)
img_blank[:] = 255
for i in range(1, 10):
    cv2.putText(img_blank, str(i), (math.floor((w-font_size*coord_scale)/2), math.floor((h+font_size*coord_scale)/2)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), thickness=5)
    cv2.imwrite('obj_' + str(i) + '.png', img_blank)
    img_blank[:] = 255

