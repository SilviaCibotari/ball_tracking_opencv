# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from math import *
import numpy as np
#open image from local env
from PIL import Image
img = Image.open(r"C:\Users\Silvia\Desktop\ball.jpg")
#img.show()

#get the pixels values
pixels = img.load() #list(img.getdata())

width, height = img.size
print(width, height)

print(pixels[1,1])
#convert to grayscale
grays=[]
for i in range(width-1):
    for j in range(height-1):
        r = pixels[i, j][0]
        g = pixels[i, j][1]
        b = pixels[i, j][2]
        print(r, g, b)
        med = floor(0.2989 * r + 0.5870 * g + 0.1140 * b)
        print(med)
        pixels[i, j] = (med, med, med)
      
#myarray = np.asarray(grays)        

#image = Image.fromarray(pixels)
img.save("ballgray.png")
img.show()