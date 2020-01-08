# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.cm as cm
from collections import defaultdict
from PIL import Image, ImageDraw

initial_image = 'balls2.png'

def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            mean = floor(0.2989 * pixel[0] + 0.5870 * pixel[1] + 0.1140 * pixel[2])
            input_pixels[x, y] = (mean, mean, mean)
            grayscale[x,y] = mean
    return grayscale


def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)

def canny_edge_detector(input_image):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    grayscaled = compute_grayscale(input_pixels, width, height)
    input_image.save('grays.png')
    image3 = Image.fromarray(np.uint8(grayscaled).transpose())
    image3.save('grays2.png')
    # Blur it to remove noise
    blurred = compute_blur(grayscaled, width, height)
    image2 = Image.fromarray(np.uint8(blurred).transpose())
    image2.save('blurred.png')

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)
    image4 = Image.fromarray(np.uint8(gradient).transpose())
    image4.save('gradient.png')
    image5 = Image.fromarray(np.uint8(direction).transpose())
    image5.save('direction.png')

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)
    image6 = Image.fromarray(np.uint8(gradient).transpose())
    image6.save('filter_non_maximum.png')

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep

def detect(image, rmin, rmax, steps, threshold, out):
    points = []
    for r in range(rmin, rmax+1):
        for t in range(steps):
            points.append((r, int(r * np.cos(2 * np.pi * t / steps)), int(r * np.sin(2 * np.pi * t / steps))))
            
    acumulator = defaultdict(int)
    img_ = Image.open(initial_image)
    width,height = img_.size
    w = canny_edge_detector(img_)
    print(str(len(w)) + ' edges out of ' + str(width*height) + ' pixels')
    
    for x, y in w:
        for r, dx, dy in points:
            a=x-dx
            b=y-dy
            acumulator[(a,b,r)] += 1
    
    circles = []
    for k, v in sorted(acumulator.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            circles.append((x, y, r))
    
    for x, y, r in circles:
        ImageDraw.Draw(out).ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))  


def main():
    
    image = Image.open(initial_image)
    
    output_image = Image.new("RGB", image.size)
    output_image.paste(image)
    
    detect(image, 8,15, 2000, 0.6, output_image)
    
    output_image.save("result.png")
    
if __name__ == '__main__':
    main()
    print('Done!')
