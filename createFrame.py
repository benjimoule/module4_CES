# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:02:42 2020

@author: benjamin.policand
"""

import cv2
vidcap = cv2.VideoCapture('videoprojet.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("image/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1