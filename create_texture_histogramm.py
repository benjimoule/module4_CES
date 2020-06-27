# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:05:19 2020

@author: benjamin.policand
"""

import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import csv
from scipy.stats import wasserstein_distance 

def lbp_histogram(color_image):
    img = color.rgb2gray(color_image)
    patterns = local_binary_pattern(img, 8, 1)
    hist, _ = np.histogram(patterns, bins=np.arange(2**8 + 1), density=True)
    return hist

with open('frameChanged.csv', 'w') as csvfile:
            fieldnames = ['frameA', 'frameB','seconde']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for i in range(1, 500, 1):
                j=i+1;
                image1 = io.imread('image/frame'+str(i)+'.jpg')
                image2 = io.imread('image/frame'+str(j)+'.jpg')
                
                
                image1_feats = lbp_histogram(image1)
                image2_feats = lbp_histogram(image2)
                
                
                
                hmax = max([image1_feats.max(), image2_feats.max()])
                fig, ax = plt.subplots(2, 2)
                
                ax[0, 0].imshow(image1)
                ax[0, 0].axis('off')
                ax[0, 0].set_title('image1')
                ax[1, 0].plot(image1_feats)
                ax[1, 0].set_ylim([0, hmax])
                
                ax[0, 1].imshow(image2)
                ax[0, 1].axis('off')
                ax[0, 1].set_title('image2')
                ax[1, 1].plot(image1_feats)
                ax[1, 1].set_ylim([0, hmax])
                ax[1, 1].axes.yaxis.set_ticklabels([])
                plt.show(fig)  
                print('distance frame '+str(i))
                print(wasserstein_distance(image1_feats, image2_feats))
                if wasserstein_distance(image1_feats, image2_feats)>0.001:
                     
                        print('scene changed')
                        writer.writerow({'frameA': 'frame'+str(i)+'.jpg','frameB':'frame'+str(j)+'.jpg','seconde':i/25})
                        print('change scene')
       