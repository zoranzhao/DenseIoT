import numpy as np
#import cv2
import pprint
import os

from os import listdir
from os.path import isfile, join
mypath ="./data/val2017/"

#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyimgs = [f for f in listdir(mypath) if ".jpg" in f ]

#pprint.pprint(onlyfiles)
#pprint.pprint(onlyimgs)

for i in range(len(onlyimgs)):
   os.rename(mypath+onlyimgs[i], mypath+(str(i)+".jpg"))

# Load an color image in grayscale
# img = cv2.imread('.jpg',0)
