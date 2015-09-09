#!/home/bginsburg/anaconda/bin/python

"""Extract confusion matrix from log file
   usage: file.log
   
"""

import sys
import re
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

#--------------------------------------------------------------------
def extract_confusion_matrix(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net \(#0\)', openFile)
  print iteration
  numLabels = re.findall(r'Test net output #\d*: confusion_matrix \[(\d*)', openFile) 
  numClasses=int(numLabels[0])
  #print numClasses
  class_prob = re.findall(r'       confusion_matrix    \d*: ([d*.\d*  ]*)', openFile)
  print class_prob
  temp=[]
  for i in range(len(class_prob)) :
    temp.append(map(float, class_prob[i].split()))
  temp2=np.array(temp)

  dim=temp2.shape
  temp3= temp2.reshape(dim[0]/ dim[1], dim[1], dim[1])
  print temp3.shape
  print temp3[-1]
  return temp3 

#----------------------------------------------------------------------
def main():
  # command-line parsing
  filename = sys.argv[1]
  if not filename:
    print 'usage: file.log '
    sys.exit(1)
 
  cm = extract_confusion_matrix(filename)
  
  fig = plt.figure()
  num_steps = cm.shape[0]
  print num_steps
  cmap=plt.cm.brg
  ims = []
  for i in range(num_steps):
    plt.title('Num of steps: ' + str(i))
    ims.append((plt.imshow(cm[i],interpolation='nearest', cmap=cmap ),))
   
  im_ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=100000, blit=True)   
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  
  
if __name__ == '__main__':
  main()
