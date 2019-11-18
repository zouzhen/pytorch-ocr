import cv2
import os
file_path = 'D:\\SoftWork\\WorkSpaces\\Node_interconnection\\Code\\data_generator\\background'

file_list = os.listdir(file_path)

for file_name in file_list:

  image = cv2.imread(os.path.join(file_path,file_name))

  img = cv2.resize(image,(450,220),interpolation=cv2.INTER_NEAREST)
  cv2.imwrite(os.path.join(file_path,file_name),img)

# import numpy