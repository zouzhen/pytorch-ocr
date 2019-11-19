import cv2
import os
file_path = '/media/lzc274500/Elements/11-18/plate_20191118'

file_list = os.listdir(file_path)
save_path = '/media/lzc274500/Elements/11-18/result'

for file_name in file_list:

  image = cv2.imread(os.path.join(file_path,file_name))

  img = cv2.resize(image,(450,150),interpolation=cv2.INTER_NEAREST)
  cv2.imwrite(os.path.join(save_path,file_name),img)

# import numpy