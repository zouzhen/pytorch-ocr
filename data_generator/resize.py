import cv2

image = cv2.imread('/home/lzc274500/WorkSpace/ZOUZHEN/Pytorch/crnn_chinese_characters_rec/13.jpg')

img = cv2.resize(image,(450,150),interpolation=cv2.INTER_NEAREST)
cv2.imwrite('/home/lzc274500/WorkSpace/ZOUZHEN/Pytorch/crnn_chinese_characters_rec/13.jpg',img)

# import numpy