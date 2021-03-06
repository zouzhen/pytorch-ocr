'''
该脚本实现了生产者消费者模型
以多进程或者多线程的方式
'''

import cv2
import os
import copy
import time
import multiprocessing
from multiprocessing import Process, Manager
import os.path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw
from datetime import datetime
import random
img_label = {"皖":'an_hui', "沪":'shang_hai', "津":'tian_jin', "渝":'chong_qing', "冀":'he_bei', "晋":'jin_shan_xi', "蒙":'nei_meng', "辽":'liao_ning', "吉":'ji_lin', "黑":'hei_long_jiang', "苏":'su_zhou', "浙":'zhe_jiang', "京":'bei_jing', "闽":'fu_jian', "赣":'jiang_xi', "鲁":'shan_dong', "豫":'he_nan', "鄂":'hu_bei', "湘":'hu_nan', "粤":'guang_dong', "桂":'guang_xi', "琼":'hai_nan', "川":'shi_chuan', "贵":'gui_zhou', "云":'yun_nan', "藏":'xi_zang', "陕":'shan_shan_xi', "甘":'gan_su', "青":'qing_hai', "宁":'nan_jing', "挂":'gua_che', 'A':'A', 'B':'B', 'C':'C', 'D':'D', 'E':'E', 'F':'F', 'G':'G', 'H':'H', 'J':'J', 'K':'K', 'L':'L', 'M':'M', 'N':'N', 'P':'P', 'Q':'Q', 'R':'R', 'S':'S', 'T':'T', 'U':'U', 'V':'V', 'W':'W','X':'X', 'Y':'Y', 'Z':'Z', '0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9', '·':'Pointer'}


class Producer(Process):

    def __init__(self, queue, food):  # 重写.
        super().__init__()  # 加入父类init.
        self.queue = queue
        self.food = food

    def run(self):  # call start()时 就会调用run(run为单进程).
        while True:
        # print('1')
            if type(self.food) == list:
                if len(self.food)==0:
                    print("生产者生产完毕")
                    break                
                item = self.food.pop()  # left is closed and right is closed.
                self.queue.put(item)
                print("Producer-->%s" % item)
                time.sleep(0.1)



class Consumer(Process):
    def __init__(self, queue, label, **args):  # 重写.
        super().__init__()  # 加入父类init.
        self.queue = queue
        self.label = label
        self.args = args

    def horizontal_mirror_imgs(self, imgs_path, xml_path, item, save_path):
        # tree = xmlET.parse(os.path.join(imgs_path, item))
        tree = xmlET.parse(xml_path)
        root = tree.getroot()
        root.find('filename').text = item
        label_str = item[:7]
        for index,obj in enumerate(root.findall('object')):
            obj.find('name').text = img_label[label_str[index]]
        tree.write(os.path.join(save_path, item.split(".")[0]+'.xml'))

    def run(self):  # call start()时 就会调用run(run为单进程).
      while True:
          item = self.queue.get()
          self.horizontal_mirror_imgs(self.args['imgs_path'],self.args['xml_path'],item,self.args['save_path'])
          print("Consumer-->%s" % item, self.label)
          self.queue.task_done()


if __name__ == '__main__':
	imgs_path = '/media/lzc274500/Elements/11-18/result'
	xml_path = '/media/lzc274500/Elements/11-18/京A65359_266930.xml'
	save_path = '/media/lzc274500/Elements/11-18/save_path'
	pathlist = os.listdir(imgs_path)
	# 统计计算内部的核心进程数
	if not os.path.exists(save_path):
	  os.makedirs(save_path)
	cores = multiprocessing.cpu_count()
	qMar = Manager()
	# 取核心进程数的一半建立数据队列
	q1 = qMar.Queue(cores-5)
	p = Producer(q1, pathlist)
	processes = []
	processes.append(p)
	# print(int(cores/2))
	for i in range(cores-5):
		processes.append(Consumer(q1,i,imgs_path=imgs_path,xml_path=xml_path, save_path=save_path))
		
	[process.start() for process in processes]
	[process.join() for process in processes]