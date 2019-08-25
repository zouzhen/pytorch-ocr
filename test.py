import numpy as np
import sys, os
import time
import cv2
sys.path.append(os.getcwd())
# crnn packages
import torch
from torch.autograd import Variable
from utils import utils
import models.crnn as crnn
from utils.parse_config import *
from utils import alphabets
# import params
import argparse
str2 = alphabets.alphabet
str1 = parse_data_name('./data/ocr.names')

assert str1 == str2

params = parse_dict2params('./cfg/params.cfg')
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='test_images/7.jpeg', help='the path to your images')
opt = parser.parse_args()


# crnn params
# 3p6m_third_ac97p8.pth
crnn_model_path = 'weights/mixed_second_finetune_acc97p7.pth'
# crnn_model_path = 'crnn_Rec_done_1.pth'
alphabet = str1
print(len(alphabet))
nclass = len(alphabet)+1

# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    ### ratio
    ### 280是中文训练集中图片的宽度，160是将原始图片缩小后的图片宽度
    w_now = int(image.shape[1] / (280 * 1.0 / int(params.imgW)))
    h, w = image.shape
    image = cv2.resize(image, (0,0), fx=w_now/w, fy=int(params.imgH)/h, interpolation=cv2.INTER_CUBIC)
    image = (np.reshape(image, (int(params.imgH), w_now, 1))).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(float(params.mean)).div_(float(params.std))
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


if __name__ == '__main__':

	# crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    
    started = time.time()
    ## read an image
    image = cv2.imread(opt.images_path)

    crnn_recognition(image, model)
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
    
