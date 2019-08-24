from __future__ import print_function
import os
import re
import argparse
import random
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
from models.crnn import *
from utils.utils import *
from utils import utils
from utils import torch_utils
from utils.parse_config import *
import torch.distributed as dist
from dataset_v2 import baiduDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:  # not installed: install help: https://github.com/NVIDIA/apex/issues/259
    mixed_precision = False


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(crnn, val_loader, criterion, iteration, dataset, device, is_mixed, converter,optimizer,loss_avg,val_dataset,max_i=1000):

    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    for i_batch, (image, index) in enumerate(val_loader):
        image = image.to(device)
        label = utils.get_batch_label(val_dataset, index)
        preds = crnn(image,is_mixed=is_mixed)
        if is_mixed:
            preds = preds.permute(1, 0, 2)
        batch_size = image.size(0)
        index = np.array(index.data.numpy())
        text, length = converter.encode(label)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        print(preds.shape, text.shape, preds_size.shape, length.shape)
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, label):
            if pred == target:
                n_correct += 1

        if (i_batch+1)%params.displayInterval == 0:
            print('[%d/%d][%d/%d]' %
                      (iteration, params.niter, i_batch, len(val_loader)))

        if i_batch == max_i:
            break
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct)
    print(max_i * params.val_batchSize)
    accuracy = n_correct / float(max_i * params.val_batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    return accuracy

def train(crnn, train_loader, criterion, iteration, dataset, device,is_mixed, converter,optimizer,loss_avg):

    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    for i_batch, (image, index) in enumerate(train_loader):
        print('训练轮数：',i_batch,mixed_precision)
        image = image.to(device)
        label = utils.get_batch_label(dataset, index)
        preds = crnn(image,is_mixed=is_mixed)
        if is_mixed:
            preds = preds.permute(1, 0, 2)
        # print('preds_shape:',preds.shape)
        batch_size = image.size(0)
        index = np.array(index.data.numpy())
        text, length = converter.encode(label)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        # print(preds.shape, text.shape, preds_size.shape, length.shape)
        # torch.Size([41, 16, 6736]) torch.Size([160]) torch.Size([16]) torch.Size([16])
        cost = criterion(preds, text, preds_size, length)/ batch_size
        # cost = criterion(preds, text, preds_size, length) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)

        if (i_batch+1) % int(params.displayInterval) == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (iteration, params.niter, i_batch, len(train_loader), loss_avg.val()))
            loss_avg.reset()

def main(cfg,
          data,
          epochs=300):  # effective bs = batch_size * accumulate = 16 * 4 = 64
    # Initialize
    init_seeds()
    weights = 'weights' + os.sep
    last = weights + 'last.pth'
    best = weights + 'best.pth'
    device = torch_utils.select_device(apex=mixed_precision)

    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    val_path = data_dict['valid']
    img_path = data_dict['images']
    alphabet = parse_data_name(data_dict['alphabet'])
    params = parse_dict2params(cfg)
    # print(type(params.workers))
    # print(params)

    dataset = baiduDataset(img_path, train_path, alphabet, False, (int(params.imgW), int(params.imgH)))
    val_dataset = baiduDataset(img_path, val_path, alphabet, False, (int(params.imgW), int(params.imgH)))

    train_loader = DataLoader(dataset, batch_size=int(params.batchSize), shuffle=True, num_workers=int(params.workers))
    # shuffle=True, just for time consuming.
    val_loader = DataLoader(val_dataset, batch_size=int(params.val_batchSize), shuffle=True, num_workers=int(params.workers))
    converter = utils.strLabelConverter(dataset.alphabet)
    nclass = len(alphabet) + 1
    nc = 1

    # criterion
    criterion = torch.nn.CTCLoss(reduction='sum')

    # cnn and rnn
    crnn = CRNN(32, nc, nclass, int(params.nh))
    crnn.apply(weights_init)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=float(params.lr),
                               betas=(float(params.beta1), 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=float(params.lr))
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=float(params.lr))

    crnn.register_backward_hook(backward_hook)

    crnn = crnn.to(device)
    certerion = criterion.to(device)

    if opt.resume:
        crnn.load_state_dict(torch.load(last))
    
    Iteration = 0

    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        is_mixed = True
        crnn = torch.nn.parallel.DistributedDataParallel(crnn,dim=0)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = Iteration - 1

    while Iteration < epochs:
        train(crnn, train_loader, criterion, Iteration, dataset, device, is_mixed, converter,optimizer,loss_avg)
        ## max_i: cut down the consuming time of testing, if you'd like to validate on the whole testset, please set it to len(val_loader)
        accuracy = val(crnn, val_loader, criterion, Iteration, dataset, device, is_mixed, converter,optimizer, loss_avg,val_dataset,max_i=1000)
        for p in crnn.parameters():
            p.requires_grad = True
        if accuracy > int(params.best_accuracy):
            torch.save(crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(weights, Iteration, accuracy))
            torch.save(crnn.state_dict(), '{0}/crnn_best.pth'.format(weights))
        print("is best accuracy: {0}".format(accuracy > int(params.best_accuracy)))
        Iteration+=1
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()



def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--accumulate', type=int, default=2, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/params.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/ocr.data', help='coco.data file path')
    parser.add_argument('--random_sample', action='store_true', help='train at (1/1.5)x - 1.5x sizes')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--xywh', action='store_true', help='use xywh loss instead of GIoU loss')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt)

    main(opt.cfg,opt.data,epochs=opt.epochs)
