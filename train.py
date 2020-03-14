from __future__ import print_function
import os
import re
import time
import argparse
import random
import torch
import torch.optim as optim
import numpy as np
from models.crnn import *
from utils.utils import *
from utils import utils
from utils import torch_utils
from utils.parse_config import *
import torch.distributed as dist
from utils.dataset_v2 import baiduDataset
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

def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero


def val(crnn, val_loader, criterion, iteration, dataset, device, is_mixed, converter,optimizer,loss_avg,val_dataset,params,max_i=1000):

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
        # print(preds.shape, text.shape, preds_size.shape, length.shape)
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

def train(crnn, train_loader, criterion, iteration, dataset, device,is_mixed, converter,optimizer,loss_avg,params):

    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    # pbar = tqdm(enumerate(train_loader), total=len(train_loader),miniters=0.05)
    for i_batch, (image, index) in pbar:
        # print('当前轮次：',i_batch)
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

        if (i_batch+1) % params.displayInterval == 0:
            s = '[%d/%d] Loss: %f ' % (iteration, opt.epochs, loss_avg.val())
            pbar.set_description(s)
            # print('[%d/%d][%d/%d] Loss: %f' %
            #       (iteration, params.niter, i_batch, len(train_loader), loss_avg.val()))
            loss_avg.reset()
        pbar.close()
        # time.sleep(0.1)
        
    return s

def main(cfg,
         data,
         batch_size=64,
         epochs=300):
    # Initialize
    init_seeds()
    weights = 'weights' + os.sep
    last = weights + 'last.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device(apex=mixed_precision)

    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    val_path = data_dict['valid']
    img_path = data_dict['images']
    alphabet = parse_data_name(data_dict['alphabet'])
    params = parse_dict2params(cfg)
    # print(params)
    # print(type(params.workers))
    # print(params)

    dataset = baiduDataset(img_path, train_path, alphabet, False, params)
    val_dataset = baiduDataset(img_path, val_path, alphabet, False, params)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=params.workers)
    # shuffle=True, just for time consuming.
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size/2), shuffle=True, num_workers=params.workers)
    converter = utils.strLabelConverter(dataset.alphabet)
    nclass = len(alphabet) + 1
    nc = 1

    # criterion
    criterion = torch.nn.CTCLoss(reduction='sum')

    # cnn and rnn
    crnn = CRNN(32, nc, nclass, params.nh)
    crnn.apply(weights_init)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
                               betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    crnn.register_backward_hook(backward_hook)

    crnn = crnn.to(device)
    certerion = criterion.to(device)

    start_epoch = 0
    best_accuracy = 0

    if opt.resume:
        chkpt = torch.load(last, map_location=device)
        crnn.load_state_dict(chkpt['model'])
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_accuracy']

        if chkpt['training_results'] is not None:
            with open('results.txt', 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        is_mixed = True
        crnn = torch.nn.parallel.DistributedDataParallel(crnn,dim=0)
    else:
        is_mixed = False

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        scheduler.step()
        s = train(crnn, train_loader, criterion, epoch, dataset, device, is_mixed, converter,optimizer,loss_avg,params)
        with open('results.txt', 'a') as file:
            file.write(s + '\n')
        ## max_i: cut down the consuming time of testing, if you'd like to validate on the whole testset, please set it to len(val_loader)
                

        save = (not opt.nosave) or ((not opt.evolve) and (epoch == epochs - 1))
        if save:
            if epoch % 3 == 0 and epoch != 0:
                accuracy = val(crnn, val_loader, criterion, epoch, dataset, device, is_mixed, converter,optimizer, loss_avg,val_dataset,params,max_i=1000)
                for p in crnn.parameters():
                    p.requires_grad = True
                # if accuracy > params.best_accuracy:
                #     torch.save(crnn.state_dict(), best)
                #     torch.save(crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(weights, Iteration,accuracy))
                #     print("is best accuracy: {0}".format(accuracy > params.best_accuracy))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy

                with open('results.txt', 'r') as file:
                    # Create checkpoint
                    chkpt = {'epoch': epoch,
                            'best_accuracy': best_accuracy,
                            'training_results': file.read(),
                            'model': crnn.module.state_dict() if type(
                                crnn) is nn.parallel.DistributedDataParallel else crnn.state_dict(),
                            'optimizer': optimizer.state_dict()}

                # Save last checkpoint
                torch.save(chkpt, last)

                # Save best checkpoint
                if best_accuracy == accuracy :
                    torch.save(chkpt, '{0}/crnn_Rec_done_{1}_{2}.pt'.format(weights, epoch,accuracy))
                    # torch.save(chkpt, best)

                # Save backup every 10 epochs (optional)
                # if epoch > 0 and epoch % 10 == 0:
                #     torch.save(chkpt, weights + 'backup%g.pt' % epoch)

                # Delete checkpoint
                del chkpt
        epoch+=1

    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
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

    main(opt.cfg,opt.data,epochs=opt.epochs,batch_size=opt.batch_size)
