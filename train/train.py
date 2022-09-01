import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import io
from PIL import Image, ImageOps
from argparse import ArgumentParser
import math
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import matplotlib.pyplot as plt
import dataset 
import transform

import importlib
from iouEval import iouEval, getColorEntry
from metrics import SegmentationMetric
from shutil import copyfile


NUM_CLASSES = 3  # 
NUM_CHANNELS=0
image_transform = ToPILImage()

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        x = torch.nn.functional.log_softmax(outputs, dim=1)

        return self.loss(x, targets)

def train(args, model, enc=False):

    best_acc = 0
    # TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    # create a loder to run all images and calculate histogram of labels, then create weight array using class balancing
    #w2 =  [0.843328588389030,1.228165403216355]
    # w2_eroder = [0.8125 1.2999]
    # w3_erode = [0.7502 1.9989 0.8571]
    # modu 0.7730 1.4157
    # erode3 0.7858  1.3747
    # class 3                 0.5568  15.2416   0.8785
# 0.6739
#    11.4490
#     0.6999
    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = 0.6739
        weight[1] = 11.4490
        weight[2] = 0.6999
    else:
        weight[0] = 0.6739
        weight[1] = 11.4490
        weight[2] = 0.6999

    assert os.path.exists(
        args.datadir), "Error: datadir (dataset directory) could not be loaded"

    train_transform = transform.Compose([
        transform.RandomVerticalFlip(),
        transform.RandomHorizontalFlip(),
        transform.RandomAngleRotation(),
        transform.ToTensor_and_Resize_EncodeTarget(enc)
    ])
    valid_transform = transform.Compose([
        transform.ToTensor_and_Resize_EncodeTarget(enc)
    ])

    dataset_train = dataset.teethmodel(
        root=args.datadir, transform=train_transform, subset="train", datamode=args.datamode, NUM_CHANNELS =NUM_CHANNELS,filename='filename_20')
    dataset_val = dataset.teethmodel(
        root=args.datadir, transform=valid_transform, subset="val", datamode=args.datamode, NUM_CHANNELS=NUM_CHANNELS,filename='filename_all')

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print(len(train_loader))
    print(len(val_loader))

    if args.cuda:
        weight = weight.cuda()

    loss_func = CrossEntropyLoss2d(weight)

    print(type(loss_func))
    savedir = f'./save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write(
                "Epoch\tTrain-loss\tTest-loss\tTrain-cpa1\tTest-cpa1\tTrain-mPA\tTest-mPA\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))
    # TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    # optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),
                     eps=1e-08, weight_decay=1e-4)  # scheduler 2
    def lambda1(epoch): return pow(
        (1-((epoch)/args.num_epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda1)  # scheduler 2
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001) ## scheduler 3 lxl
    scaler = GradScaler()

    start_epoch = 1
    if args.resume:  
        # Must load weights, optimizer, epoch and best value.
        if args.resumeencoder:  
            if enc:
                print("restart program from encoder")
                filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
                assert os.path.exists(
                    filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
                checkpoint = torch.load(filenameCheckpoint)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scaler.load_state_dict(checkpoint['scaler'])
                scheduler.load_state_dict(checkpoint['lr_scheduler'])
                best_acc = checkpoint['best_acc']
                print("=> Loaded checkpoint at epoch {})".format(
                    checkpoint['epoch']))
            else:
                print("decoder from epoch 1 with encoder_best.model")

        else:
            if enc:
                print("from decoder resume over encoder_train")
                start_epoch = args.num_epochs + 2
                print("start epoch is num_epochs:", (start_epoch - 2))
            else:
                filenameCheckpoint4 = savedir + '/checkpoint.pth.tar'
                assert os.path.exists(
                    filenameCheckpoint4), "Error: resume option was used but checkpoint was not found in folder"
                checkpoint4 = torch.load(filenameCheckpoint4)
                start_epoch = checkpoint4['epoch']
                model.load_state_dict(checkpoint4['state_dict'])
                optimizer.load_state_dict(checkpoint4['optimizer'])
                scheduler.load_state_dict(checkpoint4['lr_scheduler'])
                scaler.load_state_dict(checkpoint4['scaler']) 
                best_acc = checkpoint4['best_acc']
                print("=> Frome decoder Loaded checkpoint at epoch {})".format(
                    checkpoint4['epoch']))
        
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")
        epoch_loss = []
        time_train = []
        # time_tmp = []
        doIouTrain = args.iouTrain
        doIouVal = args.iouVal

        if (doIouTrain):
            # iouEvalTrain = iouEval(NUM_CLASSES)
            metricTrain = SegmentationMetric(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(train_loader):

            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()


            inputs = Variable(images)
            targets = Variable(labels)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs, only_encode=enc)
                loss = loss_func(outputs, targets[:, 0])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)
            imgPredict = outputs.max(1)[1].unsqueeze(1)

            start_time2 = time.time()
            if (doIouTrain):
                hist = metricTrain.addBatch(imgPredict, labels)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        cpa1Train =0 
        if (doIouTrain):
            cpaTrain = metricTrain.classPixelAccuracy().data.cpu().numpy().tolist()
            cpa1Train = cpaTrain[1]
            mpaTrain = sum(cpaTrain)/3
            print('cPA is :', cpaTrain) 
            print('mpa is :', mpaTrain) 

        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        epoch_loss2_val = []
        time_val = []

        if (doIouVal):
            metricVal = SegmentationMetric(NUM_CLASSES)
        with torch.no_grad():
            for step, (images, labels) in enumerate(val_loader):
                start_time = time.time()
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                inputs = Variable(images)
                targets = Variable(labels)

                outputs = model(inputs, only_encode=enc)
                loss = loss_func(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())
                time_val.append(time.time() - start_time)

                if (doIouVal):
                    metricVal.addBatch(outputs.max(1)[1].unsqueeze(1),targets)
                if args.steps_loss > 0 and step % args.steps_loss == 0:
                    average = sum(epoch_loss_val) / len(epoch_loss_val)
          
                    print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})',
                          "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

            average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        scheduler.step()  # scheduler 2
        cpa1Val =0
        cpaVal = metricVal.classPixelAccuracy().data.cpu().numpy().tolist()
        cpa1Val = cpaVal[1]
        mpaVal = sum(cpaVal)/3
        print('cPA is :', cpaVal)  # 列表
        print('mpa is :', mpaVal)  # 列表

        if cpa1Val == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = cpa1Val
        
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'

        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'scaler': scaler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()
        }, is_best, filenameCheckpoint, filenameBest)

        # SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'

        if args.epochs_save > 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write(
                        "Best epoch is %d, with Val-cpa1= %.4f" % (epoch, cpa1Val))
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write(
                        "Best epoch is %d, with Val-cpa1= %.4f" % (epoch, cpa1Val))
     
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, cpa1Train, cpa1Val, mpaTrain,mpaVal,usedLr ))
    if enc:
        filenameCheckpoint1 = savedir + '/model_best_enc.pth.tar'
        assert os.path.exists(
            filenameCheckpoint1), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint1 = torch.load(filenameCheckpoint1)
        model.load_state_dict(checkpoint1['state_dict'])
        print(
            "=> From best checkpoint at epoch {} - 1)".format(checkpoint1['epoch']))
        return(model)  # return model (convenience for encoder-decoder training)
    else:
        return(model)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)


def main(args):

    datamode = args.datamode
    if(datamode==0 or datamode==1 or datamode==2):
        NUM_CHANNELS =1
    elif(datamode==3 or datamode==4 or datamode==5):
        NUM_CHANNELS =2
    elif(datamode==6 or datamode==7):
        NUM_CHANNELS =3

    savedir = f'./save/{args.savedir}'
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    assert os.path.exists("train/" + args.model +
                          ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CHANNELS,NUM_CLASSES)
    copyfile("train/" + args.model + ".py", savedir + '/' + args.model + ".py")
    print(torch.cuda.is_available())
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.state:
        # if args.state is provided then load this state for training
        # Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """
        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            return model

        # print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder
    
    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """

    #train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True)  # Train encoder
    # CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0.
    # We must reinit decoder weights or reload network passing only encoder in order to train decoder
    else:
        print("==========  TRAINING ===========")
        model = model_file.Net(NUM_CHANNELS,NUM_CLASSES)
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        # When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
        model = train(args, model, False)  # Train decoder
        print("========== TRAINING FINISHED ===========")
    # print("========== DECODER TRAINING ===========")

    # if (not args.state):
    #     if args.pretrainedEncoder:
    #         print("Loading encoder pretrained in imagenet")
    #         from erfnet_imagenet import ERFNet as ERFNet_imagenet
    #         pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
    #         pretrainedEnc.load_state_dict(torch.load(
    #             args.pretrainedEncoder)['state_dict'])
    #         pretrainedEnc = next(pretrainedEnc.children()).features.encoder
    #         if (not args.cuda):
    #             # because loaded encoder is probably saved in cuda
    #             pretrainedEnc = pretrainedEnc.cpu()
    #     else:
    #         print("next")
    #         pretrainedEnc = next(model.children()).encoder
    #     # Add decoder to encoder
    #     model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)
    #     if args.cuda:
    #         model = torch.nn.DataParallel(model).cuda()
    #     # When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    # model = train(args, model, False)  # Train decoder
    # print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="erfnet_avgpool")
    parser.add_argument('--state')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument(
        '--datadir', default="/home/luoxiaolong/DataSet/CalibrateData/20210822/binary-11#-499V-20210822/")
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=0)
    # You can use this value to save model every X epochs
    parser.add_argument('--epochs-save', type=int, default=1)
    parser.add_argument('--savedir', default="quarter/modugray_3Class_AvgPool_pmg_modu_new")
    parser.add_argument('--decoder', action='store_true',default=True)
    parser.add_argument('--pretrainedEncoder')
    parser.add_argument('--iouTrain', action='store_true', default=True)
    parser.add_argument('--iouVal', action='store_true', default=True)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resumeencoder', default=False, action='store_true')
    parser.add_argument('--datamode', type=int, default=5) 
    main(parser.parse_args())
