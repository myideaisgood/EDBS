from tripletloss import TripletLoss
import torch
from torch.utils.data import DataLoader

import numpy as np
import logging
import os
from time import time

from config import parse_args
from model_classifier import Classifier
from utils.data_loaders import ClassificationDataset_Triplet
from utils.helpers import *
from utils.average_meter import AverageMeter

def train_classifier(args, subset):

    # Design Parameters
    LAMBDA_CE = args.lambda_ce

    # Model Parameters
    HIDDEN_UNIT = args.hidden_unit

    # Session Parameters
    GPU_NUM = args.gpu_num
    BATCH_SIZE = args.batch_size_c
    EPOCHS = args.epochs_c
    LR = args.lr_c
    PRINT_EVERY = args.print_every

    # Directory Parameters
    EXP_NAME = args.experiment_name
    EXP_DIR = 'experiments/' + EXP_NAME
    CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
    LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
    WEIGHTS_CLASSIFIER = args.weights_classifier

    # Check if directory does not exist
    create_path('experiments/')
    create_path(EXP_DIR)
    create_path(CKPT_DIR)
    create_path(LOG_DIR)
    create_path(os.path.join(LOG_DIR, 'train'))
    create_path(os.path.join(LOG_DIR, 'test'))

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up Dataset
    dataset = ClassificationDataset_Triplet(args, subset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
        drop_last=True
    )

    cls_num = dataloader.dataset.get_cls_num()

    # Set up model / optimizer
    model = Classifier(in_channels=3, out_features=cls_num, hidden_size=HIDDEN_UNIT)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Set up Loss Functions
    triplet_criterion = TripletLoss()
    ce_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Load the pretrained model if exists
    init_epoch = 0
    best_metrics = 0.0

    if os.path.exists(os.path.join(CKPT_DIR, subset + '_' + WEIGHTS_CLASSIFIER)):
        init_epoch = EPOCHS-1        
        logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, subset + '_' + WEIGHTS_CLASSIFIER))
        checkpoint = torch.load(os.path.join(CKPT_DIR, subset + '_' + WEIGHTS_CLASSIFIER))
        best_metrics = checkpoint['best_metrics_classifier']
        model.load_state_dict(checkpoint['model_classifier'])
        logging.info('Recover completed. Current epoch = #%d, best metrics = %.3f' % (init_epoch,best_metrics))

    for epoch_idx in range(init_epoch+1, EPOCHS):

        model.train()

        loss_avg = AverageMeter()
        ce_loss_avg = AverageMeter()
        triplet_loss_avg = AverageMeter()
        acc_avg = AverageMeter()

        for _, data in enumerate(dataloader):

            anc, pos, neg, label, _ = data

            anc = anc.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anc_label = label[0].to(device)
            pos_label = label[1].to(device)
            neg_label = label[2].to(device)

            opt.zero_grad()

            anc_f, anc_logit = model(anc)
            pos_f, pos_logit = model(pos)
            neg_f, neg_logit = model(neg)

            triplet_loss = triplet_criterion(anc_f, pos_f, neg_f)
            anc_ce_loss = ce_criterion(anc_logit, anc_label)
            pos_ce_loss = ce_criterion(pos_logit, pos_label)
            neg_ce_loss = ce_criterion(neg_logit, neg_label)
            
            ce_loss = LAMBDA_CE * (anc_ce_loss + pos_ce_loss + neg_ce_loss)
            loss = triplet_loss + ce_loss

            loss.backward()
            opt.step()

            anc_acc = (anc_logit.argmax(dim=1) == anc_label).sum().item() / BATCH_SIZE
            pos_acc = (pos_logit.argmax(dim=1) == pos_label).sum().item() / BATCH_SIZE
            neg_acc = (neg_logit.argmax(dim=1) == neg_label).sum().item() / BATCH_SIZE

            acc = (anc_acc + pos_acc + neg_acc) / 3.0

            loss_avg.update(loss.item())
            ce_loss_avg.update(ce_loss.item())
            triplet_loss_avg.update(triplet_loss.item())
            acc_avg.update(acc)

        if epoch_idx % PRINT_EVERY == 0:
            logging.info('[Epoch %d/%d] Loss = %.4f  Triplet Loss = %.4f   CE Loss = %.4f   Acc : %.3f' %(epoch_idx, EPOCHS, loss_avg.avg(), triplet_loss_avg.avg(), ce_loss_avg.avg(), acc_avg.avg()))

            if acc_avg.avg() > best_metrics:
                output_path = os.path.join(CKPT_DIR, subset + '_' + WEIGHTS_CLASSIFIER)
                best_metrics = acc_avg.avg()

                torch.save({
                    'best_metrics_classifier': best_metrics,
                    'model_classifier': model.state_dict()
                }, output_path)
                logging.info('Saved checkpoint to %s ... ' % output_path)

            logging.info('Best acc = %.4f' %(best_metrics))               

    logging.shutdown()

if __name__ == '__main__':
    args = parse_args()
    train_classifier(args, 'train')