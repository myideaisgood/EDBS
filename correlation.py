import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import os
import random
import cv2
import tqdm
from time import time

from config import parse_args
from model_classifier import Classifier
from utils.data_loaders import CorrelationDataset
from utils.helpers import *

def get_correlation(args, subset):
    # Model Parameters
    HIDDEN_UNIT = args.hidden_unit

    # Session Parameters
    GPU_NUM = args.gpu_num
    BATCH_SIZE = args.batch_size_c

    # Directory Parameters
    EXP_NAME = args.experiment_name
    EXP_DIR = 'experiments/' + EXP_NAME
    CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
    WEIGHTS_CLASSIFIER = args.weights_classifier

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up Dataset
    dataset = CorrelationDataset(args, subset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
        drop_last=False
    )

    cls_num = dataloader.dataset.get_cls_num()

    # Set up model
    model = Classifier(in_channels=3, out_features=cls_num, hidden_size=HIDDEN_UNIT)
    model.to(device)

    print('Recovering from %s ...' % os.path.join(CKPT_DIR, subset + '_' + WEIGHTS_CLASSIFIER))
    checkpoint = torch.load(os.path.join(CKPT_DIR, subset + '_' + WEIGHTS_CLASSIFIER))
    best_metrics = checkpoint['best_metrics_classifier']
    model.load_state_dict(checkpoint['model_classifier'])
    print('Recover completed. Best metrics = %.3f' % (best_metrics))
    
    class_features = torch.ones(cls_num, HIDDEN_UNIT)

    for class_idx in tqdm.tqdm(range(cls_num)):

        dataloader.dataset.set_cls(class_idx)

        for idx, data in enumerate(dataloader):
   
            imgs, _ = data

            imgs = imgs.to(device)

            features, _ = model(imgs)
            features = torch.sum(features, dim=0)

            if idx == 0:
                cls_feature = features
            else:
                cls_feature += features

        cls_feature = cls_feature / len(dataloader)
        cls_feature = F.normalize(torch.unsqueeze(cls_feature, dim=0), p=2)[0]
        cls_feature = cls_feature.detach().cpu()
        class_features[class_idx] = cls_feature

    correlation = torch.matmul(class_features, torch.transpose(class_features, 0, 1))
    correlation = (correlation+1)/2
    correlation_sort = torch.argsort(correlation, dim=1, descending=True)

    return correlation.numpy(), correlation_sort.numpy()

def save_correlation(args, subset, correlation_sort):

    BATCH_SIZE = 4
    SAVE_NUM = 5    
    cls_num = len(correlation_sort)

    for i in range(cls_num):
        print(correlation_sort[i])

    # Set up Dataset
    dataset = CorrelationDataset(args, subset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
        drop_last=False
    )    

    for corr_idx in range(cls_num):
        pos_idx = correlation_sort[corr_idx][:SAVE_NUM]
        neg_idx = correlation_sort[corr_idx][cls_num-SAVE_NUM:]

        # Save Correlation Image
        dataloader.dataset.set_cls(corr_idx)

        img, _ = next(iter(dataloader))

        img = img.permute(0,2,3,1).numpy()
        img = img.astype(np.uint8)

        for b_idx in range(BATCH_SIZE):
            cur_img = img[b_idx]
            
            if b_idx == 0:
                save_img = cur_img
            else:
                save_img = np.concatenate([save_img, cur_img], axis=0)

        # Save Positive Image
        for p_idx in pos_idx:
            dataloader.dataset.set_cls(p_idx)

            img, _ = next(iter(dataloader))

            img = img.permute(0,2,3,1).numpy()
            img = img.astype(np.uint8)
        
            for b_idx in range(BATCH_SIZE):
                cur_img = img[b_idx]

                if b_idx == 0:
                    pos_img = cur_img
                else:
                    pos_img = np.concatenate([pos_img, cur_img], axis=0)
            
            save_img = np.concatenate([save_img, pos_img], axis=1)
        
        # Save Negative Image
        for n_idx in neg_idx:
            dataloader.dataset.set_cls(n_idx)

            img, _ = next(iter(dataloader))

            img = img.permute(0,2,3,1).numpy()
            img = img.astype(np.uint8)
        
            for b_idx in range(BATCH_SIZE):
                cur_img = img[b_idx]

                if b_idx == 0:
                    neg_img = cur_img
                else:
                    neg_img = np.concatenate([neg_img, cur_img], axis=0)
            
            save_img = np.concatenate([save_img, neg_img], axis=1)        

        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        img_name = 'playground/' + str(corr_idx).zfill(3) + '.jpg'
        
        cv2.imwrite(img_name, save_img)


def check_correlation(correlation):

    cls_num = len(correlation)

    bin_num = 10

    np_hist = np.zeros(bin_num)

    for i in range(cls_num):
        for j in range(cls_num):
            if i < j:
                cur_cor = correlation[i][j]
                for k in range(bin_num):
                    if cur_cor < (1/bin_num)*(k+1):
                        np_hist[k] += 1
                        break
    
    np_hist /= np.sum(np_hist)

    print("======= Correlation Histogram =======")
    for k in range(bin_num):
        print("%.2f : %.3f" % ((1/bin_num)*(k+1), np_hist[k]))

def check_difficulty(correlation):

    SAMPLE_NUM = 5000
    cls_num = len(correlation)

    bin_num = 20
    np_hist = np.zeros(bin_num)

    for _ in range(SAMPLE_NUM):
        cls_idx = random.sample(list(np.arange(cls_num)), 5)
        cnt = 0
        diff = 0

        for i in range(5):
            for j in range(5):
                if i < j :
                    diff += correlation[cls_idx[i]][cls_idx[j]]
                    cnt += 1
        
        diff /= cnt

        for k in range(bin_num):
            if diff < (1/bin_num) * (k+1):
                np_hist[k] += 1
                break

    np_hist /= np.sum(np_hist)

    print("======= Difficulty Histogram =======")
    for k in range(bin_num):
        print("%.2f : %.3f" % ((1/bin_num)*(k+1), np_hist[k]))

if __name__ == '__main__':
    
    args = parse_args()
    subset = 'train'

    correlation, correlation_sort = get_correlation(args, subset)
    #save_correlation(args, subset, correlation_sort)
    check_correlation(correlation)
    check_difficulty(correlation)