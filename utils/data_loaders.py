from __future__ import print_function
import csv
import glob
import os
import cv2
from shutil import copy2

from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CorrelationDataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.subset = subset
        self.classes = []
        self.images = {}
        self.img_size = 84
        
        data_dir = os.path.join(args.data_dir, args.dataset, subset)

        for file in os.listdir(data_dir):
            self.classes.append(file)

        self.classes = sorted(self.classes)

        for cls in self.classes:
            self.images[cls] = []
            for file in os.listdir(os.path.join(data_dir, cls)):
                if file.endswith('.jpg'):
                    self.images[cls].append(file)
            self.images[cls] = sorted(self.images[cls])

        self.cls_num = len(self.classes)
        self.img_num = len(self.images[self.classes[0]])
        self.cls_idx = 0

    def __getitem__(self, idx):

        DATA_DIR = os.path.join(self.args.data_dir, self.args.dataset, self.subset)

        cls_name = self.classes[self.cls_idx]

        img_name = self.images[cls_name][idx]

        img_dir = os.path.join(DATA_DIR, cls_name, img_name)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).permute(2,0,1).float()

        return img, cls_name

    def __len__(self):
        return self.img_num

    def get_cls_num(self):
        return self.cls_num

    def get_cls_name(self):
        return self.classes

    def set_cls(self, class_idx):
        self.cls_idx = class_idx

class ClassificationDataset_Triplet(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.subset = subset
        self.classes = []
        self.images = {}
        self.img_size = 84
        
        data_dir = os.path.join(args.data_dir, args.dataset, subset)

        for file in os.listdir(data_dir):
            self.classes.append(file)

        self.classes = sorted(self.classes)

        for cls in self.classes:
            self.images[cls] = []
            for file in os.listdir(os.path.join(data_dir, cls)):
                if file.endswith('.jpg'):
                    self.images[cls].append(file)
            self.images[cls] = sorted(self.images[cls])

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.cls_num = len(self.classes)
        self.img_num = len(self.images[self.classes[0]])

    def get_cls_num(self):
        return self.cls_num

    def __getitem__(self, idx):

        cls_idx = random.sample(list(np.arange(self.cls_num)), 2)
        img_idx = random.sample(list(np.arange(self.img_num)), 3)

        pos_cls_idx = cls_idx[0]
        neg_cls_idx = cls_idx[1]

        anc_img_idx = img_idx[0]
        pos_img_idx = img_idx[1]
        neg_img_idx = img_idx[2]

        anc_img_name = self.images[self.classes[pos_cls_idx]][anc_img_idx]
        pos_img_name = self.images[self.classes[pos_cls_idx]][pos_img_idx]
        neg_img_name = self.images[self.classes[neg_cls_idx]][neg_img_idx]

        anc_img = self.read_img(self.classes[pos_cls_idx], anc_img_name)
        pos_img = self.read_img(self.classes[pos_cls_idx], pos_img_name)
        neg_img = self.read_img(self.classes[neg_cls_idx], neg_img_name)

        label = [pos_cls_idx, pos_cls_idx, neg_cls_idx]
        cls_name = [self.classes[pos_cls_idx], self.classes[pos_cls_idx], self.classes[neg_cls_idx]]

        return anc_img, pos_img, neg_img, label, cls_name

    def read_img(self, cls_name, img_name):

        DATA_DIR = os.path.join(self.args.data_dir, self.args.dataset, self.subset)

        img_dir = os.path.join(DATA_DIR, cls_name, img_name)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).permute(2,0,1).float()

        return img

    def __len__(self):
        return self.cls_num * self.img_num

def show_samples_triplet(args, subset):

    BATCH_SIZE = 4
    IMG_SIZE = 84
    PLAYGROUND_DIR = 'playground/'

    dataset = ClassificationDataset_Triplet(args, subset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader, 0):

        anc, pos, neg, label, cls_name = data

        anc = anc.permute(0,2,3,1).numpy()
        pos = pos.permute(0,2,3,1).numpy()
        neg = neg.permute(0,2,3,1).numpy()

        anc_label = label[0].numpy()
        pos_label = label[1].numpy()
        neg_label = label[2].numpy()

        anc_cls_name = cls_name[0]
        pos_cls_name = cls_name[1]
        neg_cls_name = cls_name[2]

        for b_idx in range(BATCH_SIZE):
            cur_anc = anc[b_idx]
            cur_pos = pos[b_idx]
            cur_neg = neg[b_idx]

            cur_anc_label = anc_label[b_idx]
            cur_pos_label = pos_label[b_idx]
            cur_neg_label = neg_label[b_idx]

            cur_anc_name = anc_cls_name[b_idx]
            cur_pos_name = pos_cls_name[b_idx]
            cur_neg_name = neg_cls_name[b_idx]
            
            cur_anc = cv2.cvtColor(cur_anc, cv2.COLOR_RGB2BGR)
            cur_pos = cv2.cvtColor(cur_pos, cv2.COLOR_RGB2BGR)
            cur_neg = cv2.cvtColor(cur_neg, cv2.COLOR_RGB2BGR)

            anc_text = str(cur_anc_label) + '_' + cur_anc_name
            pos_text = str(cur_pos_label) + '_' + cur_pos_name
            neg_text = str(cur_neg_label) + '_' + cur_neg_name

            cv2.putText(cur_anc, anc_text, (10,10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(cur_pos, pos_text, (10,10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(cur_neg, neg_text, (10,10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)

            cur_img = np.concatenate([cur_anc, cur_pos, cur_neg], axis=1)

            if b_idx == 0:
                show_img = cur_img
            else:
                show_img = np.concatenate([show_img, cur_img], axis=0)

        img_name = PLAYGROUND_DIR +  str(i).zfill(3) + '.jpg'
        cv2.imwrite(img_name, show_img)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.subset = subset
        self.classes = []
        self.images = {}
        self.img_size = 84
        
        data_dir = os.path.join(args.data_dir, args.dataset, subset)

        for file in os.listdir(data_dir):
            self.classes.append(file)

        self.classes = sorted(self.classes)

        for cls in self.classes:
            self.images[cls] = []
            for file in os.listdir(os.path.join(data_dir, cls)):
                if file.endswith('.jpg'):
                    self.images[cls].append(file)
            self.images[cls] = sorted(self.images[cls])

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.cls_num = len(self.classes)
        self.img_num = len(self.images[self.classes[0]])

    def __getitem__(self, idx):

        DATA_DIR = os.path.join(self.args.data_dir, self.args.dataset, self.subset)

        cls_idx = int(idx/self.img_num)
        img_idx = idx - self.img_num * cls_idx

        cls_name = self.classes[cls_idx]
        img_name = self.images[cls_name][img_idx]

        img_dir = os.path.join(DATA_DIR, cls_name, img_name)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = cls_idx

        img = torch.from_numpy(img).permute(2,0,1).float()

        return img, label, cls_name

    def __len__(self):
        return self.cls_num * self.img_num

def show_samples_classification(args, subset):

    BATCH_SIZE = args.batch_size
    IMG_SIZE = 84
    PLAYGROUND_DIR = 'playground/'

    dataset = ClassificationDataset(args, subset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader, 0):

        img, label, cls_name = data

        img = img.permute(0,2,3,1).numpy()
        label = label.numpy()

        for b_idx in range(BATCH_SIZE):
            cur_img = img[b_idx]
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR)
            cur_label = label[b_idx]
            cur_cls_name = cls_name[b_idx]
            cur_text = str(cur_label) + '_' + cur_cls_name
            cv2.putText(cur_img, cur_text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            if b_idx == 0:
                show_img = cur_img
            else:
                show_img = np.concatenate([show_img, cur_img], axis=1)

        img_name = PLAYGROUND_DIR +  str(i).zfill(3) + '.jpg'
        cv2.imwrite(img_name, show_img)

class Dataset_Difficulty(torch.utils.data.Dataset):
    def __init__(self, args, subset, correlation):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.subset = subset
        self.classes = []
        self.images = {}
        self.img_size = 84
        
        data_dir = os.path.join(args.data_dir, args.dataset, subset)

        for file in os.listdir(data_dir):
            self.classes.append(file)

        for cls in self.classes:
            self.images[cls] = []
            for file in os.listdir(os.path.join(data_dir, cls)):
                if file.endswith('.jpg'):
                    self.images[cls].append(file)

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.cls_num = len(self.classes)

        self.correlation = correlation
        self.diff_th = 0
        self.cls_hist = np.zeros(self.cls_num)
        self.prob = 0.7

    def set_prob(self, prob):
        self.prob = prob

    def set_diff_th(self, th):
        self.diff_th = th

    def get_task_diff(self, cls_idx):
        
        N_WAY = self.args.N_way
        cnt = 0
        diff = 0

        for i in range(N_WAY):
            for j in range(N_WAY):
                if i < j :
                    diff += self.correlation[cls_idx[i]][cls_idx[j]]
                    cnt += 1
        
        diff /= cnt

        return diff

    def get_cls_hist(self):
        return self.cls_hist / np.sum(self.cls_hist)

    def reset_cls_hist(self):
        self.cls_hist = np.zeros(self.cls_num)

    def get_cls_num(self):
        return self.cls_num

    def __getitem__(self, idx):

        N_WAY = self.args.N_way
        K_SHOT = self.args.K_shot
        QUERY_NUM = self.args.query_num
        IMG_SIZE = self.img_size
        DATA_DIR = os.path.join(self.args.data_dir, self.args.dataset, self.subset)

        if random.random() > self.prob : 
            diff = 0
            while diff <= self.diff_th:
                cls_idx = random.sample(list(np.arange(self.cls_num)), N_WAY)
                diff = self.get_task_diff(cls_idx)
        else:
            cls_idx = random.sample(list(np.arange(self.cls_num)), N_WAY)
            diff = self.get_task_diff(cls_idx)

        for c_idx in cls_idx:
            self.cls_hist[c_idx] += 1

        # Sample classes
        sample_cls = []
        for c_idx in cls_idx:
            sample_cls.append(self.classes[c_idx])

        # Sample images from classes
        sample_img = {}
        for cls in sample_cls:
            img_dir = self.images[cls]
            imgs = random.sample(img_dir, K_SHOT + QUERY_NUM)
            sample_img[cls] = imgs

        # Support / Query
        support_x = np.zeros([N_WAY, K_SHOT, IMG_SIZE, IMG_SIZE, 3]).astype(np.uint8)
        support_y = np.zeros([N_WAY, K_SHOT]).astype(int)
        query_x = np.zeros([N_WAY, QUERY_NUM, IMG_SIZE, IMG_SIZE, 3]).astype(np.uint8)
        query_y = np.zeros([N_WAY, QUERY_NUM]).astype(int)

        # Support / Query
        for cls_idx, cls in enumerate(sample_cls, 0):
            img_names = sample_img[cls]
            for img_idx, img_name in enumerate(img_names, 0):
                img_dir = os.path.join(DATA_DIR, cls, img_name)
                img = cv2.imread(img_dir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if img_idx < K_SHOT:
                    support_x[cls_idx, img_idx] = img
                    support_y[cls_idx, img_idx] = cls_idx
                else:
                    query_x[cls_idx, img_idx-K_SHOT] = img
                    query_y[cls_idx, img_idx-K_SHOT] = cls_idx

        # Shuffle Support / Query
        support_x = np.reshape(support_x, [N_WAY*K_SHOT, IMG_SIZE, IMG_SIZE, 3])
        support_y = np.reshape(support_y, [N_WAY*K_SHOT])
        query_x = np.reshape(query_x, [N_WAY*QUERY_NUM, IMG_SIZE, IMG_SIZE, 3])
        query_y = np.reshape(query_y, [N_WAY*QUERY_NUM])

        sup_idx = np.arange(N_WAY*K_SHOT)
        query_idx = np.arange(N_WAY*QUERY_NUM)  
        np.random.shuffle(sup_idx)
        np.random.shuffle(query_idx)

        # Support x : [N_WAY * K_SHOT, IMG_SIZE, IMG_SIZE, 3]
        # Support y : [N_WAY * K_SHOT]
        # Query x : [N_WAY * QUERY_NUM, IMG_SIZE, IMG_SIZE, 3]
        # Query y : [N_WAY * QUERY_NUM]
        support_x = support_x[sup_idx]
        support_y = support_y[sup_idx]
        query_x = query_x[query_idx]
        query_y = query_y[query_idx]

        support_x = ((torch.from_numpy(support_x)).permute(0,3,1,2)).float()
        support_y = torch.from_numpy(support_y)
        query_x = (torch.from_numpy(query_x).permute(0,3,1,2)).float()
        query_y = torch.from_numpy(query_y)

        return support_x, support_y, query_x, query_y, diff

    def __len__(self):
        return 10000000

def show_samples_difficulty(args, subset, correlation):

    TASK_NUM = args.task_num
    N_WAY = args.N_way
    K_SHOT = args.K_shot
    QUERY_NUM = args.query_num
    IMG_SIZE = 84
    PLAYGROUND_DIR = 'playground/'

    dataset = Dataset_Difficulty(args, subset, correlation)
    dataloader = DataLoader(dataset, batch_size=TASK_NUM, shuffle=False, num_workers=0)

    dataloader.dataset.set_diff_th(0.60)

    for i, data in enumerate(dataloader, 0):

        support_x = (support_x.permute(0,1,3,4,2)).numpy()
        support_y = support_y.numpy()
        query_x = (query_x.permute(0,1,3,4,2)).numpy()
        query_y = query_y.numpy()
        diff = diff.numpy()

        BATCH_SIZE = len(support_x)

        for b_idx in range(BATCH_SIZE):
            cur_sup_x = support_x[b_idx]
            cur_sup_y = support_y[b_idx]
            cur_que_x = query_x[b_idx]
            cur_que_y = query_y[b_idx]
            cur_diff = round(diff[b_idx],3)

            show_sup = np.zeros([N_WAY*IMG_SIZE, K_SHOT*IMG_SIZE, 3]).astype(np.uint8)

            for row_idx in range(N_WAY):
                for col_idx in range(K_SHOT):
                    idx = K_SHOT * row_idx + col_idx
                    cur_img = cv2.cvtColor(cur_sup_x[idx], cv2.COLOR_RGB2BGR)
                    cur_label = cur_sup_y[idx]                    
                    show_sup[IMG_SIZE*row_idx:IMG_SIZE*(row_idx+1), IMG_SIZE*col_idx:IMG_SIZE*(col_idx+1)] = cur_img

            show_query = np.zeros([N_WAY*IMG_SIZE, QUERY_NUM*IMG_SIZE, 3]).astype(np.uint8)

            for row_idx in range(N_WAY):
                for col_idx in range(QUERY_NUM):
                    idx = QUERY_NUM * row_idx + col_idx
                    cur_img = cv2.cvtColor(cur_que_x[idx], cv2.COLOR_RGB2BGR)
                    cur_label = cur_que_y[idx]
                    show_query[IMG_SIZE*row_idx:IMG_SIZE*(row_idx+1), IMG_SIZE*col_idx:IMG_SIZE*(col_idx+1)] = cur_img

            cv2.putText(show_sup, str(cur_diff), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)            
            img_name = PLAYGROUND_DIR +  str(i).zfill(3) + '_' + str(b_idx) + '_'

            cv2.imwrite(img_name + 'support.jpg', show_sup)
            cv2.imwrite(img_name + 'query.jpg', show_query)   


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        assert (subset == 'train') or (subset == 'val') or (subset == 'test')

        self.args = args
        self.subset = subset
        self.classes = []
        self.images = {}
        self.img_size = 84
        
        data_dir = os.path.join(args.data_dir, args.dataset, subset)

        for file in os.listdir(data_dir):
            self.classes.append(file)

        for cls in self.classes:
            self.images[cls] = []
            for file in os.listdir(os.path.join(data_dir, cls)):
                if file.endswith('.jpg'):
                    self.images[cls].append(file)

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __getitem__(self, idx):

        N_WAY = self.args.N_way
        K_SHOT = self.args.K_shot
        QUERY_NUM = self.args.query_num
        IMG_SIZE = self.img_size
        DATA_DIR = os.path.join(self.args.data_dir, self.args.dataset, self.subset)

        # Sample classes
        sample_cls = random.sample(self.classes, N_WAY)

        # Sample images from classes
        sample_img = {}
        for cls in sample_cls:
            img_dir = self.images[cls]
            imgs = random.sample(img_dir, K_SHOT + QUERY_NUM)
            sample_img[cls] = imgs

        # Support / Query
        support_x = np.zeros([N_WAY, K_SHOT, IMG_SIZE, IMG_SIZE, 3]).astype(np.uint8)
        support_y = np.zeros([N_WAY, K_SHOT]).astype(int)
        query_x = np.zeros([N_WAY, QUERY_NUM, IMG_SIZE, IMG_SIZE, 3]).astype(np.uint8)
        query_y = np.zeros([N_WAY, QUERY_NUM]).astype(int)

        # Support / Query
        for cls_idx, cls in enumerate(sample_cls, 0):
            img_names = sample_img[cls]
            for img_idx, img_name in enumerate(img_names, 0):
                img_dir = os.path.join(DATA_DIR, cls, img_name)
                img = cv2.imread(img_dir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if img_idx < K_SHOT:
                    support_x[cls_idx, img_idx] = img
                    support_y[cls_idx, img_idx] = cls_idx
                else:
                    query_x[cls_idx, img_idx-K_SHOT] = img
                    query_y[cls_idx, img_idx-K_SHOT] = cls_idx

        # Shuffle Support / Query
        support_x = np.reshape(support_x, [N_WAY*K_SHOT, IMG_SIZE, IMG_SIZE, 3])
        support_y = np.reshape(support_y, [N_WAY*K_SHOT])
        query_x = np.reshape(query_x, [N_WAY*QUERY_NUM, IMG_SIZE, IMG_SIZE, 3])
        query_y = np.reshape(query_y, [N_WAY*QUERY_NUM])

        sup_idx = np.arange(N_WAY*K_SHOT)
        query_idx = np.arange(N_WAY*QUERY_NUM)  
        np.random.shuffle(sup_idx)
        np.random.shuffle(query_idx)

        # Support x : [N_WAY * K_SHOT, IMG_SIZE, IMG_SIZE, 3]
        # Support y : [N_WAY * K_SHOT]
        # Query x : [N_WAY * QUERY_NUM, IMG_SIZE, IMG_SIZE, 3]
        # Query y : [N_WAY * QUERY_NUM]
        support_x = support_x[sup_idx]
        support_y = support_y[sup_idx]
        query_x = query_x[query_idx]
        query_y = query_y[query_idx]

        support_x = ((torch.from_numpy(support_x)).permute(0,3,1,2)).float()
        support_y = torch.from_numpy(support_y)
        query_x = (torch.from_numpy(query_x).permute(0,3,1,2)).float()
        query_y = torch.from_numpy(query_y)

        return support_x, support_y, query_x, query_y

    def __len__(self):
        return 10000

def proc_images(args):
    DATA_DIR = args.data_dir
    DATASET = args.dataset

    path_to_images = os.path.join(DATA_DIR, DATASET, 'images/')

    all_images = glob.glob(path_to_images + '*')

    # Resize images
    for i, image_file in enumerate(all_images):
        im = Image.open(image_file)
        im = im.resize((84, 84), resample=Image.LANCZOS)
        im.save(image_file)
        if i % 500 == 0:
            print(i)

    # Put in correct directory
    for datatype in ['train', 'val', 'test']:

        dir = os.path.join(DATA_DIR, DATASET, datatype)

        os.system('mkdir ' + dir)

        with open(dir + '.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            last_label = ''
            for i, row in enumerate(reader):
                if i == 0:  # skip the headers
                    continue
                label = row[1]
                image_name = row[0]
                if label != last_label:
                    cur_dir = dir + '/' + label + '/'
                    if not os.path.exists(cur_dir):
                        os.mkdir(cur_dir)
                    last_label = label
                copy2(path_to_images + image_name, cur_dir)

def show_samples(args, subset):

    TASK_NUM = args.task_num
    N_WAY = args.N_way
    K_SHOT = args.K_shot
    QUERY_NUM = args.query_num
    IMG_SIZE = 84
    PLAYGROUND_DIR = 'playground/'

    dataset = Dataset(args, subset)
    dataloader = DataLoader(dataset, batch_size=TASK_NUM, shuffle=False, num_workers=0)

    for i, data in enumerate(dataloader, 0):

        support_x, support_y, query_x, query_y = data

        support_x = (support_x.permute(0,1,3,4,2)).numpy()
        support_y = support_y.numpy()
        query_x = (query_x.permute(0,1,3,4,2)).numpy()
        query_y = query_y.numpy()

        BATCH_SIZE = len(support_x)

        for b_idx in range(BATCH_SIZE):
            cur_sup_x = support_x[b_idx]
            cur_sup_y = support_y[b_idx]
            cur_que_x = query_x[b_idx]
            cur_que_y = query_y[b_idx]

            show_sup = np.zeros([N_WAY*IMG_SIZE, K_SHOT*IMG_SIZE, 3]).astype(np.uint8)

            for row_idx in range(N_WAY):
                for col_idx in range(K_SHOT):
                    idx = K_SHOT * row_idx + col_idx
                    cur_img = cv2.cvtColor(cur_sup_x[idx], cv2.COLOR_RGB2BGR)
                    cur_label = cur_sup_y[idx]
                    cv2.putText(cur_img, str(cur_label), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    show_sup[IMG_SIZE*row_idx:IMG_SIZE*(row_idx+1), IMG_SIZE*col_idx:IMG_SIZE*(col_idx+1)] = cur_img

            show_query = np.zeros([N_WAY*IMG_SIZE, QUERY_NUM*IMG_SIZE, 3]).astype(np.uint8)

            for row_idx in range(N_WAY):
                for col_idx in range(QUERY_NUM):
                    idx = QUERY_NUM * row_idx + col_idx
                    cur_img = cv2.cvtColor(cur_que_x[idx], cv2.COLOR_RGB2BGR)
                    cur_label = cur_que_y[idx]
                    cv2.putText(cur_img, str(cur_label), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    show_query[IMG_SIZE*row_idx:IMG_SIZE*(row_idx+1), IMG_SIZE*col_idx:IMG_SIZE*(col_idx+1)] = cur_img

            img_name = PLAYGROUND_DIR +  str(i).zfill(3) + '_' + str(b_idx) + '_'

            cv2.imwrite(img_name + 'support.jpg', show_sup)
            cv2.imwrite(img_name + 'query.jpg', show_query)   

if __name__ == '__main__':
    import sys
    sys.path.append("./")

    from config import parse_args    
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    subset = 'train'

    show_samples_difficulty(args, subset)
    show_samples_triplet(args, subset)
    show_samples(args, subset)