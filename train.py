import torch
from torch.utils.data import DataLoader

import numpy as np
import logging
import os
from time import time

from config import parse_args
from train_classifier import train_classifier
from correlation import get_correlation

from model import ConvModel
from utils.data_loaders import Dataset, Dataset_Difficulty
from utils.helpers import *
from utils.average_meter import AverageMeter
import higher
import tqdm as tqdm

def main():

    args = parse_args()

    # Few Shot Parameters
    N_WAY = args.N_way
    K_SHOT = args.K_shot
    QUERY_NUM = args.query_num
    EVALUATE_TASK = args.evaluate_task

    # MAML Parameters
    TASK_NUM = args.task_num
    NUM_STEPS_TRAIN = args.num_steps_train
    NUM_STEPS_TEST = args.num_steps_test
    STEP_SIZE = args.step_size
    FIRST_ORDER = args.first_order

    # Model Parameters
    HIDDEN_UNIT = args.hidden_unit

    # Session Parameters
    GPU_NUM = args.gpu_num
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    META_LR = args.meta_lr
    PRINT_EVERY = args.print_every
    EVALUATE_EVERY = args.evaluate_every

    # Directory Parameters
    DATASET = args.dataset
    EXP_NAME = args.experiment_name
    EXP_DIR = 'experiments/' + EXP_NAME
    CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
    LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
    WEIGHTS_FEWSHOT = args.weights_fewshot
    WEIGHTS_CLASSIFIER = args.weights_classifier

    # Check if directory does not exist
    create_path('experiments/')
    create_path(EXP_DIR)
    create_path(CKPT_DIR)
    create_path(LOG_DIR)
    create_path(os.path.join(LOG_DIR, 'train'))
    create_path(os.path.join(LOG_DIR, 'test'))

    # Set up logger
    filename = os.path.join(LOG_DIR, 'logs.txt')
    logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))
        logging.info('\t%15s:\t%s' % (key, value))

    # Train Classifier for Correlation Map
    train_classifier(args, 'train')
    train_classifier(args, 'val')
    train_classifier(args, 'test')

    train_corr, _ = get_correlation(args, 'train')
    val_corr, _ = get_correlation(args, 'val')
    test_corr, _ = get_correlation(args, 'test')

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up Dataset
    train_dataset = Dataset_Difficulty(args, 'train', train_corr)
    val_dataset = Dataset_Difficulty(args, 'val', val_corr)
    test_dataset = Dataset_Difficulty(args, 'test', test_corr)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=TASK_NUM,
        num_workers=0,
        shuffle=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=TASK_NUM,
        num_workers=0,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=TASK_NUM,
        num_workers=0,
        shuffle=False
    )

    train_dataloader.dataset.set_prob(0.0)
    val_dataloader.dataset.set_prob(0.0)
    test_dataloader.dataset.set_prob(0.0)

    # Set up model / optimizer
    model = ConvModel(in_channels=3, out_features=N_WAY, hidden_size=HIDDEN_UNIT)
    model.to(device)

    meta_opt = torch.optim.Adam(model.parameters(), lr=META_LR)

    # Set up Loss Functions
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Load the pretrained model if exists
    init_epoch = 0
    best_metrics_val = 0.0
    best_metrics_test = 0.0

    if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS_FEWSHOT)):
        logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS_FEWSHOT))
        checkpoint = torch.load(os.path.join(CKPT_DIR, WEIGHTS_FEWSHOT))
        init_epoch = checkpoint['epoch_idx']
        best_metrics_val = checkpoint['best_metrics_val']
        best_metrics_test = checkpoint['best_metrics_test']
        model.load_state_dict(checkpoint['model'])
        logging.info('Recover completed. Current epoch = #%d, best metrics (val) = %.3f, best metrics (test) = %.3f' % (init_epoch, best_metrics_val, best_metrics_test))

    for epoch_idx in range(init_epoch+1, EPOCHS):

        # Get harder example every 500 steps
        if epoch_idx < 500:
            train_dataloader.dataset.set_diff_th(0.55)
        elif epoch_idx < 1000:
            train_dataloader.dataset.set_diff_th(0.60)
        elif epoch_idx < 1500:
            train_dataloader.dataset.set_diff_th(0.65)        
        elif epoch_idx < 2000:
            train_dataloader.dataset.set_diff_th(0.70)        

        loss, acc = train(train_dataloader, model, device, meta_opt, criterion, args)

        if epoch_idx % PRINT_EVERY == 0:
            logging.info('[Epoch %d/%d] Loss = %.4f  Accuracy = %.4f' %(epoch_idx, EPOCHS, loss.avg(), acc.avg()))

        if epoch_idx % EVALUATE_EVERY == 0:

            loss_val, acc_val = evaluate(val_dataloader, model, device, criterion, 0.0, args)

            # Evaluate for different difficulties
            eval_diffs = [0.0, 0.5, 0.55, 0.60, 0.65, 0.70]
            loss_tests = []
            acc_tests = []

            for eval_diff in eval_diffs:
                loss_test, acc_test = evaluate(test_dataloader, model, device, criterion, eval_diff, args)
                
                loss_tests.append(loss_test)
                acc_tests.append(acc_test)

            logging.info('Val  [Epoch %d/%d] Loss = %.4f  Accuracy = %.4f' %(epoch_idx, EPOCHS, loss_val, acc_val))
            logging.info('Test [Epoch %d/%d] Loss = %.4f  Accuracy = %.4f' %(epoch_idx, EPOCHS, loss_tests[0], acc_tests[0]))

            out_string = 'Test Difficulty : '
            out_tuple = ()
            for diff, acc in zip(eval_diffs, acc_tests):
                out_string += 'Diff = %.3f   Acc = %.4f /'
                out_tuple += (diff, acc)
            logging.info(out_string % out_tuple)

            # Save if test accuracy is best
            if acc_tests[0] > best_metrics_test:
                output_path = os.path.join(CKPT_DIR, WEIGHTS_FEWSHOT)
                best_metrics_val = acc_val
                best_metrics_test = acc_tests[0]

                torch.save({
                    'epoch_idx': epoch_idx,
                    'best_metrics_val': best_metrics_val,
                    'best_metrics_test': best_metrics_test,
                    'model': model.state_dict()
                }, output_path)

                logging.info('Saved checkpoint to %s ... ' % output_path)
            logging.info('Best Test Accuracy = %.4f' %(best_metrics_test))                

def train(dataloader, model, device, meta_opt, criterion, args):

    STEP_SIZE = args.step_size
    TASK_NUM = args.task_num
    NUM_STEPS_TRAIN = args.num_steps_train
    QUERY_NUM = args.query_num
    N_WAY = args.N_way
    BATCH_SIZE = args.batch_size

    model.train()

    qry_loss_avg = AverageMeter()
    qry_acc_avg = AverageMeter()

    for batch_idx, data in enumerate(dataloader):

        support_x, support_y, query_x, query_y, _ = data
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        inner_opt = torch.optim.SGD(model.parameters(), lr=STEP_SIZE)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()

        for i in range(TASK_NUM):
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                for _ in range(NUM_STEPS_TRAIN):
                    spt_logits = fnet(support_x[i])
                    spt_loss = criterion(spt_logits, support_y[i])
                    diffopt.step(spt_loss)

                qry_logits = fnet(query_x[i])
                qry_loss = criterion(qry_logits, query_y[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(dim=1) == query_y[i]).sum().item() / (QUERY_NUM*N_WAY)
                qry_accs.append(qry_acc)

                qry_loss.backward()
        
        meta_opt.step()
        qry_losses = sum(qry_losses) / TASK_NUM
        qry_accs = 100. * sum(qry_accs) / TASK_NUM

        qry_loss_avg.update(qry_losses)
        qry_acc_avg.update(qry_accs)

        if (batch_idx+1) % BATCH_SIZE == 0:
            break

    return qry_loss_avg, qry_acc_avg        

def evaluate(dataloader, model, device, criterion, difficulty, args):

    STEP_SIZE = args.step_size
    TASK_NUM = args.task_num
    NUM_STEPS_TEST = args.num_steps_test
    QUERY_NUM = args.query_num
    N_WAY = args.N_way
    EVALUATE_TASK = args.evaluate_task

    qry_losses = []
    qry_accs = []

    model.train()

    dataloader.dataset.set_diff_th(difficulty)

    for batch_idx, data in enumerate(dataloader):

        support_x, support_y, query_x, query_y, _ = data
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        inner_opt = torch.optim.SGD(model.parameters(), lr=STEP_SIZE)

        for i in range(TASK_NUM):
            with higher.innerloop_ctx(model, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                for _ in range(NUM_STEPS_TEST):
                    spt_logits = fnet(support_x[i])
                    spt_loss = criterion(spt_logits, support_y[i])
                    diffopt.step(spt_loss)

                qry_logits = fnet(query_x[i]).detach()
                qry_loss = criterion(qry_logits, query_y[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(dim=1) == query_y[i]).sum().item() / (QUERY_NUM*N_WAY)
                qry_accs.append(qry_acc)

        if (batch_idx+1) % EVALUATE_TASK == 0:
            break
    
    qry_losses = sum(qry_losses) / (EVALUATE_TASK * TASK_NUM)
    qry_accs = 100. * sum(qry_accs) / (EVALUATE_TASK * TASK_NUM)

    return qry_losses, qry_accs

if __name__ == '__main__':
    main()