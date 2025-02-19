from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from cgi import test
from pprint import pprint
from libs.utils import (AverageMeter, save_checkpoint, IntervalSampler,
                        create_optim, create_scheduler, print_and_log, save_gt, generate_cm, extract_labels)
from libs.models import EncoderDecoder as ModelBuilder
from libs.models import Point_dis_loss, calculate_dis, get_criterion, wer_sliding_window
from libs.core import load_config
from libs.dataset import generate_data  # , gen_test
import torch.multiprocessing as mp
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import random
import numpy as np
import pandas as pd
import logging


# python imports
import argparse
import os
import re
import time
import math
import cv2
import pickle

# control the number of threads (to prevent blocking ...)
# should work for both numpy / opencv
# it really depends on the CPU / GPU ratio ...
TARGET_NUM_THREADS = '4'
os.environ['OMP_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['MKL_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = TARGET_NUM_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = TARGET_NUM_THREADS
# os.environ['CUDA_VISIBLE_DEVICES'] = GPUs.select()
# numpy imports
# torch imports

# for visualization
# from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# the arg parser
parser = argparse.ArgumentParser(description='Hand pose from mutliple views')
parser.add_argument('--print-freq', default=30, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--valid-freq', default=3, type=int,
                    help='validation frequency (default: 5)')
parser.add_argument('-o', '--output', default='temp', type=str,
                    help='the name of output file')
parser.add_argument('-i', '--input', default='', type=str,
                    help='overwrites dataset.data_file')
parser.add_argument('--stacking', default='', choices=['vertical', 'channel', ''], type=str,
                    help='overwrites input.stacking')
parser.add_argument('-p', '--path', default='', type=str,
                    help='path to dataset parent folder, overwrites dataset.path')
parser.add_argument('-ts', '--test-sessions', default='', type=str,
                    help='overwrites dataset.test_sessions, comma separated, e.g. 5,6,7')
parser.add_argument('--exclude-sessions', default='', type=str,
                    help='remove these sessions from training AND testing, comma separated, e.g. 5,6,7')
parser.add_argument('--train-sessions', default='', type=str,
                    help='overwrites dataset.train_sessions, default using all but testing sessions for training, comma separated, e.g. 5,6,7')
parser.add_argument('-g', '--visible-gpu', default='', type=str,
                    help='visible gpus, comma separated, e.g. 0,1,2 (overwrites config file)')
# parser.add_argument('-m', '--mode', default='', type=str,
#                     help='the mode of training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=-1, type=int,
                    help='total epochs to run, overwrites optimizer.epochs')
parser.add_argument('--epochs-to-run', default=-1, type=int,
                    help='how many epochs left to run, overwrites --epochs')
parser.add_argument('--lr', default=0, type=float,
                    help='learning rate, overwrites optimizer.learning_rate')
parser.add_argument('--bb', default='',
                    help='backbone, overwrites network.backbone')
parser.add_argument('--bn', default=0, type=int,
                    help='batchsize, overwrites input.batch_size')
parser.add_argument('--coi', default='',
                    help='channels of interest, comma-separated, overwrites input.channels_of_interest')
parser.add_argument('-v', '--variance-files', action='append',
                    help='variance files to be added during training, overwrites input.variance_files')
parser.add_argument('--test', action='store_true',
                    help='test only')
parser.add_argument('-f', '--training-file', default='both', type=str,
                        help='Train with original, diff, or both echo profiles (default: both)')
# parser.add_argument('-a', '--augment', default=0, type=int,
#                     help='if use the image augment')
# parser.add_argument('--all', action='store_true',
#                     help='use the model to run over training and testing set')
# parser.add_argument('--train_file', nargs='+')

# parser.add_argument('--test_file', nargs='+')

# main function for training and testing

def save_array(pred, loaded_gt, filename, cm):
    if pred is not None:
        save_arr = [(loaded_gt[i][0], loaded_gt[i][1], loaded_gt[i][2], loaded_gt[i][3], pred[i]) for i in range(len(pred))]
        if False and cm:
            truths = [int(x[0]) for x in save_arr]
            preds = [x[4] for x in save_arr]
            labels = extract_labels(loaded_gt)
            generate_cm(np.array(truths), np.array(preds), labels, filename[:-4] + '_cm.png')
    else:
        save_arr = loaded_gt
    save_gt(save_arr, filename)

def main(args):
    # ===================== initialization for training ================================
    print(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
    torch.cuda.empty_cache()
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)
    # parse args
    # best_metric = 100000.0
    best_metric = None
    metric_text = 'metric'
    args.start_epoch = 0
    
    torch.set_num_threads(int(TARGET_NUM_THREADS))

    config = load_config()  # load the configuration
    #print('Current configurations:')
    # pprint(config)
    #raise KeyboardInterrupt
    # prepare for output folder
    output_dir = args.output
    os.environ['CUDA_VISIBLE_DEVICES'] = config['network']['visible_devices']
    if len(args.visible_gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu
    print_and_log('Using GPU # %s' % os.environ['CUDA_VISIBLE_DEVICES'])
    if len(args.path):
        config['dataset']['path'] = args.path.rstrip('/')
        config['dataset']['root_folder'] = os.path.join(config['dataset']['path'], 'dataset')
        output_dir = os.path.basename(config['dataset']['path'])
    if not args.output == 'temp':
        output_dir = (os.path.basename(config['dataset']['path']) + '_' + args.output).replace('__', '_')
    if len(args.input):
        config['dataset']['data_file'] = args.input
    if len(args.stacking):
        config['input']['stacking'] = args.stacking
    if len(args.test_sessions):
        all_session_names = os.listdir(config['dataset']['root_folder'])
        all_session_names.sort()
        test_sessions = [s for s in args.test_sessions.split(',') if len(s)]
        train_sessions = [s for s in args.train_sessions.split(',') if len(s)]
        exclude_sessions = [s for s in args.exclude_sessions.split(',') if len(s)]
        config['dataset']['train_sessions'] = []
        config['dataset']['test_sessions'] = []
        for ss in all_session_names:
            if not args.test and re.match(r'session_\w+', ss) is None:
                continue
            session_suffix = re.findall(r'session_(\w+)', ss)[0]
            if session_suffix in test_sessions:# and session_suffix not in exclude_sessions:
                config['dataset']['test_sessions'] += [ss]
            elif (len(args.train_sessions) == 0 or session_suffix in train_sessions) and session_suffix not in exclude_sessions:
                config['dataset']['train_sessions'] += [ss]
    if args.epochs > 0:
        config['optimizer']['epochs'] = args.epochs
    if args.epochs_to_run > 0:
        config['optimizer']['epochs'] = args.start_epoch + args.epochs_to_run
    if args.lr > 0:
        config['optimizer']['learning_rate'] = args.lr
    if args.bn > 0:
        config['input']['batch_size'] = args.bn
    if len(args.bb) > 0:
        config['network']['backbone'] = args.bb
    if len(args.coi) > 0:
        config['input']['channels_of_interest'] = [int(x) for x in args.coi.split(',')]
    config['input']['variance_files'] = args.variance_files
    torch.cuda.empty_cache()
    ckpt_folder = os.path.join('./ckpt', output_dir)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    log_path = os.path.join(ckpt_folder, 'logs.txt')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
    # use spawn for mp, this will fix a deadlock by OpenCV (do not need)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    # fix the random seeds (the best we can)
    fixed_random_seed = 20220217
    torch.manual_seed(fixed_random_seed)
    np.random.seed(fixed_random_seed)
    random.seed(fixed_random_seed)

    # set up transforms and dataset
    # ===================== packet the training data and testinf data ================================
    train_dataset, val_dataset, test_gt = generate_data(input_config=config['input'], data_config=config['dataset'], is_train=(not args.test), training_file=args.training_file)

    if config['network']['loss_type'] == 'ce':  # classification task, determine output dimension based on number of labels
        config['network']['output_dims'] = max([int(x[0]) for x in test_gt]) + 1
        print_and_log('Classification task detected, output dimension: %d' % config['network']['output_dims'])
    elif config['network']['loss_type'] == 'ctc':  # classification task, determine output dimension based on number of labels
        config['network']['output_dims'] = max([max([int(xx) for xx in x[0].split()]) for x in test_gt]) + 2
        print_and_log('Classification task detected, output dimension: %d' % config['network']['output_dims'])
    else:
        config['network']['output_dims'] = test_gt[0].shape[0] - 4
        print_and_log('Regression task detected, output dimension: %d' % config['network']['output_dims'])
    config['network']['input_channels'] = config['input']['model_input_channels']
    print_and_log(time.strftime('finish data: %Y-%m-%d %H:%M:%S', time.localtime()))
    # create model w. loss
    model = ModelBuilder(config['network'])  # load the designed model
    # GPU you will use in training
    master_gpu = config['network']['devices'][0]
    model = model.cuda(master_gpu)  # load model from CPU to GPU
    # create optimizer
    optimizer = create_optim(model, config['optimizer'])  # gradient descent
    # data parallel
    # if you want use multiple GPU
    model = nn.DataParallel(model, device_ids=config['network']['devices'])
    logging.info(model)
    # set up learning rate scheduler
    if not args.test:
        num_iters_per_epoch = len(train_dataset)
        scheduler = create_scheduler(
            optimizer, config['optimizer']['schedule'],
            config['optimizer']['epochs'], num_iters_per_epoch)

    # ============================= retrain the trained model (if need usually not) =========================================
    # resume from a checkpoint?
    if args.resume:
        #not args.test = 0
        print_and_log('loading trained model.....')
        if os.path.isfile(args.resume):
            print_and_log('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(master_gpu))
            # args.start_epoch = 0
            args.start_epoch = checkpoint['epoch']
            best_metric = np.inf
            # best_metric = checkpoint['best_metric']
            # encoder_only_dict = {x: checkpoint['state_dict'][x] for x in checkpoint['state_dict'] if 'decoder' not in x}
            # encoder_only_dict.update({x: model.state_dict()[x] for x in model.state_dict() if 'decoder' in x})
            # model.load_state_dict(encoder_only_dict)
            model.load_state_dict(checkpoint['state_dict'])
            if args.epochs_to_run > 0:
                config['optimizer']['epochs'] = args.start_epoch + args.epochs_to_run
            # only load the optimizer if necessary
            if not args.test:
                # best_metric = ['best_metric']
                scheduler = create_scheduler(
                    optimizer, config['optimizer']['schedule'],
                    config['optimizer']['epochs'] - args.start_epoch, num_iters_per_epoch)
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # scheduler.load_state_dict(checkpoint['scheduler'])
            print_and_log('=> loaded checkpoint {} (epoch {}, metric {:.3f}, best_metric {:.3f})'
                  .format(args.resume, checkpoint['epoch'], checkpoint['ckpt_metric'], best_metric))
        else:
            print_and_log('=> no checkpoint found at {}'.format(args.resume))
            return

    # =================================== begin training =========================================
    # training: enable cudnn benchmark
    cudnn.enabled = True
    cudnn.benchmark = True

    # model architecture
    model_arch = '{:s}-{:s}'.format(
        config['network']['backbone'], config['network']['decoder'])

    # start the training
    if not args.test:
        # if not os.path.isfile(args.resume):
        if best_metric is None:
            if config['network']['loss_type'] == 'ce':
                best_metric = 0
            else:
                best_metric = np.inf
        # save the current config
        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint(config, stream=fid)
        # print('Training model {:s} ...'.format(model_arch))
        #save_test_data(val_dataset, config_filename)
        torch.cuda.empty_cache()

        for epoch in range(args.start_epoch, config['optimizer']['epochs']):
            # train for one epoch
            # print('epoch', epoch)
            #print(time.strftime('begin: %Y-%m-%d %H:%M:%S', time.localtime()))
            # (training data, model, others (configuration, gradient descent))
            train(train_dataset, model, optimizer,
                  scheduler, epoch, args, config)
            # torch.cuda.empty_cache()

            # evaluate on validation set once in a while
            # test on every epoch at the end of training
            # Note this will also run after first epoch (make sure training is on track)
            if epoch % args.valid_freq == 0 \
                    or (epoch > 0.9 * config['optimizer']['epochs']):
                # run bn once before validation
                prec_bn(train_dataset, model, args)
                # (testing data, model, others (configuration, gradient descent))
                metric, loss, pred_array, fake_gts = validate(val_dataset, model, args, config)
                # if epoch // args.valid_freq == 0:
                # print(metric)
                # remember best metric and save checkpoint
                if config['network']['loss_type'] == 'ctc' and config['input']['test_sliding_window']['applied']:
                    # print(pred_array[:10])
                    metric, pred_array, _ = wer_sliding_window(pred_array, fake_gts, test_gt, config['input']['test_sliding_window']['pixels_per_label'])
                    # print(pred_array[:10])
                display_str = '**** testing loss: {:.4f}, metric: {:.4f}, '.format(loss, metric)
                if config['network']['loss_type'] == 'ce':
                    is_best = metric > best_metric
                    best_metric = max(metric, best_metric)
                    if abs(best_metric - 1) < 1e-6:
                        break
                else:
                    is_best = metric < best_metric
                    best_metric = min(metric, best_metric)
                    if abs(best_metric - 0) < 1e-6:
                        break
                if config['network']['loss_type'] in ['ce', 'ctc']:
                    save_array(pred_array, test_gt, os.path.join(ckpt_folder, 'ckpt_pred.txt'), config['network']['loss_type'] == 'ce')
                    exact_match_acc = np.mean([test_gt[x][0] == pred_array[x] for x in range(len(test_gt))])
                    display_str += 'exact match acc: %.2f%%, ' % (100 * exact_match_acc)
                else:
                    save_array(None, pred_array, os.path.join(ckpt_folder, 'ckpt_pred.npy'), False)
                    save_array(None, test_gt, os.path.join(ckpt_folder, 'test_gt.npy'), False)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_arch': model_arch,
                    'state_dict': model.state_dict(),
                    'ckpt_metric': metric,
                    'best_metric': best_metric,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, file_folder=ckpt_folder)
                # print('saving')
                if is_best:
                    display_str += f'\033[92mbest_%s = %.4f\033[0m' % (metric_text, best_metric)
                    if config['network']['loss_type'] in ['ce', 'ctc']:
                        save_array(pred_array, test_gt, os.path.join(ckpt_folder, 'best_pred.txt'), config['network']['loss_type'] == 'ce')
                    else:
                        save_array(None, pred_array, os.path.join(ckpt_folder, 'best_pred.npy'), False)
                        save_array(None, test_gt, os.path.join(ckpt_folder, 'test_gt.npy'), False)
                else:
                    display_str += f'best_%s = %.4f' % (metric_text, best_metric)
                display_str += ' ' * 20
                print_and_log(display_str)
                # manually reset mem
            torch.cuda.empty_cache()
################################ save the file ###########################################

    if args.test:
        metric_test, loss_test, pred_all_test, fake_gts = validate(val_dataset, model, args, config)
        if config['network']['loss_type'] == 'ctc' and config['input']['test_sliding_window']['applied']:
            # print(pred_all_test[:10])
            # print(fake_gts[:10])
            metric_test, pred_all_test, raw_preds_all = wer_sliding_window(pred_all_test, fake_gts, test_gt, config['input']['test_sliding_window']['pixels_per_label'])
            # print(raw_preds_all.shape)
            np.save('raw_preds_all.npy', raw_preds_all)
        print_and_log('**** Testing loss: %.4f, metric: %.4f ****' % (loss_test, metric_test))
        if config['network']['loss_type'] in ['ce', 'ctc']:
            save_array(pred_all_test, test_gt, os.path.join(ckpt_folder, 'test_pred.txt'), config['network']['loss_type'] == 'ce')
            # for i in range(len(test_gt)):
            #     print(test_gt[i][0], pred_all_test[i])
            #     if test_gt[i][0] != pred_all_test[i]:
            #         print(test_gt[i])
            all_acc = np.mean([test_gt[x][0] == pred_all_test[x] for x in range(len(pred_all_test))])
            print_and_log('Exact match acc: %.4f' % all_acc)
        else:
            save_array(None, pred_all_test, os.path.join(ckpt_folder, 'test_pred.npy'), False)
            save_array(None, test_gt, os.path.join(ckpt_folder, 'test_gt.npy'), False) 
        # save_array(pred_all_test, test_gt, os.path.join(ckpt_folder, 'test_pred.txt'))

    print_and_log(time.strftime('end: %Y-%m-%d %H:%M:%S', time.localtime()))

def train(train_loader, model, optimizer, scheduler, epoch, args, config=None):
    '''Training the model'''
    # set up meters
    num_iters = len(train_loader)
    batch_time = AverageMeter()
    # loss is our err here
    losses = AverageMeter()
    metrics = AverageMeter()

    # switch to train mode
    model.train()
    master_gpu = config['network']['devices'][0]
    end = time.time()

    loss_type = config['network']['loss_type']
    criterion = get_criterion(loss_type)

    output_str_length = 0
    for i, (input_arr, target) in enumerate(train_loader):
        # print(target, type(target))
        # print(type(input_arr), input_arr.shape)
        # input_arr = augment(input_arr, config['dataset'])
        #print(type(input_arr), input_arr.shape)
        #raise KeyboardInterrupt
        input_arr = input_arr.cuda(master_gpu, non_blocking=True)
        if loss_type == 'ce':
            target = torch.LongTensor([int(x) for x in target])
        if not loss_type == 'ctc':
            target = target.cuda(master_gpu, non_blocking=True)
        # compute output
        output, loss = model(input_arr, targets=target)
        loss = loss.mean()
        # input_arr.cpu()
        # output.cpu()
        # target.cpu()
        # compute gradient and do SGD step
        # !!!! important (update the parameter of your model with gradient descent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metric, preds, raw_preds = criterion(output, target)
        # if epoch % 5 == 0:
        #     print(target, preds)

        losses.update(loss.data.item(), input_arr.size(0))
        metrics.update(metric, input_arr.size(0))

        del loss
        del output

        # printing the loss of traning
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            # make sure tensors are properly detached from the graph

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            # printing
            output_str = 'Epoch: [{:3d}][{:4d}/{:4d}], Time: {:.2f} ({:.2f}), loss: {:.3f} ({:.3f}), metric: {:.4f} ({:.4f})'\
                            .format(epoch, i, num_iters, batch_time.val,
                                    batch_time.avg, losses.val,
                                    losses.avg, metrics.val, metrics.avg)
            print_and_log(output_str, end='\r')
            output_str_length = max(output_str_length, len(output_str))
        # step the lr scheduler after each iteration
        scheduler.step()

    # print the learning rate
    lr = scheduler.get_last_lr()[0]
    output_str = 'Epoch {:d} finished with lr={:.6f}, loss={:.3f}, metric={:.4f}'.format(epoch, lr, losses.avg, metrics.avg)
    output_str += ' ' * (output_str_length - len(output_str) + 1)
    print_and_log(output_str)
    # lr = scheduler.get_lr()[0]
    # print('\nEpoch {:d} finished with lr={:f}'.format(epoch + 1, lr))
    # log metric
    # writer.add_scalars('data/metric', {'train' : metric.avg}, epoch + 1)i
    # print(metric.avg)
    return metrics.avg


def validate(val_loader, model, args, config):
    '''Test the model on the validation set'''
    # set up meters
    batch_time = AverageMeter()
    metrics = AverageMeter()
    losses = AverageMeter()

    # metric_action = AverageMeter()
    # metric_peak = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # master_gpu = config['network']['devices'][0]
    end = time.time()

    # prepare for outputs
    pred_list = []
    # truth_list = []

    loss_type = config['network']['loss_type']
    criterion = get_criterion(loss_type)
    # criterion = get_criterion(loss_type)
    # criterion = Point_dis_loss
    # criterion = calculate_dis
    # criterion = nn.functional.l1_loss
    # criterion = nn.functional.cross_entropy

    output_str_length = 0
    sliding_window = (loss_type == 'ctc' and config['input']['test_sliding_window']['applied'])
    # if sliding_window:
    fake_gts = []
        # raw_preds_all = []

    # loop over validation set
    for i, (input_arr, target) in enumerate(val_loader):
        # print(target)
        # print(input_arr.shape)
        if loss_type == 'ce':
            target = torch.LongTensor([int(x) for x in target])
        if not loss_type == 'ctc':
            target = target.cuda(config['network']['devices'][0], non_blocking=True)
        if sliding_window:
            for s, e in zip(target[0].numpy(), target[1].numpy()):
                fake_gts += [(s, e)]
            target = None
        # forward the model
        # print(input_arr.shape)
        with torch.no_grad():
            output, loss = model(input_arr, targets=target)
        # loss = loss.mean()
        # print(type(output.data), output.data.shape)

        # measure metric and record loss
        metric, pred, raw_preds = criterion(output, target, require_pred=True)
        # print(pred)
        # print(raw_preds)
        # # err = nn.functional.l1_loss(pred, truth)
        # err = criterion(output, target)
        if loss is not None:
            losses.update(loss.data.item(), input_arr.size(0))
        if metric is not None:
            # print(metric)
            metrics.update(metric, input_arr.size(0))
        # if sliding_window:
        #     raw_preds_all += raw_preds

       # print(type(pred), pred.shape)
        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # append the output list
        if sliding_window:
            pred_list.append(raw_preds)
        else:
            pred_list.append(pred)

        # printing
        if i % (args.print_freq * 2) == 0 or i == len(val_loader) - 1:
            output_str = 'Test: [{:4d}/{:4d}], Time: {:.2f} ({:.2f}), loss {:.2f} ({:.2f}), metric {:.4f} ({:.4f})'.format(i + 1, len(val_loader),
                                                  batch_time.val, batch_time.avg,
                                                  losses.val, losses.avg,
                                                  metrics.val, metrics.avg)
            print_and_log(output_str, end='\r')
            output_str_length = max(output_str_length, len(output_str))
            # print('Test: [{:d}/{:d}]\t'
            #       'Time {:.2f} ({:.2f})\t'
            #       'MSD {:.3f} ({:.3f})\t'.format(
            #           i, len(val_loader), batch_time.val, batch_time.avg, metric.val, metric.avg), end='\r')
    # output_str = '**** testing loss: {:.4f}, metric: {:.4f}'.format(losses.avg, metrics.avg)
    # output_str += ' ' * (output_str_length - len(output_str) + 1)
    # print_and_log(output_str, end=' ') 
    # print('\n******MSD {:3f}'.format(metric.avg))
    pred_list = np.concatenate(pred_list)
    # print(pred_list[:100])
    return metrics.avg, losses.avg, pred_list, fake_gts

def prec_bn(train_loader, model, args):
    '''Aggreagate precise BN stats on train set'''
    # set up meters
    batch_time = AverageMeter()
    # switch to train mode (but do not require gradient / update)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.05
    end = time.time()

    # loop over validation set
    for i, (input_arr, _,) in enumerate(train_loader):
        # forward the model
        with torch.no_grad():
            _ = model(input_arr)

        # printing
        if i % (args.print_freq * 2) == 0:
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / (args.print_freq * 2))
            end = time.time()
            print_and_log('Prec BN: [{:d}/{:d}], Time: {:.2f} ({:.2f})'.format(
                      i, len(train_loader), batch_time.val, batch_time.avg), end='\r')
    # print('\n')
    return


################################################################################
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)