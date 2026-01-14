import os
import torch
import pprint
import random
import argparse
import numpy as np
from termcolor import colored
import time
import torch.nn.functional as F
from torch import nn
import math
import glob


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cos_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    logits = F.cosine_similarity(a, b, dim=-1)
    return logits

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'max_acc.pth' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.pth'.format(max_epoch))
    return resume_file

def setup_run(arg_mode='train'):
    args = parse_args(arg_mode=arg_mode)
    pprint(vars(args))

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    args.save_path = os.path.join(f'./checkpoints/{args.dataset}/{args.shot}shot-{args.way}way/', args.extra_dir)
    ensure_path(args.save_path)


    if args.dataset == 'miniimagenet':
        args.num_class = 64
    elif args.dataset == 'cub':
        args.num_class = 100
    elif args.dataset == 'aircraft':
        args.num_class = 64
    elif args.dataset == 'tieredimagenet':
        args.num_class = 351
    elif args.dataset == 'cifar_fs':
        args.num_class = 50
    elif args.dataset == 'cars':
        args.num_class = 130
    elif args.dataset == 'dogs':
        args.num_class = 70
    elif args.dataset == 'flowers':
        args.num_class = 51
    elif args.dataset == 'fc100':
        args.num_class = 60
    elif args.dataset == 'tiered_meta_iNat':
        args.num_class = 781
    return args

def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


def compute_accuracy(logits, labels):

    pred = torch.argmax(logits, dim=1)
    # print("labels",labels)
    # print("pred", pred)

    return (pred == labels).type(torch.float).mean().item() * 100.


_utils_pp = pprint.PrettyPrinter()


def save_list_to_txt(name,input_list):
    f=open(name,mode='w')
    for item in input_list:
        f.write(item+'\n')
    f.close()

def pprint(x):
    _utils_pp.pprint(x)


def load_model(model, dir):

    model_dict = model.state_dict()
    pretrained_dict = torch.load(dir)['params']

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:
        print('loading model from :', dir)
        if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
            if 'module' in list(pretrained_dict.keys())[0]:
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
        model.load_state_dict(model_dict)
        print("============load sucess===========")

    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:

        print('manual seed:', seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def detect_grad_nan(model):
    for name,param in model.named_parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()


def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow


def parse_args(arg_mode):
    parser = argparse.ArgumentParser(description='MS-SFC')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='cub',  choices=['tiered_meta_iNat','fc100','flowers','aircraft','cars','dogs','miniimagenet', 'cub', 'tieredimagenet', 'cifar_fs'])

    # parser.add_argument('-data_dir', type=str, default='../../../../../data1/zhangzhimin/FRN/fine-grained', help='dir of datasets')
    #parser.add_argument('-data_dir', type=str, default='/home/yly/workspace/renet/datasets', help='dir of datasets')
    parser.add_argument('-data_dir', type=str, default='/home/yuliyun/workspace/renet/datasets', help='dir of datasets')

    parser.add_argument('-step_size', type=int, default=10)

    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=64, help='auxiliary batch size')
    parser.add_argument('-temperature', type=float, default=2, metavar='tau', help='temperature for metric-based loss')
    parser.add_argument('-lamb', type=float, default=1.5, metavar='lambda', help='loss balancing term')
    parser.add_argument('-temperature_attn', type=float, default=5.0, metavar='gamma', help='temperature for softmax in computing cross-attention')
    parser.add_argument('-method', type=str, default='Task_trans', help='method')
    parser.add_argument('-resnet', action='store_true', help='model')
    parser.add_argument('-num_token', type=int, default=64, help='number of testing episodes after training')
    parser.add_argument('-coff', type=float, default=1, help='number of testing episodes after training')


    ''' about training schedules '''
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
    parser.add_argument('-max_epoch', type=int, default=120, help='max epoch to run')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=str, default=[80,100], help='milestones for MultiStepLR')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('-resume', action='store_true', help='continue from previous trained model with largest epoch')
    parser.add_argument('-save_freq', default=80, type=int, help='Save frequency')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1, metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=50, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')

    ''' about env '''
    parser.add_argument('-gpu', default='0', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-extra_dir', type=str, default='cub', help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=8, help='random seed')
    args = parser.parse_args()
    return args
