import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"
import time
import torch
import numpy as np
import random
import argparse
import json
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run
import sys
from datetime import datetime


def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--ratio', default=[0.8, 0.2])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['base_model'] = args.base_model
        config['task'] = args.task
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
    return args, config

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

# if __name__ == '__main__':
#     config_path = r'.\config.json'
#     args, config = prepare(config_path)
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     path = os.path.abspath(os.path.dirname(__file__))
#     type = sys.getfilesystemencoding()
#     current_date = datetime.now().date()

#     date_string = current_date.strftime("%Y-%m-%d")
#     path=r'E:\Result\PTUPCDR\%s.txt'%date_string
#     path_result=r'E:\Result\PTUPCDR\%s_result.txt'%date_string
#     sys.stdout = Logger(path)

#     if args.process_data_mid:
#         for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
#             DataPreprocessingMid(config['root'], dealing).main()
#     if args.process_data_ready:
#         for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
#             for task in ['1', '2', '3']:
#                 DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()
#     print('task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};'.
#           format(args.task, args.base_model, args.ratio, args.epoch, args.lr, args.gpu, args.seed))
#     for config['task'] in ['1','2','3']:
#         for config['ratio'] in [0.2,0.5,0.8]:
#             print("task:"+config["src_tgt_pairs"][config['task']]['src']+" and "+config["src_tgt_pairs"][config['task']]["tgt"])
#             print("Ration:"+str(config['ratio']))
#             print("lr:"+str(config['lr']))
#             config['epoch']=70
#             print('epoch：'+str(config['epoch']))
#             run = Run(config)
#             run.main()

    # if not args.process_data_mid and not args.process_data_ready:
    #     Run(config).main()
if __name__ == '__main__':
    config_path = 'config.json'
    args, config = prepare(config_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    current_date = datetime.now().date()

    date_string = current_date.strftime("%Y-%m-%d")
    path=r'.\%s.txt'%date_string
    path_result=r'.\%s_result.txt'%date_string
    sys.stdout = Logger(path)
    for config['task'] in ['1']: #['1','2','3']:
        for config['ratio'] in  [[0.8, 0.2]]:#[[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
            print("task:"+config["src_tgt_pairs"][config['task']]['src']+" and "+config["src_tgt_pairs"][config['task']]["tgt"])
            print("Ration:"+str(config['ratio']))
            print("lr:"+str(config['lr']))
            config['epoch']=20
            print('epoch：'+str(config['epoch']))
            run = Run(config)
            run.main()
