import numpy as np
from model_trainer import run_training
import backbone_models
import model_lr
import argparse
import tensorflow as tf
import os
import imageio as io
import json
from pathlib import Path
import random
import specific_models
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', dest='model')
parser.add_argument('-lr', dest='lr')
parser.add_argument('-n','--name', dest='name')
parser.add_argument('-e','--epochs', dest='epochs')
parser.add_argument('-s','--steps', dest='steps', default=0)
parser.add_argument('-b','--batch', dest='batch', default=16)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-mg','--memorygrow', dest='mem_growth',
                    action='store_true',default=False)
parser.add_argument('-ml','--memorylimit', dest='mem_limit',
                    default=False)
parser.add_argument('-pf','--profile', dest='profile',
                    action='store_true',default=False)
parser.add_argument('--load',dest='load', default=False)
args = parser.parse_args()

if args.mem_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
if args.mem_limit:
    memory_limit = int(args.mem_limit)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=memory_limit
                )]
            )
        except RuntimeError as e:
            print(e)


train_dir = 'data/save'
val_dir = 'data/val'

specific_fs={
    'nose' : specific_models.conv3_16,
    'tail' : specific_models.conv3_16,
}

backbone_f = getattr(backbone_models, args.model)
lr_f = getattr(model_lr, args.lr)
name = args.name
epochs = int(args.epochs)
steps_per_epoch = int(args.steps)
batch_size = int(args.batch)
mixed_float = args.mixed_float
load_model_path = args.load
profile = args.profile
class_labels = ['nose', 'tail']

kwargs = {}
kwargs['backbone_f'] = backbone_f
kwargs['specific_fs'] = specific_fs
kwargs['lr_f'] = lr_f
kwargs['name'] = name
kwargs['epochs'] = epochs
kwargs['steps_per_epoch'] = steps_per_epoch
kwargs['batch_size'] = batch_size
kwargs['class_labels'] = class_labels
kwargs['train_dir'] = train_dir
kwargs['val_dir'] = val_dir
kwargs['img_size'] = (240,320)
kwargs['mixed_float'] = mixed_float
kwargs['notebook'] = False
kwargs['load_model_path'] = load_model_path
kwargs['profile'] = profile

run_training(**kwargs)