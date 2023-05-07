import argparse
import json
import os
import h5py
import cv2

from pycocotools import mask
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', dest='file', default='hdf5_annotations.json', help='coco annotation json file')
parser.add_argument('-i', '--image_index', dest='image_index', default=0, help='image over which to annotate, uses the rgb rendering', type=int)
parser.add_argument('-b', '--base_path', dest='base_path', default='/home/asl/catkin_ws/src/oaisys/oaisys_tmp/2022-12-23-09-23-18/', help='path to folder with hdf5_annotation.json and images', type=str)
parser.add_argument('--save', '-s', action='store_true', help='saves visualization of coco annotations under base_path/coco_annotated_x.png ')
parser.add_argument('--skip_vis', action='store_true', help='skips the visualization and only saves the annotated file')

args = parser.parse_args()
writer = SummaryWriter()

annot_file = args.file
image_idx = args.image_index
base_path = Path(args.base_path)
save = args.save
skip_vis = args.skip_vis

if skip_vis:
    save = True
img = None
for hdf5_idx in range(len([entry for entry in base_path.glob('*.hdf5')])):
    hdf5_file = os.path.join(base_path, f"{hdf5_idx}.hdf5")
    img = None
    with h5py.File(hdf5_file, "r") as f:
        img =  cv2.cvtColor(f['sensor_1_rgb'][()], cv2.COLOR_RGB2BGR)
        writer.add_image(f'image', img, hdf5_idx, dataformats='HWC')
        writer.add_image(f'depth', f['sensor_1_pinhole_depth'][()], hdf5_idx, dataformats='HW')
        writer.add_image(f'segmentation', f['sensor_1_segmentation'][()], hdf5_idx, dataformats='HWC')
        writer.add_image(f'label', f['sensor_1_sem_label'][()], hdf5_idx, dataformats='HW')
        label = ((f['sensor_1_sem_label'][()]))
        segmentation = f['sensor_1_segmentation'][()]
        filtered = np.where(label<23, label, 3)
        writer.add_image(f'filtered', filtered, hdf5_idx, dataformats='HW')
    if img is None:
        print("error loading hdf5 image")
    if hdf5_idx == 20:
        writer.close()
        break


