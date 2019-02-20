from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import cv2
import torch
import torchvision.utils as tutils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

def adjust_learning_rate(optimizer, epoch, i):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch == 0) & (i <= 550):
        power = 4
        lr = 1e-3 * (i / 550) ** power
        for g in optimizer.param_groups:
            g['lr'] = lr

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--epoch_start", type=int, default=5, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
print(hyperparams)
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
# model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=decay)
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 80)]
n_iter = 0

for epoch in range(opt.epoch_start, opt.epochs):
    writer = SummaryWriter()
    for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
        # adjust_learning_rate(optimizer, epoch, batch_i)
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)
        loss.backward()
        if((loss.cpu().item()<15 and epoch>0) or epoch==0):
            optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )
        n_iter += 1
        writer.add_scalar('data/losses/x', model.losses['x'], n_iter)
        writer.add_scalar('data/losses/y', model.losses['y'], n_iter)
        writer.add_scalar('data/losses/w', model.losses['w'], n_iter)
        writer.add_scalar('data/losses/h', model.losses['y'], n_iter)
        writer.add_scalar('data/losses/conf', model.losses['conf'], n_iter)
        writer.add_scalar('data/losses/cls', model.losses['cls'], n_iter)
        writer.add_scalar('data/losses/total', loss.item(), n_iter)
        if batch_i % 50 == 0:
            with torch.no_grad():
                img = np.array(Image.open(img_path[0]))
                image_pred = imgs[0].permute(1, 2, 0)
                image_pred = image_pred.cpu().numpy() * 255
                image_pred = image_pred.astype(np.uint8)
                detections = model(imgs[0].unsqueeze(0))
                detections = non_max_suppression(detections, 1, opt.conf_thres, opt.nms_thres)
                detections = detections[0]

                # The amount of padding that was added
                pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
                # Image height and width after padding is removed
                unpad_h = opt.img_size - pad_y
                unpad_w = opt.img_size - pad_x

                # Draw bounding boxes and labels of detections
                if detections is not None:
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        # Rescale coordinates to original dimensions
                        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                        try:
                            img = cv2.rectangle(img, (x1, y1), (x1 + box_w, y1 + box_h), (255, 0, 0),
                                                   thickness=2)
                        except:
                            img = None

                # Save generated image with detections
                try:
                    X = torch.from_numpy(img.astype(np.float16) / 255)
                    X = X.permute(2, 0, 1).unsqueeze(0)
                    X = tutils.make_grid(X)
                    writer.add_image('Image', X, n_iter)
                except:
                    continue
        model.seen += imgs.size(0)
        if epoch % opt.checkpoint_interval == 0:
            model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
