import torch.nn as nn
import torch.nn.parallel
import numpy as np
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm

import time

'''MODEL LOAD'''


def load_model(ckpt_path):
    vgg16_bn = models.vgg16_bn(pretrained=False)

    for param in vgg16_bn.parameters():
        param.requires_grad = True
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default

    vgg16_bn.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Linear(4096, 7*7*5))
    # assuming that the fc7 layer has 512 neurons, otherwise change it

    print vgg16_bn

    model = vgg16_bn

    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint)

    model.eval()

    return model


'''Non-Max Supression'''
# http://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.


def non_max_suppression_fast(bbx_matrix, confidenceThresh, overlapThresh):
    # arr[arr > 255] = x
    bbx_matrix[bbx_matrix[..., 4] < confidenceThresh] = 0

    bbx_list = np.reshape(bbx_matrix, (7 * 7, 5))

    # Remove bounding boxes with confidence lower than confidenceThresh
    indexes = np.sum(bbx_list, axis=1)
    indexes = np.where(indexes == 0)
    bbx_list = np.delete(bbx_list, indexes, axis=0)

    # Bounding box coordinates
    x = bbx_list[:, 0]
    y = bbx_list[:, 1]
    w = bbx_list[:, 2]
    h = bbx_list[:, 3]

    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    pick = []
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return bbx_list[pick]


'''Predict Bounding Boxes'''

# Load model
# model = load_model('./Data/Training/full_scale_l2_simple_detection_car/models/loss.pkl')

# Load image
image_PIL = Image.open('./Data/Datasets/kitti/image/007479.png')

# Transform PIL image to Tensor in correct format
transforms_imgs = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Predict new bounding boxes
for n in tqdm(range(100)):
    tensor_img = transforms_imgs(image_PIL)
    tensor_img = Variable(tensor_img.view(-1, 3, 224, 224), requires_grad=False)
    # bbx_matrix = model(tensor_img).view(7, 7, 5).data.numpy()
    bbx_matrix = np.random.rand(7, 7, 5)
    print non_max_suppression_fast(bbx_matrix, 0.5, 0.9)
    exit()
