import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import cv2


def normalise_mask(mask, threshold=0.5):
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    return mask


def metrics_line(data):
    phases = list(data.keys())
    metrics = list(data[phases[0]][0].keys())

    i = 0
    fig, axs = plt.subplots(1, len(metrics))
    fig.set_figheight(4)
    fig.set_figwidth(4 * len(metrics))
    for metric in metrics:
        for phase in phases:
            axs[i].plot([i[metric] for i in data[phase]], label=phase)
        axs[i].set_title(metric)
        i += 1

    plt.legend()
    plt.show()


class Plot(object):

    def __init__(self, plots_dir):
        super().__init__()

        self.plots_dir = plots_dir
        self.segments_colors = [(201, 58, 64), (242, 207, 1), (0, 152, 75),
                                (101, 172, 228), (56, 34, 132), (160, 194, 56)]

    def reverse_transform(self, inp):
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        inp = (inp * 255).astype(np.uint8)
        return inp

    def plot_img_array(self, img_array, ncol=3, index=None, is_mask=False):
        nrow = len(img_array) // ncol

        f, plots = plt.subplots(nrow, ncol, sharex='all',
                                sharey='all', figsize=(ncol * 4, nrow * 4))

        for i in range(len(img_array)):
            # plots[i // ncol, i % ncol]
            plots[i].imshow(img_array[i])

        plt.savefig(self.plots_dir + 'PLOT_{}_{: 03d}.png'.format(
            'MASK' if is_mask else 'BBOXS', index))

    def plot_image_truemask_predictedmask(self, images, labels, preds, index):

        input_images_rgb = [self.reverse_transform(x) for x in images]
        target_masks_rgb = [self.masks_to_coloredmasks(x) for x in labels]
        pred_rgb = [self.masks_to_coloredmasks(x) for x in preds]

        img_arrays = [input_images_rgb, target_masks_rgb, pred_rgb]
        flatten_list = reduce(lambda x, y: x+y, zip(*img_arrays))

        self.plot_img_array(np.array(flatten_list),
                            ncol=len(img_arrays), index=index,
                            is_mask=True)

    def draw_bboxs(self, im, bboxs):
        # print('==========================')
        # print('BEF: ', im)
        # print('BEF', type(im))

        for label, bbox in bboxs:
            color = self.segments_colors[label]
            xmin, ymin, xmax, ymax = bbox
            im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 2)
            if type(im) is not np.ndarray:
                im = im.get()
            # print('IN', type(im))

        # if len(bboxs) > 0:
        #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #     print('convert cv -> np')
        # else:
        #     print('no convert')

        # print('AFTER ', im)
        # print('AFT', type(im))
        return im

    def plot_image_truebbox_predictedbbox(self, images, bboxs, bboxs_preds, index):

        input_image_rgb = [self.reverse_transform(img) for img in images]
        image_with_real_masks = [self.draw_bboxs(
            img, bboxs) for img in input_image_rgb]
        image_with_pred_masks = [self.draw_bboxs(
            img, bboxs_preds) for img in input_image_rgb]

        img_arrays = [input_image_rgb,
                      image_with_real_masks,
                      image_with_pred_masks]

        flatten_list = reduce(lambda x, y: x+y, zip(*img_arrays))
        self.plot_img_array(np.array(flatten_list),
                            ncol=len(img_arrays), index=index)

    def apply_mask_color(self, mask, mask_color):
        colored_mask = np.concatenate(
            ([mask[..., np.newaxis] * color for color in mask_color]), axis=2)
        return colored_mask.astype(np.uint8)

    def masks_to_coloredmasks(self, mask, normalise=True):
        if normalise:
            normalise_mask(mask)

        mask_colored = np.concatenate(
            [[self.apply_mask_color(mask[i], self.segments_colors[i])] for i in range(len(mask))])
        mask_colored = np.max(mask_colored, axis=0)

        mask_colored = np.where(
            mask_colored.any(-1, keepdims=True), mask_colored, 255)

        return mask_colored

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# from meditorch.utils.images import resize_image_to_square
# import numpy as np

# """
# Created on Wed Nov 21 13:24:24 2018

# @author: ead2019
# """
# def convert_boxes(boxes, class_names, datatype, imgshape):
#     nrows, ncols = imgshape
#     data = []

#     if len(boxes) > 0:
#         for bbox in boxes:
#             if datatype=='GT':
#                 cls, b1, b2, b3, b4 = bbox
#             elif datatype=='Pred':
#                 cls, conf, b1, b2, b3, b4 = bbox
#             else:
#                 raise Exception('datatype should be either \'GT\' or \'Pred\'. The value of datatype was: {}'.format(datatype))

#             # check whether we have been given the name already.
#             try:
#                 cls = int(cls)
#                 cls_name = class_names[int(cls)]
#             except:
#                 cls_name = str(cls)

#             # check whether yolo or not.
#             bbox_bounds = np.hstack([b1,b2,b3,b4]).astype(np.float)

#             if bbox_bounds.max() < 1.1:
#                 # yolo:
# #                x1 = (bbox_bounds[0] - bbox_bounds[2]) / 2. * ncols
# #                y1 = (bbox_bounds[1] - bbox_bounds[3]) / 2. * nrows
# #                x2 = (bbox_bounds[0] + bbox_bounds[2]) / 2. * ncols
# #                y2 = (bbox_bounds[1] + bbox_bounds[3]) / 2. * nrows

#                 # yolo 2 voc
#                 x = bbox_bounds[0] * ncols
#                 w = bbox_bounds[2] * ncols

#                 y = bbox_bounds[1] * nrows
#                 h = bbox_bounds[3] * nrows

#                 x1 = x - w/2
#                 x2 = x + w/2
#                 y1 = y - h/2
#                 y2 = y + h/2

#             else:
#                 # assume voc:
#                 x1,y1,x2,y2 = bbox_bounds

#             # clip to image bounds
#             x1 = int(np.clip(x1, 0, ncols-1))
#             y1 = int(np.clip(y1, 0, nrows-1))
#             x2 = int(np.clip(x2, 0, ncols-1))
#             y2 = int(np.clip(y2, 0, nrows-1))

#             # strictly speaking we should have the following but we can implement a filter instead.
#             # assert(x2>x1 and y2>y1) # check this is true! for voc
#             if x2>x1 and y2>y1:
#                 # only append if this constraint is satisfied.
#                 if datatype=='GT':
#                     data.append([cls_name, x1, y1, x2, y2])
#                 elif datatype=='Pred':
#                     data.append([cls_name, float(conf), x1,y1,x2,y2])

#         if len(data) > 0:
#             return np.vstack(data) # create an array.
#         else:
#             return data
#     else:
#         return data

# def read_img(imfile):
#     import cv2
#     return cv2.imread(imfile)[:,:,::-1]


# def read_boxes(txtfile):

#     import numpy as np
#     lines = []

#     with open(txtfile, "r") as f:
#         for line in f:
#             #print('line:',line)
#             line = line.strip()
#             line_list = line.split()
#             try:
#                 cls = line_list[4]
#                 coords = line_list[:4]
#                 val = None
#                 if cls=='BE':
#                     val=0
#                 elif cls=='suspicious':
#                     val=1
#                 elif cls=='HGD':
#                     val=2
#                 elif cls=='cancer':
#                     val=3
#                 elif cls=='polyp':
#                     val=4
#                 newline=[]
#                 newline.append(str(val))
#                 newline.extend(coords)
#                 box = np.hstack(newline).astype(np.float)
#                 box[0] = int(box[0])
#                 #print("coordinates:",box)
#                 lines.append(box)
#             except:
#                 print('ERRRRRRRRRRR')
#     return np.array(lines)


# def plot_boxes(ax, boxes, labels):
#     for b in boxes:
#         col = None
#         cls, x1, y1, x2, y2 = b
#         if cls==0:
#             col='cyan'
#         elif cls==1:
#             col='blue'
#         elif cls==2:
#             col='green'
#         elif cls==3:
#             col='red'
#         elif cls==4:
#             col='black'
#         print(x1,y1,x2,y2)
#         y1=512-y1
#         y2=512-y2
#         print(x1,y1,x2,y2)
#         x1=int(np.clip(x1, 0, 224-1))
#         x2=int(np.clip(x2, 0, 224-1))
#         y1=int(np.clip(y1, 0, 224-1))
#         y2=int(np.clip(y2, 0, 224-1))
#         print(x1,y1,x2,y2)
#         ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1],lw=2, color=col)
#     return []


# def read_obj_names(textfile):

#     import numpy as np
#     classnames = []

#     with open(textfile) as f:
#         for line in f:
#             line = line.strip('\n')
#             if len(line)>0:
#                 classnames.append(line)

#     return np.hstack(classnames)

# def plt_rectangle(plt,label,x1,y1,x2,y2):
#   '''
#   plt   : matplotlib.pyplot object
#   label : string containing the object class name
#   x1    : top left corner x coordinate
#   y1    : top left corner y coordinate
#   x2    : bottom right corner x coordinate
#   y2    : bottom right corner y coordinate
#   '''
#   linewidth = 3
#   color = "yellow"
#   plt.text(x1,y1,label,fontsize=20,backgroundcolor="magenta")
#   plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color)
#   plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color)
#   plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color)
#   plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color)


# if __name__=="__main__":

#     """
#     Example script to read and plot bounding box  (which are provided in <x1,y1,x2,y2> (VOC)format)
#     """
#     import pylab as plt
#     import sys
#     imgfile='./EDD2020/EDD2020_release-I_2020-01-15/originalImages/EDD2020_ACB0001.jpg'
#     bboxfile='./EDD2020/EDD2020_release-I_2020-01-15/bbox/EDD2020_ACB0001.txt'
#     masksfile1='./EDD2020/EDD2020_release-I_2020-01-15/masks/EDD2020_ACB0001_BE.tif'
#     masksfile2='./EDD2020/EDD2020_release-I_2020-01-15/masks/EDD2020_ACB0001_suspicious.tif'
#     classfile = './EDD2020/EDD2020_release-I_2020-01-15/class_list.txt'

#     print('TEST')
#     img = read_img(imgfile)
#     img = resize_image_to_square(img, 224, pad_cval=0)
#     boxes = read_boxes(bboxfile)
#     mask1 = read_img(masksfile1)
#     mask1 = resize_image_to_square(mask1, 224, pad_cval=0)
#     mask2 = read_img(masksfile2)
#     mask2 = resize_image_to_square(mask2, 224, pad_cval=0)
#     classes = read_obj_names(classfile)
#     boxes=convert_boxes(boxes,['0','1','2','3','4'],'GT',img.shape[:2])
#     boxes=boxes.astype(np.float)
#     fig, ax = plt.subplots(nrows=1,ncols=3)
#     plot_boxes(ax[0], boxes, classes)
#     ax[0].imshow(img)
#     ax[1].imshow(mask1)
#     ax[2].imshow(mask2)
#     plt.show()
