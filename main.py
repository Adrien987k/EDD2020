import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from argparse import ArgumentParser

from functools import reduce
from models.unet import Unet
from models.unetplus import UnetPlus
from models.unetresnet import UNetResNet

import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import Trainer
from trainer import calc_loss, compute_metrics, print_metrics

from plot import Plot
from dataloader import get_edd_loader
from util import (resize_my_images, create_dir, to_numpy,
                  save_to_tif, compute_bboxs_from_masks,
                  bbox_tensor_to_bbox)


def get_model(model_name, in_channel, n_classes):
    return {
        'unetresnet': UNetResNet(in_channel, n_classes),
        'unet': Unet(in_channel, n_classes),
        'unetplus': UnetPlus(in_channel, n_classes)
    }[model_name]


def main():
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./EDD2020/')  # './'
    parser.add_argument('--model', type=str, default='unetresnet')
    args = parser.parse_args()

    base_dir = args.base_dir
    model_name = args.model

    success = create_dir(base_dir + 'resized_masks/')
    if success:
        resize_my_images(base_dir + 'EDD2020_release-I_2020-01-15/masks/',
                         base_dir + 'resized_masks/',
                         is_masks=True)

    success = create_dir(base_dir + 'resized_images/')
    success &= create_dir(base_dir + 'resized_bboxs/')
    if success:
        resize_my_images(base_dir + 'EDD2020_release-I_2020-01-15/originalImages/',
                         base_dir + 'resized_images/',
                         is_masks=False,
                         bboxs_src=base_dir + 'EDD2020_release-I_2020-01-15/bbox/',
                         bboxs_dst=base_dir + 'resized_bboxs/')

    loader = get_edd_loader(base_dir, shuffle_dataset=True)

    model = get_model(model_name, in_channel=3, n_classes=5)
    optimizer_func = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer_func, step_size=10, gamma=0.1)

    trainer = Trainer(model, optimizer=optimizer_func, scheduler=scheduler)
    trainer.train_model(loader, num_epochs=30)

    create_dir(base_dir + 'test/')
    create_dir(base_dir + 'test/images')
    create_dir(base_dir + 'test/masks')
    create_dir(base_dir + 'test/masks_pred')
    create_dir(base_dir + 'test/bboxs')
    create_dir(base_dir + 'test/bboxs_pred')
    create_dir(base_dir + 'test/plots')

    metrics = defaultdict(float)
    plot = Plot(base_dir + 'test/plots/')

    index = 0
    for epoch, (images, bboxs, masks) in enumerate(loader['test']):
        masks_preds = trainer.predict(images)

        # print('B ', bboxs)
        bboxs = bbox_tensor_to_bbox(bboxs.squeeze(0))
        # print('A ', bboxs)

        bboxs_preds = compute_bboxs_from_masks(masks_preds.squeeze(0))

        plot.plot_image_truemask_predictedmask(
            images, masks, masks_preds, index)
        plot.plot_image_truebbox_predictedbbox(
            images, bboxs, bboxs_preds, index)

        calc_loss(torch.Tensor(masks_preds), masks, metrics)

        image, mask, mask_pred = images[0], masks[0], masks_preds[0]

        save_to_tif(
            base_dir + 'test/masks/MASK_{:03d}.tif'.format(index),
            to_numpy(mask))

        save_to_tif(
            base_dir + 'test/masks_pred/MASK_PRED_{:03d}.tif'.format(index),
            mask_pred)

        image = image.detach().cpu().numpy().swapaxes(0, 2).swapaxes(0, 1)
        plt.imsave(
            base_dir + 'test/images/IMG_{:03d}.png'.format(index), image)

        index += 1

    computed_metrics = compute_metrics(metrics, epoch + 1)
    print_metrics(computed_metrics, 'test')


if __name__ == '__main__':
    main()
