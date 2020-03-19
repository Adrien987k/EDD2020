import torch

from functools import reduce
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from meditorch.nn.models import UNetResNet

from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
from meditorch.nn import Trainer
from meditorch.nn.trainer import calc_loss, compute_metrics, print_metrics

# from meditorch.utils.plot import plot_image_truemask_predictedmask
import numpy as np
from EDD import EDD
from util import resize_my_images


from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from skimage.measure import label, regionprops

np.random.seed(42)


########################


def metrics_line(data):
    phases = list(data.keys())
    metrics = list(data[phases[0]][0].keys())

    i = 0
    fig, axs = plt.subplots(1, len(metrics))
    fig.set_figheight(4)
    fig.set_figwidth(4*len(metrics))
    for metric in metrics:
        for phase in phases:
            axs[i].plot([i[metric] for i in data[phase]], label=phase)
        axs[i].set_title(metric)
        i += 1

    plt.legend()
    plt.show()


def normalise_mask(mask, threshold=0.5):
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    return mask


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp


def plot_img_array(img_array, ncol=3, index=None):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all',
                            sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        # plots[i // ncol, i % ncol]
        plots[i].imshow(img_array[i])

    plt.savefig('./EDD2020/test/plots/PLOT_{:03d}.png'.format(index))


def plot_image_truemask_predictedmask(images, labels, preds, index):

    input_images_rgb = [reverse_transform(x) for x in images]
    target_masks_rgb = [masks_to_coloredmasks(x) for x in labels]
    pred_rgb = [masks_to_coloredmasks(x) for x in preds]

    img_arrays = [input_images_rgb, target_masks_rgb, pred_rgb]
    flatten_list = reduce(lambda x, y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays), index=index)


def draw_bboxs(im, bboxs):
    segments_colors = np.asarray(
        [(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228), (56, 34, 132), (160, 194, 56)])

    for label, bbox in bboxs:
        color = segments_colors[label]
        x, y, w, h = bbox
        im = cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)

    return im


def plot_image_truebbox_predictedbbox(images, bboxs, bboxs_preds, index):

    input_images_rgb = [reverse_transform(x) for x in images]

    images_with_real_masks = [draw_bboxs(im, bboxs)
                              for im, bboxs in zip(images, bboxs)]
    images_with_pred_masks = [draw_bboxs(im, bboxs)
                              for im, bboxs in zip(images, bboxs_preds)]

    img_arrays = [input_images_rgb,
                  images_with_real_masks, images_with_pred_masks]

    flatten_list = reduce(lambda x, y: x+y, zip(*img_arrays))
    plot_img_array(np.array(flatten_list), ncol=len(img_arrays), index=index)


def apply_mask_color(mask, mask_color):
    colored_mask = np.concatenate(
        ([mask[..., np.newaxis] * color for color in mask_color]), axis=2)
    return colored_mask.astype(np.uint8)


def masks_to_coloredmasks(mask, normalise=True, colors=None):
    segments_colors = np.asarray(
        [(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228), (56, 34, 132), (160, 194, 56)])
    if colors is not None:
        segments_colors = colors

    if normalise:
        normalise_mask(mask)

    mask_colored = np.concatenate(
        [[apply_mask_color(mask[i], segments_colors[i])] for i in range(len(mask))])
    mask_colored = np.max(mask_colored, axis=0)

    mask_colored = np.where(
        mask_colored.any(-1, keepdims=True), mask_colored, 255)

    return mask_colored

#########################


def get_edd_loader(path, validation_split=.10, test_split=.10, shuffle_dataset=True):
    dataset = EDD(path)  # instantiating the data set.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split_val = int(np.floor(validation_split * dataset_size))
    split_test = int(np.floor(test_split * dataset_size))

    if shuffle_dataset:
        np.random.shuffle(indices)

    train_indices = indices[split_val + split_test:]
    val_indices = indices[split_test:split_test + split_val]
    test_indices = indices[:split_test]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    loader = {
        'train': DataLoader(dataset, batch_size=2, sampler=train_sampler),
        'val': DataLoader(dataset, batch_size=2, sampler=valid_sampler),
        'test': DataLoader(dataset, batch_size=1, sampler=test_sampler)
    }
    return loader


def create_dir(dirname):
    try:
        os.mkdir(dirname)
    except OSError:
        pass


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def save_to_tif(path, data):
    with open(path, 'wb') as f:
        np.save(f, data, allow_pickle=True)


def main():

    np.random.seed(42)
    # seting up the data set

    # create_dir('./EDD2020/resized_masks/')
    # resize_my_images('./EDD2020/EDD2020_release-I_2020-01-15/masks/',
    #                  './EDD2020/resized_masks/', is_masks=True)

    # create_dir('./EDD2020/resized_images/')
    # resize_my_images('./EDD2020/EDD2020_release-I_2020-01-15/originalImages/',
    #                  './EDD2020/resized_images/', is_masks=False)

    loader = get_edd_loader('./EDD2020/', shuffle_dataset=True)

    # using UNet+ResNet combo
    model = UNetResNet(in_channel=3, n_classes=5)
    optimizer_func = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer_func, step_size=10, gamma=0.1)
    trainer = Trainer(model, optimizer=optimizer_func, scheduler=scheduler)
    # training
    trainer.train_model(loader, num_epochs=30)

    # images, masks = next(iter(loader['test']))

    create_dir('./EDD2020/test/')
    create_dir('./EDD2020/test/images')
    create_dir('./EDD2020/test/masks')
    create_dir('./EDD2020/test/masks_pred')
    create_dir('./EDD2020/test/bboxs')
    create_dir('./EDD2020/test/bboxs_pred')
    create_dir('./EDD2020/test/plots')

    metrics = defaultdict(float)

    index = 0
    for epoch, (images, masks) in enumerate(loader['test']):
        masks_preds = trainer.predict(images)

        bboxs = ...
        bboxs_preds = ...

        plot_image_truemask_predictedmask(images, masks, masks_preds, index)
        plot_image_truebbox_predictedbbox(images, bboxs, bboxs_preds, index)

        calc_loss(torch.Tensor(masks_preds), masks, metrics)

        for image, mask, mask_pred in zip(images, masks, masks_preds):
            save_to_tif(
                './EDD2020/test/masks/MASK_{:03d}.tif'.format(index),
                to_numpy(mask))

            save_to_tif(
                './EDD2020/test/masks_pred/MASK_PRED_{:03d}.tif'.format(index),
                mask_pred)

            image = image.detach().cpu().numpy().swapaxes(0, 2).swapaxes(0, 1)
            plt.imsave(
                './EDD2020/test/images/IMG_{:03d}.png'.format(index), image)

            # Compute BBOX ###########
            lbl_0 = label(mask)
            props = regionprops(lbl_0)

            #img_1 = img_0.copy()
            #print ('Image', image)

            for prop in props:
                print('Found bbox', prop.bbox)
                cv2.rectangle(
                    img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)

            ax1.imshow(img_0)
            ax1.set_title('Image')
            ax2.set_title('Mask')
            ax3.set_title('Image with derived bounding box')
            ax2.imshow(mask[..., 0], cmap='gray')
            ax3.imshow(img_1)
            plt.show()
            ##########################

            index += 1

    computed_metrics = compute_metrics(metrics, epoch + 1)
    print_metrics(computed_metrics, 'test')


if __name__ == '__main__':
    main()
