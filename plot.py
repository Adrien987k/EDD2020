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
        for label, bbox in bboxs:
            color = self.segments_colors[label]
            xmin, ymin, xmax, ymax = bbox
            im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 2)
            if type(im) is not np.ndarray:
                im = im.get()
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
