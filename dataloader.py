from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch
import os
import numpy as np
import glob
from PIL import Image
from util import load_bboxs, load_set


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


class EDD(Dataset):
    '''
    Class for preparing the EDD2020 dataset
    '''

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.original_images = None
        self.masks = None
        self.labels = None
        self.bboxs = None
        self._extract_images_and_segments(root)

    def __getitem__(self, index):
        img = self.original_images[index]
        mask = self.masks[index]
        label = self.labels[index]
        label = torch.as_tensor(label, dtype=torch.int32)

        bboxs_this = self.bboxs[index]

        bboxs_tensor = []
        num_max_bboxs = 10
        num_bboxs = len(bboxs_this)
        for i in range(num_max_bboxs):
            if i < num_bboxs:
                bbox = bboxs_this[i]
                lab, values = bbox
                bboxs_tensor.append(
                    [lab, values[0], values[1], values[2], values[3]])
            else:
                bboxs_tensor.append(
                    [-1, -1, -1, -1, -1])
        bboxs_tensor = torch.as_tensor(bboxs_tensor, dtype=torch.int32)

        if self.transform:
            img = self.transform(img)
        else:
            transform_to_tensor = transforms.Compose([transforms.ToTensor(), ])
            img = transform_to_tensor(img)

        return img, bboxs_tensor, mask

    def __len__(self):
        return len(self.original_images)

    def _extract_images_and_segments(self, global_path):
        '''
        Function to process images and their respective masks.
        It sets  self.original_images and self.masks to processed images at the end.
        '''
        images_path = os.path.join(global_path, 'resized_images')
        all_images, img_filenames = load_set(folder=images_path, is_mask=False)
        img_filenames_with_ext = [os.path.split(fn)[-1] for fn in img_filenames]
        img_filenames_wo_ext = [fn[:fn.rfind('.')]
                                for fn in img_filenames_with_ext]

        bboxs_path = os.path.join(global_path, 'resized_bboxs')
        bboxs_filenames = [os.path.join(bboxs_path, filename +
                                        '.txt') for filename in img_filenames_wo_ext]
        all_bboxs = [load_bboxs(path) for path in bboxs_filenames]

        classes = ['BE', 'suspicious', 'HGD', 'cancer', 'polyp']

        masks_path = os.path.join(global_path, 'resized_masks')
        all_masks, mask_filenames = load_set(folder=masks_path, is_mask=True)
        mask_filenames_with_ext = [
            os.path.split(fn)[-1] for fn in mask_filenames]
        mask_filenames_wo_ext = [
            fn[:fn.rfind('.')] for fn in mask_filenames_with_ext]
        temp_dict = {}  # contains 502 mask filenames as keys and respective masks as values
        for i in range(len(all_masks)):
            temp_dict[mask_filenames_wo_ext[i]] = all_masks[i]

        all_masks = []
        all_labels = []
        for img in img_filenames_wo_ext:
            masks_for_img = []
            temp_labels = []
            for c in classes:
                try:
                    mask_file_name = img+'_'+c
                    temp_dict[mask_file_name] = np.where(
                        temp_dict[mask_file_name] > 0, 1, 0)
                    temp_dict[mask_file_name] = temp_dict[mask_file_name].astype(
                        np.float32)
                    masks_for_img.append(temp_dict[mask_file_name].reshape(
                        temp_dict[mask_file_name].shape + (1,)))
                    temp_labels.append(1)
                except KeyError:
                    dummy = np.zeros((224, 224)).astype(np.float32)
                    masks_for_img.append(dummy.reshape(dummy.shape + (1,)))
                    temp_labels.append(0)
            temp = None
            # temp.shape     (224, 224, 5)
            temp = np.concatenate(masks_for_img, 2)
            temp = temp.reshape((1,)+temp.shape)  # temp.shape (1, 224, 224, 5)
            all_masks.append(temp)
            all_labels.append(temp_labels)

        all_masks = np.vstack(all_masks)  # all_masks.shape (386, 224, 224, 5)
        # all_masks.shape (386, 5, 224, 224)
        all_masks = np.moveaxis(all_masks, source=3, destination=1)

        all_images = np.asarray(all_images)
        all_images = all_images.astype(np.uint8)

        print('len(all_images):', len(all_images), 'len(all_masks):',
              len(all_masks), ' len(all_labels):', len(all_labels))

        print('>>>>>>>>>>>Images<<<<<<<<<<<')
        print('type(all_images):', type(all_images),
              ' all_images.shape:', all_images.shape)
        print('type(all_images[1]):', type(all_images[1]),
              ' all_images[1].shape:', all_images[1].shape)
        print('.'*100)
        print('>>>>>>>>>>>Masks<<<<<<<<<<<<')
        print('type(all_masks):', type(all_masks),
              'all_masks.shape:', all_masks.shape)
        print('type(all_masks[1]):', type(all_masks[1]),
              'all_masks[1].shape:', all_masks[1].shape)
        print('.'*100)

        self.masks = all_masks
        self.original_images = all_images
        self.labels = all_labels
        self.bboxs = all_bboxs
