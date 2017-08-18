import numpy as np
import random
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import cv2
import scipy.io as sio


def load_color_LUT_21(fn):
    contents = sio.loadmat(fn)
    return contents.values()[0]


def get_list(filename):
    # [[data, label] ...]
    l = []
    # get data and label filename list
    with open(filename) as f:
        for line in f:
            d, la = line.split(' ')[0: 2]
            if la[-1] == '\n':
                la = la[:-1]
            l.append((d, la))
    return l


# a generator which generate data and corresponding label for batch_size numer
def transform(source, data_root, label_root, mean, crop_height, crop_width, is_shuffle, is_mirror, phase, show):
    # get_list
    l = get_list(source)

    while True:
        # shuffle
        if is_shuffle:
            np.random.shuffle(l)

        for data_source, label_source in l:
            # load image
            im_data = cv2.imread('{}/{}'.format(data_root, data_source))
            im_label = cv2.imread('{}/{}'.format(label_root, label_source))[:, :, 0]

            # resize image
            if phase == 'train':
                scale = [0.5, 0.75, 1, 1.25, 1.5]
                resize_scale = np.random.choice(scale, 1)
                im_data = cv2.resize(im_data, (im_data.shape[1]*resize_scale, im_data.shape[0]*resize_scale), interpolation=cv2.INTER_LINEAR)
                im_label = cv2.resize(im_label, (im_label.shape[1]*resize_scale, im_label.shape[0]*resize_scale), interpolation=cv2.INTER_NEAREST)
                if show:
                    print 'resize_scale is', resize_scale
                    resize_data = np.uint8(im_data)
                    cv2.imshow('after resize', resize_data)

            # convert to np.float32
            im_data = np.float32(im_data)

            # pad image
            pad_height = max(crop_height - im_data.shape[0], 0)
            pad_width = max(crop_width - im_data.shape[1], 0)

            if pad_height > 0 or pad_width > 0:
                im_data = cv2.copyMakeBorder(im_data, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=mean)
                im_label = cv2.copyMakeBorder(im_label, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=255)
            if show:
                print 'pad hight is', pad_height, 'pad width is', pad_width
                pad_data = np.uint8(im_data)
                cv2.imshow('after pad', pad_data)

            # crop
            if phase == 'train':
                h_off = np.random.randint(im_data.shape[0] - crop_height + 1)
                w_off = np.random.randint(im_data.shape[1] - crop_width + 1)
            else:
                h_off = (im_data.shape[0] - crop_height) / 2
                w_off = (im_data.shape[1] - crop_width) / 2

            im_data = im_data[h_off: h_off + crop_height, w_off: w_off + crop_width, :]
            im_label = im_label[h_off: h_off + crop_height, w_off: w_off + crop_width]

            # mirror image
            if is_mirror:
                if random.randint(0, 1) == 0:
                    im_data = cv2.flip(im_data, 1)
                    im_label = cv2.flip(im_label, 1)

            # process data image
            im_data -= mean  # subtract mean

            if phase == 'train':
               
                im_data = im_data.transpose((2, 0, 1))  # transpose to channel x height x width
                # process label image
                im_label = np.array(im_label, dtype=np.uint8)
                im_label = im_label[np.newaxis, :, :]

                # mask
                ignore = im_label == 255
                im_label[ignore] = 0

                assert im_label.max() < 21

                im_label_w = im_label.copy()
                im_label_w[ignore] = 0
                im_label_w[~ignore] = 1

                yield [im_data.T, im_label.T, im_label_w.T]


class SegReader:
    support_keys = ['source', 'data_root', 'label_root', 'crop_height',
                    'crop_width', 'is_shuffle', 'is_mirror']

    def config(self, cfg):
        source = cfg['source']
        data_root = cfg['data_root']
        label_root = cfg['label_root']
        # mean = (104.00698793, 116.66876762, 122.67891434)
        mean = (104.008, 116.669, 122.675)
        crop_height = cfg.get('crop_height', 321)
        crop_width = cfg.get('crop_width', 321)
        is_shuffle = cfg.get('is_shuffle', True)
        is_mirror = cfg.get('is_mirror', True)
        phase = cfg.get('phase', 'train')
        self.gen = transform(source, data_root, label_root,
                             mean, crop_height, crop_width, is_shuffle, is_mirror, phase, False)

    def read(self):
        return self.gen.next()


def test():

    reader = SegReader()
    reader.config({
        'source': '/home/yhcao6/VOC_arg/train.txt',
        'data_root': '/home/yhcao6/VOC_arg',
        'label_root': '/home/yhcao6/VOC_arg'
        })

    for i in range(10):
        reader.read()


if __name__ == '__main__':
    test()
else:
    from parrots.dnn import reader
    reader.register_pyreader(SegReader, 'seg_reader')
