import numpy as np
import random
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import cv2
import scipy.io as sio


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


def crop(im_data, im_label, new_h, new_w):

    # print 'original size is', im_data.shape
    # cv2.imshow('original', im_data)

    h = im_data.shape[0]
    w = im_data.shape[1]
    if h <= new_h:
        up = 0
        len_h = h
        off_u = (new_h - h) / 2
        off_d = new_h - h - off_u
    else:
        up = random.randint(0, (h - new_h)/2)
        len_h = new_h
        off_u = 0
        off_d = 0

    if w <= new_w:
        left = 0
        len_w = w
        off_l = (new_w - w) / 2
        off_r = new_w - w - off_l
    else:
        left = random.randint(0, (w - new_w)/2)
        len_w = new_w
        off_l = 0
        off_r = 0

    im_data = im_data[up: up+len_h, left: left+len_w]
    im_label = im_label[up: up+len_h, left: left+len_w]

    im_data = cv2.copyMakeBorder(im_data, off_u, off_d, off_l, off_r, cv2.BORDER_CONSTANT, value = (104, 116, 122))
    im_label = cv2.copyMakeBorder(im_label, off_u, off_d, off_l, off_r, cv2.BORDER_CONSTANT, value=255)

    # cv2.imshow('crop', im_data)

    return im_data, im_label


# a generator which generate data and corresponding label for batch_size numer
def transform(source, data_root, label_root, mean, crop_height, crop_width, is_shuffle, is_mirror):
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
            resize_scale = random.random()*(1.0-0.8) + 0.8
            im_data = cv2.resize(im_data, (int(resize_scale*im_data.shape[1]), int(resize_scale*im_data.shape[0])), interpolation=cv2.INTER_LINEAR )
            im_label = cv2.resize(im_label, (int(resize_scale*im_label.shape[1]), int(resize_scale*im_label.shape[0])), interpolation=cv2.INTER_NEAREST)

            # mirror image
            if is_mirror:
                if random.randint(0, 1) == 0:
                    im_data = cv2.flip(im_data, 1)
                    im_label = cv2.flip(im_label, 1)

            # crop data and corresponding label
            im_data, im_label = crop(im_data, im_label, crop_height, crop_width)

            # cv2.imshow('crop', im_data)

            # process data image
            in_data = np.array(im_data, dtype=np.float32)  # cast image to float
            in_data -= mean  # subtract mean
            in_data = in_data.transpose((2, 0, 1))  # transpose to channel x height x width

            # process label image
            in_label = np.array(im_label, dtype=np.uint8)

            # shrink 8
            loss_height = int(np.ceil(crop_height/8.0))
            loss_width = int(np.ceil(crop_width/8.0))
            in_label = cv2.resize(in_label, (loss_height, loss_width), interpolation=cv2.INTER_NEAREST)
            in_label = in_label[np.newaxis, :, :]

            # mask 
            ignore = in_label == 255
            in_label[ignore] = 0

            # print in_label.max(), in_label.min()
            # print in_label.max()
            assert in_label.max() < 21

            in_label_w = in_label.copy()
            in_label_w[ignore] = 0
            in_label_w[~ignore] = 1


            # cv2.imshow('im', im)
            # cv2.waitKey(0)
            # print in_data.shape, in_label.shape, in_label_w.shape
            # cv2.destroyAllWindows

            yield [in_data.T, in_label.T, in_label_w.T]


class SegReader:
    support_keys = ['source', 'data_root', 'label_root', 'crop_height',
                    'crop_width', 'is_shuffle', 'is_mirror']

    def config(self, cfg):
        source = cfg['source']
        data_root = cfg['data_root']
        label_root = cfg['label_root']
        mean = (104.00698793, 116.66876762, 122.67891434)
        crop_height = cfg.get('crop_height', 321)
        crop_width = cfg.get('crop_width', 321)
        is_shuffle = cfg.get('is_shuffle', True)
        is_mirror = cfg.get('is_mirror', True)
        self.gen = transform(source, data_root, label_root,
                             mean, crop_height, crop_width, is_shuffle, is_mirror)

    def read(self):
        return self.gen.next()


def test():

    reader = SegReader()
    reader.config({
        # 'source': '/home/yhcao6/VOC_arg/train.txt',
        'source': '/home/yhcao6/VOC_arg/train.txt',
        'data_root': '/home/yhcao6/VOC_arg',
        'label_root': '/home/yhcao6/VOC_arg'
        })

    # gen = transform(args.source, args.data_root, args.label_root, eval(args.mean), args.crop_height,
    #                 args.crop_width, args.is_shuffle, args.is_mirror)

    # data_mat, label_mat = gen.next()

    for i in range(10):
        data_mat, label_mat, label_w_mat = reader.read()
        im_data = data_mat.T
        im_label = label_mat.T

        mean = (104.00698793, 116.66876762, 122.67891434)
        im_data = im_data.transpose(1, 2, 0) + mean
        im_data = np.array(im_data, dtype=np.uint8)

        im_label.transpose(1, 2, 0)
        mask = label_w_mat.T
        a = (mask == 0)
        im_label[a] = 255
        LUT = sio.loadmat('./VOC_color_LUT_21.mat').values()[0]
        tmp = np.zeros((256, 3))
        tmp[0: 21, ...] = LUT
        tmp[255] = [0, 1, 1]
        LUT = tmp
        im_label = np.uint8(LUT[im_label]*255)
        im_label = im_label.transpose(1, 2, 3, 0)
        im_label = im_label[:, :, :, 0]
        print im_label.shape
        
        # im = np.concatenate((im_data, im_label), axis=1)
        # cv2.imwrite('./images/'+str(i)+'.jpg', im_label)

        # TODO: check ignore is set right


if __name__ == '__main__':
    test()
else:
    import os
    import sys
    parrots_home = os.environ.get('PARROTS_HOME')
    if not parrots_home:
        raise EnvironmentError(
            'The environment variable "PARROTS_HOME" is not set.')
    sys.path.append(os.path.join(parrots_home, 'parrots/python'))
    from parrots.dnn import reader
    reader.register_pyreader(SegReader, 'seg_reader')
