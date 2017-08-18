import cv2
import numpy as np
import scipy.io as sio
import os
import sys
import yaml
import h5py

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# parrots_home = os.environ.get('PARROTS_HOME')
# sys.path.append(os.path.join(parrots_home, 'parrots', 'python'))

# from pyparrots.dnn import Session, Model, config
from parrots.dnn import Session, Model, config

sys.path.append('/home/yhcao6')
import ext_layer

from parrots import base
base.set_debug_log(True)

def model_and_session(model_file, session_file):
    with open(model_file) as fin:
        model_text = fin.read()
    with open(session_file) as fin:
        session_cfg = yaml.load(fin.read(), Loader=Loader)
    session_cfg = config.ConfigDict(session_cfg)
    session_cfg = config.ConfigDict.to_dict(session_cfg)
    session_text = yaml.dump(session_cfg, Dumper=Dumper)

    model = Model.from_yaml_text(model_text)
    session = Session.from_yaml_text(model, session_text)

    return model, session


class Tester():
    def __init__(self, model, session, param):
        self.model, self.session = model_and_session(model, session)
        self.session.setup()
        self.flow = self.session.flow('val')
        self.flow.load_param(param)

    def predict(self, inputs, query):
        for k in inputs.keys():
            self.flow.set_input(k, inputs[k])
        self.flow.forward()
        # np.set_printoptions(threshold='nan')
        # print self.flow.data('data_res075').value().T[0, 0, 0]
        # f = h5py.File('1.h5', 'w')
        # f.create_dataset('parrots_data', data=self.flow.data('data').value().T[0])
        # f.create_dataset('parrots_data_res05', data=self.flow.data('data_res05').value().T[0])
        # f.create_dataset('parrots_data_res075', data=self.flow.data('data_res075').value().T[0])
        # print self.flow.data('data').value().T[0, 0, 0].shape

        return self.flow.data(query).value().T


def load_color_LUT_21(fn):
    contents = sio.loadmat(fn)
    return contents.values()[0]


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def test_eval_seg(model, session, param, test_list, data_root, gt_root, query, mean=np.array([104.008, 116.669, 122.675]), batch_size=1, uniform_size=513, show=False, save=False):

    with open(test_list) as infile:
        img_list = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

    base_size=[uniform_size, uniform_size]

    tester = Tester(model, session, param)

    hist = np.zeros((21, 21))
    for i in range(0, len(img_list), batch_size):
    # for i in range(0, 1, batch_size):
        if i % (batch_size*100) == 0:
            print 'Processing: %d/%d' % (i, len(img_list))
        true_batch_size = min(batch_size, len(img_list)-i)
        # resize input into 0.5, 0.75, 1 resolution
        batch_data = np.zeros((batch_size, 3, base_size[0], base_size[1]), dtype=np.float)
        for k in range(true_batch_size):
            # put img into center of a 513 x 513 x 3 array

            img = cv2.imread(data_root + img_list[i + k].split(' ')[0])
            # print 'original_shape is', img.shape

            img = np.float32(img)

            pad_height = max(513 - img.shape[0], 0)
            pad_width = max(513 - img.shape[1], 0)
            if pad_height > 0 or pad_width > 0:
                img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=mean)

            img -= mean

            # N x C x H x W
            batch_data[k, ...] = img.transpose((2, 0, 1))

        # feed data
        data = {}
        data['data'] = batch_data.T

        # predict
        pred = tester.predict(data, query)

        for k in range(0, true_batch_size):
            origin_img = cv2.imread(data_root + img_list[i + k].split(' ')[0])
            origin_shape = origin_img.shape
            
            gt_img = cv2.imread(data_root + img_list[i + k].split(' ')[1])[:, :, 0]

            tmp = pred[k, ...].transpose((1, 2, 0))
            tmp = tmp.argmax(axis=2)
            cls_map = np.array(tmp, dtype=np.uint8)[0: origin_shape[0], 0: origin_shape[1]]

            if save:
                cv2.imwrite('/home/yhcao6/resnet_seg/res/' + img_list[i + k].split(' ')[0].split('/')[-1].split('.')[0] + '.png', cls_map)
                
            if show:
                LUT = load_color_LUT_21('./VOC_color_LUT_21.mat')
                tmp = np.zeros((256, 3))
                tmp[0:21, ...] = LUT
                tmp[255] = [0, 1, 1]
                LUT = tmp

                # visualize
                out_map = np.uint8(LUT[cls_map] * 255)
                gt_map = np.uint8(LUT[gt_img] * 255)

                both = np.hstack((out_map, gt_map))
                cv2.imshow('seg result', both)
                cv2.waitKey(0)

            hist += fast_hist(gt_img.flatten(), cls_map.flatten(), 21)

    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', 'overall accuracy', acc
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', 'mean accuracy', np.nanmean(acc)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', 'per class IU:\n', iu
    show_iu = ["{:.2f}".format(i*100) for i in iu]
    print '>>>', show_iu
    print '>>>', 'mean IU', np.nanmean(iu)


if __name__ == '__main__':
    test_eval_seg(model='/home/yhcao6/resnet_seg/test.yaml', session='/home/yhcao6/resnet_seg/val_session.yaml', param='/home/yhcao6/resnet_seg/work_dir/snapshots/iter.00020000.parrots', test_list='/home/yhcao6/val.txt', data_root='/home/yhcao6/VOC_arg', gt_root='/home/yhcao6/VOC_arg/SegmentationClass_label', query='fc_fusion', show=False, save=False, batch_size=1)








