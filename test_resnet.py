from parrots.dnn.modules import ModuleProto, GModule
from parrots.dnn.layerprotos import (Convolution, FullyConnected, Pooling, Sum, Softmax, ReLU, Dropout, SoftmaxWithLoss, Accuracy, Maxout)
from parrots.dnn import layerprotos
from parrots import base

import sys
sys.path.append('/home/yhcao6')
# interp layer
import ext_layer

# used for debug
base.set_debug_log(False)


class Interp_zoom(layerprotos.LayerProto):

    def __init__(self, zoom_factor=1):
        super(Interp_zoom, self).__init__("Interp", zoom_factor=zoom_factor)

    @property
    def num_inputs(self):
        return (1, 2)


class Interp_shrink(layerprotos.LayerProto):

    def __init__(self, shrink_factor=1, interp_type='bilinear'):
        super(Interp_shrink, self).__init__("Interp", shrink_factor=shrink_factor, interp_type=interp_type)

    @property
    def num_inputs(self):
        return (1, 2)


def BN():
    bn = layerprotos.BN(frozen=True)
    bn.param_policies[0] = {'lr_mult': 0}
    bn.param_policies[1] = {'lr_mult': 0}
    bn.param_policies[2] = {'lr_mult': 0}
    return bn


class Bottleneck(ModuleProto):

    def __init__(self, out_channels, stride=1, shortcut_type='identity', hole=1, stride_2=False, is_part4=False, is_part5=False):
        assert shortcut_type in ['identity', 'conv']
        self.out_channels = out_channels
        self.stride = stride
        self.shortcut_type = shortcut_type
        self.hole = hole
        self.is_part4 = is_part4
        self.is_part5 = is_part5
        self.stride_2 = stride_2

    def construct(self, m):
        out_channels = self.out_channels
        x = m.input_slot('x')

        if self.stride_2:
            r = x.to(Convolution(1, out_channels, pad=0, stride=2, bias=False), name='conv1')
        else:
            r = x.to(Convolution(1, out_channels, bias=False), name='conv1')
        r = r.to(BN(), name='bn1', recurrable=True)
        r = r.to(ReLU(), inplace=True, name='relu1')
        if self.is_part4:
            r = r.to(Convolution(
                3, out_channels, stride=1, pad=2, hole=2, bias=False),
                     name='conv2')
        elif self.is_part5:
            r = r.to(Convolution(
                3, out_channels, stride=1, pad=4, hole=4, bias=False),
                     name='conv2')
        else:
            r = r.to(Convolution(
                3, out_channels, stride=1, pad=1, hole=1, bias=False),
                     name='conv2')
        r = r.to(BN(), name='bn2', recurrable=True)
        r = r.to(ReLU(), inplace=True, name='relu2')
        r = r.to(Convolution(1, out_channels * 4, bias=False), name='conv3')
        r = r.to(BN(), name='bn3', recurrable=True)
        if self.shortcut_type == 'conv':
            if self.stride_2:
                x = x.to(Convolution(1, out_channels * 4, stride=2, bias=False), name='shortcut')
            else:
                x = x.to(Convolution(1, out_channels * 4, stride=1, bias=False), name='shortcut')
            x = x.to(BN(), name='shortcut_bn', recurrable=True)
        x = m.vars(x, r).to(Sum(), name='sum')
        x = x.to(ReLU(), inplace=True, name='relu')

        m.output_slots = x.name


# one branch
def one_resolution(main, size):

    if size == '':
        size = ''
    else:
        size = '_res' + size
    x = main.var('data' + size)
    x = x.to(Convolution(7, 64, stride=2, pad=3, bias=False, w_policy={'lr_mult': 1, 'decay_mult': 1}), name='conv1' + size, paramgrp='conv1')
    # batch_norm_param ?
    x = x.to(BN(), name='bn1' + size, paramgrp='bn1', recurrable=True)
    x = x.to(ReLU(), inplace=True, name='relu1' + size)
    x = x.to(Pooling('max', 3, pad=1, stride=2), name='pool1' + size)

    block = Bottleneck

    # part2
    x = x.to(block(64, 1, 'conv'), name='res2a' + size, paramgrp='res2a')
    for j in range(1, 3):
        x = x.to(block(64, 1), name='res2b{}'.format(j) + size, paramgrp='res2b{}'.format(j))

    # part3
    x = x.to(block(128, 2, 'conv', stride_2=True), name='res3a' + size, paramgrp='res3a')
    for j in range(1, 4):
        x = x.to(block(128, 1), name='res3b{}'.format(j) + size, paramgrp='res3b{}'.format(j))

    # # part4
    x = x.to(block(256, 1, 'conv', hole=2, is_part4=True), name='res4a' + size, paramgrp='res4a')
    for j in range(1, 23):
        x = x.to(block(256, 1, hole=2, is_part4=True), name='res4b{}'.format(j) + size, paramgrp='res4b{}'.format(j))

    # # part5
    x = x.to(block(512, 1, 'conv', hole=4, is_part5=True), name='res5a' + size, paramgrp='res5a')
    for j in range(1, 3):
        x = x.to(block(512, 1, hole=4, is_part5=True), name='res5b{}'.format(j) + size, paramgrp='res5b{}'.format(j))

    # # classifiers
    fc1_c0 = x.to(Convolution(3, 21, pad=6, hole=6, w_policy={'init': 'gauss(0.01)', 'lr_mult': 10, 'decay_mult': 1}, b_policy={'init': 'fill(0)', 'lr_mult': 20, 'decay_mult': 0}), name='fc1_c0' + size, paramgrp='fc1_c0')

    fc1_c1 = x.to(Convolution(3, 21, pad=12, hole=12, w_policy={'init': 'gauss(0.01)', 'lr_mult': 10, 'decay_mult': 1}, b_policy={'init': 'fill(0)', 'lr_mult': 20, 'decay_mult': 0}), name='fc1_c1' + size, paramgrp='fc1_c1')

    fc1_c2 = x.to(Convolution(3, 21, pad=18, hole=18, w_policy={'init': 'gauss(0.01)', 'lr_mult': 10, 'decay_mult': 1}, b_policy={'init': 'fill(0)', 'lr_mult': 20, 'decay_mult': 0}), name='fc1_c2' + size, paramgrp='fc1_c2')
    fc1_c3 = x.to(Convolution(3, 21, pad=24, hole=24, w_policy={'init': 'gauss(0.01)', 'lr_mult': 10, 'decay_mult': 1}, b_policy={'init': 'fill(0)', 'lr_mult': 20, 'decay_mult': 0}), name='fc1_c3' + size, paramgrp='fc1_c3')

    x = main.vars(fc1_c0, fc1_c1, fc1_c2, fc1_c3).to(Sum(), name='fc1' + size)

    return x


def create_model(depth=101, input_size=513, num_classes=21, name=None):

    if name is None:
        name = 'resnet-v1-{}'.format(depth)
    main = GModule(name)

    inputs = {
        'data': 'float32({}, {}, 3, _)'.format(input_size, input_size)
    }

    main.input_slots = tuple(inputs.keys())

    # resize image
    # for res075, firstly zoom, then shrink
    x = main.var('data')
    x_05 = x.to(Interp_shrink(shrink_factor=2), name='data_res05')
    x_075_tmp = x.to(Interp_zoom(zoom_factor=3), name='data_res075_tmp')
    x_075 = x_075_tmp.to(Interp_shrink(shrink_factor=4), name='data_res075')

    # three branches
    x = one_resolution(main, '')
    x_05 = one_resolution(main, '05')
    x_075 = one_resolution(main, '075')

    # zoom to input/4+1 size
    x_05_zoom = x_05.to(Interp_zoom(zoom_factor=2), name='fc1_res05_interp')
    x_075_interp_tmp = x_075.to(Interp_zoom(zoom_factor=4), name='fc1_res075_interp_tmp')
    x_075_zoom = x_075_interp_tmp.to(Interp_shrink(shrink_factor=3), name='fc1_res075_interp')

    # fusion three branches
    # in test_net, zoom predict 8 to compare with ground truth
    x_fusion_tmp = main.vars(x, x_05_zoom, x_075_zoom).to(Maxout(), name='fc_fusion_tmp')
    x_fusion = x_fusion_tmp.to(Interp_zoom(zoom_factor=8), name='fc_fusion')
    model = main.compile(inputs=inputs, seal=False)
    model.add_flow('main', inputs.keys(), ['fu_fusion'])

    model.seal()
    return model


if __name__ == '__main__':
    test_model = create_model(101)
    with open('test.yaml', 'w+') as test_file:
        test_file.write(test_model.to_yaml_text())
