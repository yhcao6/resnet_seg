"""Original ResNet"""
import os
import sys
parrots_home = os.environ.get('PARROTS_HOME')
if not parrots_home:
    raise EnvironmentError(
        'The environment variable "PARROTS_HOME" is not set.')
sys.path.append(os.path.join(parrots_home, 'parrots/python'))

from parrots.dnn.modules import ModuleProto, GModule
from parrots.dnn.layerprotos import (Convolution, FullyConnected, Pooling, Sum, Softmax, ReLU, Dropout, SoftmaxWithLoss, Accuracy)

from parrots.dnn import layerprotos


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
        r = r.to(BN(), name='bn1')
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
        r = r.to(BN(), name='bn2')
        r = r.to(ReLU(), inplace=True, name='relu2')
        r = r.to(Convolution(1, out_channels * 4, bias=False), name='conv3')
        r = r.to(BN(), name='bn3')
        if self.shortcut_type == 'conv':
            if self.stride_2:
                x = x.to(Convolution(1, out_channels * 4, stride=2, bias=False), name='shortcut')
            else:
                x = x.to(Convolution(1, out_channels * 4, stride=1, bias=False), name='shortcut')
            x = x.to(BN(), name='shortcut_bn')
        x = m.vars(x, r).to(Sum(), name='sum')
        x = x.to(ReLU(), inplace=True, name='relu')

        m.output_slots = x.name


class BasicBlock(ModuleProto):

    def __init__(self, out_channels, stride=1, shortcut_type='identity'):
        assert shortcut_type in ['identity', 'conv']
        self.out_channels = out_channels
        self.stride = stride
        self.shortcut_type = shortcut_type

    def construct(self, m):
        stride = self.stride
        out_channels = self.out_channels
        x = m.input_slot('x')

        r = x.to(Convolution(
            3, out_channels, stride=stride, pad=1, bias=False),
                 name='conv1')
        r = r.to(BN(), name='bn1')
        r = r.to(ReLU(), inplace=True, name='relu1')
        r = r.to(Convolution(3, out_channels, pad=1, bias=False), name='conv2')
        r = r.to(BN(), name='bn2')

        if self.shortcut_type == 'conv':
            x = x.to(Convolution(
                1, out_channels, stride=stride, bias=False),
                     name='shortcut')
            x = x.to(BN(), name='shortcut_bn')
        x = m.vars(x, r).to(Sum(), name='sum')
        x = x.to(ReLU(), inplace=True, name='relu')

        m.output_slots = x.name


def create_model(depth=101, input_size=321, num_classes=21, name=None):

    cfg = {
        18: (BasicBlock, [(2, 64), (2, 128), (2, 256), (2, 512)]),
        34: (BasicBlock, [(3, 64), (4, 128), (6, 256), (3, 512)]),
        50: (Bottleneck, [(3, 64), (4, 128), (6, 256), (3, 512)]),
        101: (Bottleneck, [(3, 64), (4, 128), (23, 256), (3, 512)]),
        152: (Bottleneck, [(3, 64), (8, 128), (36, 256), (3, 512)]),
        200: (Bottleneck, [(3, 64), (24, 128), (36, 256), (3, 512)]),
    }

    assert depth in cfg

    if name is None:
        name = 'resnet-v1-{}'.format(depth)
    main = GModule(name)
    inputs = {
        'data': 'float32({}, {}, 3, _)'.format(input_size, input_size),
        'label': 'uint32(41, 41, 1, _)',
        'label_weight': 'float32(41, 41, 1, _)',
    }
    main.input_slots = tuple(inputs.keys())

    x = main.var('data')
    x = x.to(Convolution(7, 64, stride=2, pad=3, bias=False, w_policy={'lr_mult': 1, 'decay_mult': 1}), name='conv1')
    # BN lr_mult 0 lr_mult 0 lr_mult 0 how to set?
    # in caffe there is another scale layer, no need in parrots ?
    # batch_norm_param ?
    x = x.to(BN(), name='bn1')
    x = x.to(ReLU(), inplace=True, name='relu1')
    x = x.to(Pooling('max', 3, pad=1, stride=2), name='pool1')

    block, params = cfg[depth]

    # part2
    x = x.to(block(64, 1, 'conv'), name='res2a')
    for j in range(1, 3):
        x = x.to(block(64, 1), name='res2b{}'.format(j))

    # part3
    x = x.to(block(128, 2, 'conv', stride_2=True), name='res3a')
    for j in range(1, 4):
        x = x.to(block(128, 1), name='res3b{}'.format(j))

    # # part4
    x = x.to(block(256, 1, 'conv', hole=2, is_part4=True), name='res4a')
    for j in range(1, 23):
        x = x.to(block(256, 1, hole=2, is_part4=True), name='res4b{}'.format(j))

    # # part5
    x = x.to(block(512, 1, 'conv', hole=4, is_part5=True), name='res5a')
    for j in range(1, 3):
        x = x.to(block(512, 1, hole=4, is_part5=True), name='res5b{}'.format(j))

    # # classifiers
    fc1_c0 = x.to(Convolution(3, 21, pad=6, hole=6, w_policy={'init': 'gauss(0.01)', 'lr_mult': 10, 'decay_mult': 1}, b_policy={'init': 'fill(0)', 'lr_mult': 20, 'decay_mult': 0}), name='fc1_c0')

    fc1_c1 = x.to(Convolution(3, 21, pad=12, hole=12, w_policy={'init': 'gauss(0.01)', 'lr_mult': 10, 'decay_mult': 1}, b_policy={'init': 'fill(0)', 'lr_mult': 20, 'decay_mult': 0}), name='fc1_c1')

    fc1_c2 = x.to(Convolution(3, 21, pad=18, hole=18, w_policy={'init': 'gauss(0.01)', 'lr_mult': 10, 'decay_mult': 1}, b_policy={'init': 'fill(0)', 'lr_mult': 20, 'decay_mult': 0}), name='fc1_c2')
    fc1_c3 = x.to(Convolution(3, 21, pad=24, hole=24, w_policy={'init': 'gauss(0.01)', 'lr_mult': 10, 'decay_mult': 1}, b_policy={'init': 'fill(0)', 'lr_mult': 20, 'decay_mult': 0}), name='fc1_c3')

    x = main.vars(fc1_c0, fc1_c1, fc1_c2, fc1_c3).to(Sum(), name='fc1')

    main.vars(x, 'label', 'label_weight').to(SoftmaxWithLoss(axis=2), name='loss')
    # when keeping accuracy layers, it will be wrong?
    # main.vars(x, 'label').to(Accuracy(1), name='accuracy_top1')
    # main.vars(x, 'label').to(Accuracy(5), name='accuracy_top5')
    model = main.compile(inputs=inputs, seal=False)
    model.add_flow('main', inputs.keys(), ['loss', 'accuracy_top1', 'accuracy_top5'], ['loss'])
    model.seal()
    # return main.compile(inputs=inputs)
    return model


if __name__ == '__main__':
    model = create_model(101)
    with open ('1.yaml', 'w+') as f:
        f.write(model.to_yaml_text())

