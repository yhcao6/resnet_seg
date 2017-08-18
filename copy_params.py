import caffe
import h5py
import numpy as np

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net('./voc12/config/deelab_largeFOV/train_train_aug.prototxt', './voc12/model/deelab_largeFOV/init.caffemodel', caffe.TRAIN)

# with open('1.txt', 'w+') as f:
#     for layer_name, params in net.params.iteritems():
#          for i in range(len(params)):
#             f.write('{}_{}\t{}'.format(layer_name, i, params[i].data.shape) + '\n')

with h5py.File('params.h5', 'w') as f:
    for layer_name, param in net.params.iteritems():
        i = 0
        while layer_name[i] < '1' or layer_name[i] > '9':
            i += 1
        l_n = layer_name[i]

        # if 'fc' in layer_name or 'res05' in layer_name or 'res075' in layer_name:
        if 'res05' in layer_name or 'res075' in layer_name:
            continue

        elif 'fc' in layer_name:
            n = layer_name[-1]
            f.create_dataset('fc1_c{}.w@value'.format(n), data=param[0].data)
            f.create_dataset('fc1_c{}.b@value'.format(n), data=param[1].data)

        # part 2 and 5
        elif l_n == '2' or l_n == '5':
            if 'res' in layer_name.split('_')[0]:
                # res.{l_n}{a_b}{n}.conv{k}
                a_b = layer_name.split('_')[0][-1]
                if a_b == 'a':
                    a_b = 'a'
                else:
                    a_b = 'b'

                n = layer_name.split('_')[0][-1]
                if n == 'c':
                    n = 2
                if n == 'b':
                    n = 1

                k = layer_name.split('_')[1][-1]
                if k == '1':
                    k = 'shortcut'
                if k == 'a':
                    k = 1
                if k == 'b':
                    k = 2
                if k == 'c':
                    k = 3

                if k == 'shortcut':
                    f.create_dataset('res{}a.shortcut.w@value'.format(l_n), data=param[0].data)
                else:
                    if a_b == 'b':
                        f.create_dataset('res{}b{}.conv{}.w@value'.format(l_n, n, k), data=param[0].data)
                    else:
                        f.create_dataset('res{}a.conv{}.w@value'.format(l_n, k), data=param[0].data)
            elif 'scale' in layer_name:
                # res.{l_n}{a_b}{n}.bn{k}
                a_b = layer_name.split('_')[0][6]
                if a_b == 'a':
                    a_b = 'a'
                else:
                    a_b = 'b'
                n = layer_name.split('_')[0][-1]
                if n == 'c':
                    n = 2
                if n == 'b':
                    n = 1
                k = layer_name.split('_')[1][-1]
                if k == '1':
                    k = 'shortcut'
                if k == 'a':
                    k = 1
                if k == 'b':
                    k = 2
                if k == 'c':
                    k = 3
                if k == 'shortcut':
                    f.create_dataset('res{}a.shortcut_bn.s@value'.format(l_n), data=param[0].data)
                    f.create_dataset('res{}a.shortcut_bn.b@value'.format(l_n), data=param[1].data)
                else:
                    if a_b == 'b':
                        f.create_dataset('res{}b{}.bn{}.s@value'.format(l_n, n, k), data=param[0].data)
                        f.create_dataset('res{}b{}.bn{}.b@value'.format(l_n, n, k), data=param[1].data)
                    else:
                        f.create_dataset('res{}a.bn{}.s@value'.format(l_n, k), data=param[0].data)
                        f.create_dataset('res{}a.bn{}.b@value'.format(l_n, k), data=param[1].data)
            elif 'bn' in layer_name:
                # res.{l_n}{a_b}{n}.bn{k}
                a_b = layer_name.split('_')[0][3]
                if a_b == 'a':
                    a_b = 'a'
                else:
                    a_b = 'b'
                n = layer_name.split('_')[0][-1]
                if n == 'c':
                    n = 2
                if n == 'b':
                    n = 1
                k = layer_name.split('_')[1][-1]
                if k == '1':
                    k = 'shortcut'
                if k == 'a':
                    k = 1
                if k == 'b':
                    k = 2
                if k == 'c':
                    k = 3
                scale = param[2].data[0]
                h = np.append(param[0].data, param[1].data) / float(scale)
                if k == 'shortcut':
                    f.create_dataset('res{}a.shortcut_bn.h@value'.format(l_n), data=h)
                else:
                    if a_b == 'b':
                        f.create_dataset('res{}b{}.bn{}.h@value'.format(l_n, n, k), data=h)
                    else:
                        f.create_dataset('res{}a.bn{}.h@value'.format(l_n, k), data=h)
        # part3 and part4
        elif l_n == '3' or l_n == '4':
            if 'res' in layer_name:
                # res.{l_n}{a_b}{n}.conv{k}
                a_b = layer_name.split('_')[0][4]
                n = layer_name.split('_')[0][5:]
                k = layer_name.split('_')[1][-1]
                if k == '1':
                    k = 'shortcut'
                if k == 'a':
                    k = 1
                if k == 'b':
                    k = 2
                if k == 'c':
                    k = 3
                if k == 'shortcut':
                    f.create_dataset('res{}a.shortcut.w@value'.format(l_n), data=param[0].data)
                else:
                    if a_b == 'b':
                        f.create_dataset('res{}b{}.conv{}.w@value'.format(l_n, n, k), data=param[0].data)
                    else:
                        f.create_dataset('res{}a.conv{}.w@value'.format(l_n, k), data=param[0].data)
            elif 'scale' in layer_name:
                # res.{l_n}{a_b}{n}.bn{k}
                a_b = layer_name.split('_')[0][6]
                n = layer_name.split('_')[0][7:]
                k = layer_name.split('_')[1][-1]
                if k == '1':
                    k = 'shortcut'
                if k == 'a':
                    k = 1
                if k == 'b':
                    k = 2
                if k == 'c':
                    k = 3
                if k == 'shortcut':
                    f.create_dataset('res{}a.shortcut_bn.s@value'.format(l_n), data=param[0].data)
                    f.create_dataset('res{}a.shortcut_bn.b@value'.format(l_n), data=param[1].data)
                else:
                    if a_b == 'b':
                        f.create_dataset('res{}b{}.bn{}.s@value'.format(l_n, n, k), data=param[0].data)
                        f.create_dataset('res{}b{}.bn{}.b@value'.format(l_n, n, k), data=param[1].data)
                    else:
                        f.create_dataset('res{}a.bn{}.s@value'.format(l_n, k), data=param[0].data)
                        f.create_dataset('res{}a.bn{}.b@value'.format(l_n, k), data=param[1].data)
            elif 'bn' in layer_name:
                # res.{l_n}{a_b}{n}.bn{k}
                a_b = layer_name.split('_')[0][3]
                n = layer_name.split('_')[0][4:]
                k = layer_name.split('_')[1][-1]
                if k == '1':
                    k = 'shortcut'
                if k == 'a':
                    k = 1
                if k == 'b':
                    k = 2
                if k == 'c':
                    k = 3
                scale = param[2].data[0]
                h = np.append(param[0].data, param[1].data) / float(scale)
                if k == 'shortcut':
                    f.create_dataset('res{}a.shortcut_bn.h@value'.format(l_n), data=h)
                else:
                    if a_b == 'b':
                        f.create_dataset('res{}b{}.bn{}.h@value'.format(l_n, n, k), data=h)
                    else:
                        f.create_dataset('res{}a.bn{}.h@value'.format(l_n, k), data=h)
        else:
            if layer_name == 'conv1':
                f.create_dataset('conv1.w@value', data=param[0].data)
            elif layer_name == 'bn_conv1':
                scale = param[2].data[0]
                h = np.append(param[0].data, param[1].data)/float(scale)
                f.create_dataset('bn1.h@value', data=h)
            elif layer_name == 'scale_conv1':
                f.create_dataset('bn1.s@value', data=param[0].data)
                f.create_dataset('bn1.b@value', data=param[1].data)
            else:
                print layer_name
