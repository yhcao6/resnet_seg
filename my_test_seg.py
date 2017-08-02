import os
import sys
import numpy as np
import cv2
import scipy.io as sio
import resize_uniform
from run_parrots import * 


def fast_hist(a, b, n):
    k = (a >=0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def load_color_LUT_21(fn):
    contents = sio.loadmat(fn)
    return contents.values()[0]

def get_img_size(filename):
    dim_list = list(cv2.imread(filename).shape)[:2]
    if not len(dim_list) == 2:
        print('Could not determine size of image %s' % filename)
        sys.exit(1)
    return dim_list

def check_img_list(root_dir, list_filename, laze_check=0):
    # check number of images
    with open(list_filename, 'r') as infile:
        img_list = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]
    print("checking img list. Image list size: %d" % len(img_list))

    # check if file exists
    if not os.path.isfile(root_dir + img_list[0]):
        print("Image %s not found" % (root_dir + img_list[0]))
        sys.exit(1)

    # check if all images' size are consistent
    base_size = get_img_size(root_dir + img_list[0])
    if not laze_check:
        for i in range(len(img_list)):
          img_fn = root_dir + img_list[i]
          print('checking image: %s' % img_list[i])
          if not os.path.isfile(img_fn):
              print('Image %s not found' % img_fn)
              sys.exit(1)
          img_size = get_image_size(img_fn)
          if not (base_size[0] == img_size[0] and base_size[1] == img_size[1]):
              print('Image size not consistent, images: %s vs. %s' % (img_fn, root_dir + img_list[0]))
              sys.exit(1)
    return base_size, img_list  # return image size and fn list

def test_eval_seg(test_root, gt_root, test_list, uniform_size, mean, batch_size):

    # pred dir
    if not os.path.isdir('predict/seg'):
        os.makedirs('predict/seg')
    if not os.path.isdir('predict/vlz'):
        os.makedirs('predict/vlz')

    # preparing data
    LUT = load_color_LUT_21('VOC_color_LUT_21.mat')  # load color map

    tmp = np.zeros((256, 3))
    tmp[0:21, ...] = LUT
    tmp[255] = [0, 1, 1]
    LUT = tmp

    with open(test_list, 'r') as infile:
        img_list = [line.strip() for line in infile.readlines() if len(line.strip()) > 0]

    base_size = [uniform_size, uniform_size]
    mean_map = np.tile(mean, [base_size[0], base_size[1], 1])
    
    hist = np.zeros((21, 21))

    # set model
    model_file = 'work_dir/model.yaml'
    session_file = 'val_session.yaml'
    param_file = 'work_dir/snapshots/iter.00020000.parrots'
  
    model, session = model_and_session(model_file, session_file)
    session.setup()

    flow = session.flow('val')
    flow.load_param(param_file)


    for i in range(0, len(img_list), batch_size):
    # for i in range(0, 10, batch_size):
        # every 100 iter print once
        if i%(batch_size*100) == 0:
            print('Processing: %d/%d' % (i, len(img_list)))
        
        true_batch_size = min(batch_size, len(img_list) - i)

        batch_data = np.zeros((batch_size, 3, base_size[0], base_size[1]), dtype=np.float)
        for k in range(true_batch_size):
            img = cv2.imread(test_root + img_list[i+k])
            batch_data[k, ...] = (resize_uniform.resize_pad_to_fit(img, base_size) - mean_map).transpose((2, 0, 1))

        inputs = {}
        inputs['data'] = np.array(batch_data).T
        flow.set_input('data', inputs['data'])


        flow.forward()


        pred = flow.data('fc1').value().T

 
        for k in range(0, true_batch_size):

            origin_img = cv2.imread(test_root + img_list[i+k])
            origin_shape = origin_img.shape
            gt_img = cv2.imread(gt_root + img_list[i+k].split('.')[0] + '.png')[:, :, 0]
            cls_map = np.array(pred[k].transpose(1, 2, 0).argmax(axis = 2), dtype=np.uint8)

            out_map = np.uint8(LUT[cls_map] * 255)
            cls_map_origin = resize_uniform.resize_crop_to_fit(cls_map, origin_shape[:2], interp=cv2.INTER_NEAREST)
            out_map_origin = resize_uniform.resize_crop_to_fit(out_map, origin_shape[:2], interp=cv2.INTER_NEAREST)


            gt_out_map = np.uint8(LUT[gt_img] * 255)

            hist += fast_hist(gt_img.flatten(), cls_map_origin.flatten(), 21)
            # cv2.imshow('original', origin_img)
            # both = np.hstack((gt_out_map, out_map_origin))
            # cv2.imshow('pred', both)
            # cv2.waitKey(0)           
   
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', 'overall accuracy', acc
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', 'mean accuracy', np.nanmean(acc)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', 'per class IU:\n', iu
    show_iu = ["{:.2f}".format(i*100) for i in iu]
    print '>>>', show_iu
    print '>>>', 'mean IU', np.nanmean(iu)
        
def main_eval():
    test_eval_seg('/home/yhcao6/VOC_arg/JPEGImages/', '/home/yhcao6/VOC_arg/SegmentationClass_label/', '/home/yhcao6/VOC_arg/Lists/Img/val.txt', 321, np.array([104.00699, 116.66877, 122.67892]), 1)

main_eval()


