import torch.utils.data as data
import cv2
import torch
import numpy as np
import math
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard
from . import data_augment
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from polar import polar_encode

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.num_classes = None
        self.max_objs = 100 # 
        self.image_distort =  data_augment.PhotometricDistort()

    def load_img_ids(self):
        """
        Definition: generate self.img_ids
        Usage: index the image properties (e.g. image name) for training, testing and evaluation
        Format: self.img_ids = [list]
        Return: self.img_ids
        """
        return None

    def load_image(self, index):
        """
        Definition: read images online
        Input: index, the index of the image in self.img_ids
        Return: image with H x W x 3 format
        """
        return None

    def load_annoFolder(self, img_id):
        """
        Return: the path of annotation
        Note: You may not need this function
        """
        return None

    def load_annotation(self, index):
        """
        Return: dictionary of {'pts': float np array of [bl, tl, tr, br], 
                                'cat': int np array of class_index}
        Explaination:
                bl: bottom left point of the bounding box, format [x, y]
                tl: top left point of the bounding box, format [x, y]
                tr: top right point of the bounding box, format [x, y]
                br: bottom right point of the bounding box, format [x, y]
                class_index: the category index in self.category
                    example: self.category = ['ship]
                             class_index of ship = 0
        """
        return None

    def dec_evaluation(self, result_path):
        return None

    def data_transform(self, image, annotation):
        # only do random_flip augmentation to original images
        crop_size = None
        crop_center = None
        crop_size, crop_center = random_crop_info(h=image.shape[0], w=image.shape[1])
        image, gt_pts, crop_center = random_flip(image, annotation['pts'], crop_center)
        if crop_center is None:
            crop_center = np.asarray([float(image.shape[1])/2, float(image.shape[0])/2], dtype=np.float32)
        if crop_size is None:
            crop_size = [max(image.shape[1], image.shape[0]), max(image.shape[1], image.shape[0])]  # init
        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=True)
        image = cv2.warpAffine(src=image, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        if annotation['pts'].shape[0]:
            annotation['pts'] = np.concatenate([annotation['pts'], np.ones((annotation['pts'].shape[0], annotation['pts'].shape[1], 1))], axis=2)
            annotation['pts'] = np.matmul(annotation['pts'], np.transpose(M))
            annotation['pts'] = np.asarray(annotation['pts'], np.float32)

        out_annotations = {}
        size_thresh = 3
        out_rects = []
        out_cat = []
        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):
            if (pt_old<0).any() or (pt_old[:,0]>self.input_w-1).any() or (pt_old[:,1]>self.input_h-1).any():
                pt_new = pt_old.copy()
                pt_new[:,0] = np.minimum(np.maximum(pt_new[:,0], 0.), self.input_w - 1)
                pt_new[:,1] = np.minimum(np.maximum(pt_new[:,1], 0.), self.input_h - 1)
                iou = ex_box_jaccard(pt_old.copy(), pt_new.copy())
                if iou>0.6:
                    rect = cv2.minAreaRect(pt_new/self.down_ratio)
                    if rect[1][0]>size_thresh and rect[1][1]>size_thresh:
                        out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                        out_cat.append(cat)
            else:
                rect = cv2.minAreaRect(pt_old/self.down_ratio)
                if rect[1][0]<size_thresh and rect[1][1]<size_thresh:
                    continue
                out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                out_cat.append(cat)
        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        return image, out_annotations

    def __len__(self):
        return len(self.img_ids)

    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1


    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def reorder_pts(self, tt, rr, bb, ll):
        # 经测试，ssdd数据集中大概有40+个hbb目标
        # print("wooooof, a hbb!")
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmax(pts[:,0])
        r_ind = np.argmin(pts[:,0])
        t_ind = np.argmax(pts[:,1])
        b_ind = np.argmin(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new

    def arrange_order(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        # 如果有零，说明向量垂直，则直接返回
        if (pts == 0).sum() > 0:
          return tt, rr, bb, ll
        for v in [tt, rr, bb, ll]:
          if v[0] > 0 and v[1] > 0:
            tt_new = v
          if v[0] < 0 and v[1] > 0:
            rr_new = v
          if v[0] < 0 and v[1] < 0:
            bb_new = v
          if v[0] > 0 and v[1] < 0:
            ll_new = v

        return tt_new, rr_new, bb_new, ll_new


    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = self.image_distort(np.asarray(image, np.float32))
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)

        #####  注意这里加入了一个新的参数 #####
        polar_points_num = 8
        #####################################

        wh = np.zeros((self.max_objs, polar_points_num), dtype=np.float32)

        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)

        # print('!!!!!!!!!', num_objs)
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            # print(theta)
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            # generate wh ground_truth
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

            # 将四点标注法转为极坐标距离标注
            polar_infos = {}
            polar_infos['pts_4'] = pts_4
            polar_infos['ct'] = ct
            polar_infos['theta'] = theta
            polar_pts = np.asarray(polar_encode(polar_infos, polar_points_num), dtype=np.float32)

            # rotational channel
            wh[k, :] = polar_pts

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               }

        return ret

    def __getitem__(self, index):
        image = self.load_image(index)
        image_h, image_w, c = image.shape
        if self.phase == 'test':
            img_id = self.img_ids[index]
            image = self.processing_test(image, self.input_h, self.input_w)
            return {'image': image,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}

        elif self.phase == 'train':
            annotation = self.load_annotation(index)
            image, annotation = self.data_transform(image, annotation)
            data_dict = self.generate_ground_truth(image, annotation)
            return data_dict


