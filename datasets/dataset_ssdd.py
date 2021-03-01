from .base import BaseDataset
import os
import cv2
import numpy as np
import sys
from .hrsc_evaluation_task1 import voc_eval

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

"""
数据集的文件目录结构如下：
BBA_CenterNet
    -data
        -ssdd
            -annotations   (xml标注文件, 为whxya版本的标注文件)
            -images        (所有图像在这里)
            -imageset  (存放有文件名的txt文件)
"""


class SSDD(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(SSDD, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['ship']
        self.num_classes = len(self.category)
        self.cat_ids = {cat: i for i, cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')  # 所有图像的所在地
        self.label_path = os.path.join(data_dir, 'annotations')  # 使用json标注的话需要用pycocotools库,
        # 而为了与r-centernet保持一致，且和HRSC数据集保持一致的话eval代码比较清晰
        # 的原因，这个文件夹用于存放xywha标注的xml文件

    def load_img_ids(self):
        # shit，改代码结构太麻烦了，这点需求，直接在测试inshore/offshore
        # 的时候来这里改代码就完了
        image_set_index_file = os.path.join(self.data_dir, 'imageset', self.phase + '.txt') # 
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip().strip('.xml') for line in lines]  # 因为txt生成的每行结尾有xml后缀，这里删除掉
        return image_lists

    def load_image(self, index):
        """加载一张图像, simple and done"""
        img_id = self.img_ids[index]
        # print(img_id)
        imgFile = os.path.join(self.image_path, img_id + '.jpg')  # 这里是jpg文件而非bmp
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        """annofolder内存放的是xywha格式的xml标注文件"""
        return os.path.join(self.label_path, img_id + '.xml')

    def load_annotation(self, index):
        # 加载BBA标注
        image = self.load_image(index)
        h, w, c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        target = ET.parse(self.load_annoFolder(self.img_ids[index])).getroot()
        # print(target)
        for obj in target.iter('object'):
            # print(obj.text)
            # print(obj.text)
            difficult = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            # bbox的指标目前先用不到，之后如果用到再说
            # box_xmin = int(obj.find('box_xmin').text)  # bbox
            # box_ymin = int(obj.find('box_ymin').text)
            # box_xmax = int(obj.find('box_xmax').text)
            # box_ymax = int(obj.find('box_ymax').text)
            mbox_cx = float(bbox.find('x').text)  # rbox
            mbox_cy = float(bbox.find('y').text)
            mbox_w = float(bbox.find('w').text)
            mbox_h = float(bbox.find('h').text)
            mbox_ang = float(bbox.find('a').text) * 180 / np.pi
            rect = ((mbox_cx, mbox_cy), (mbox_w, mbox_h), mbox_ang)
            pts_4 = cv2.boxPoints(rect)  # 4 x 2
            bl = pts_4[0, :]
            tl = pts_4[1, :]
            tr = pts_4[2, :]
            br = pts_4[3, :]
            valid_pts.append([bl, tl, tr, br])
            valid_cat.append(self.cat_ids['ship'])
            valid_dif.append(difficult)
        annotation = {'pts': np.asarray(valid_pts, np.float32), 'cat': np.asarray(valid_cat, np.int32),
                      'dif': np.asarray(valid_dif, np.int32)}

        # img = self.load_image(index)
        # for rect in annotation['rect']:
        #     pts_4 = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))  # 4 x 2
        #     bl = pts_4[0,:]
        #     tl = pts_4[1,:]
        #     tr = pts_4[2,:]
        #     br = pts_4[3,:]
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
        # cv2.imshow('img', np.uint8(img))
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return annotation

    def dec_evaluation(self, result_path):
        detpath = os.path.join(result_path, 'Task1_{}.txt')
        annopath = os.path.join(self.label_path,
                                '{}.xml')
        imagesetfile = os.path.join(self.data_dir, 'imageset/test.txt')
        classaps = []
        map = 0
        for classname in self.category:
            if classname == 'background':
                continue
            print('classname:', classname)
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            import pickle
            eval_results = {
                'precision': prec,
                'recall': rec,
                'ap':ap
            }
            with open('/content/drive/My Drive/BBA-CenterNet/pr/' + 'result.pkl', 'wb') as f:
                pickle.dump(eval_results, f)
            map = map + ap
            # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            print('{}:{} '.format(classname, ap * 100))
            classaps.append(ap)
            # umcomment to show p-r curve of each category
            # plt.figure(figsize=(8,4))
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.plot(rec, prec)
        # plt.show()
        map = map / len(self.category)
        print('map:', map * 100)
        # classaps = 100 * np.array(classaps)
        # print('classaps: ', classaps)
        return map
