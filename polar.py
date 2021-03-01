# Created by Yishan 2020/12/14
# Oriented Detection with Polar Vectors 

import sys
sys.path.append("./datasets/DOTA_devkit/")
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from MBB import MinimumBoundingBox, BoundingBox
import polyiou

###########--------Encode部分--------###########

def arrange_order(tt, rr, bb, ll):
    """
    点顺序定义：
    1. 水平框
    这种情况下，t表示y坐标最大的点，b表示y坐标最小的点，l表示x坐标最大的点，r表示x坐标最小的点
    2. 旋转框
    这种情况下，t表示第一象限点，r表示第二象限点，b表示第三象限点，l表示第四象限点
    """
    pts = np.asarray([tt,rr,bb,ll],np.float32)
    # 如果有零，说明向量垂直
    if (pts == 0).sum() > 0:
        l_ind = np.argmax(pts[:,0])
        r_ind = np.argmin(pts[:,0])
        t_ind = np.argmax(pts[:,1])
        b_ind = np.argmin(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new, rr_new, bb_new, ll_new       
    else:
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

def calculate_point_angle(point):
    """
    返回一个直角坐标系下点的角度，范围是(-pi, pi]
    """
    x = point[0];
    y = point[1];
    z = math.sqrt(x**2 + y**2)
    # y == 0 时，x < 0, 角度为pi，因此只能取到正pi
    if (y >= 0):
        return np.arccos(x / z)
    else:
        return -np.arccos(x / z)


def between_range(pre_angle, nxt_angle, target_angle):
    """
    判断target_angle是否在(pre_angle, nxt_angle]范围内，要考虑(-pi, pi]
    """

    '''
    当a->b->c->d这条角度链可能存在几个情况:
    1. pre < nxt 只要pre和nxt连线不过x负半轴，就不会有问题；
       这种情况就直接判断角度范围即可
    2. pre > nxt:
       pre > 0, nxt < 0: 说明pre在12象限，nxt在34象限，判断这个范围即可，注意x负半轴角度是正pi;
    3. pre == nxt: 不可能发生，如果发生，应该报错

    '''

    if pre_angle < nxt_angle:
        if target_angle > pre_angle and target_angle <= nxt_angle:
            return True
        else:
            return False
    elif pre_angle > nxt_angle:
        assert pre_angle > 0 and nxt_angle < 0
        if (target_angle > pre_angle and target_angle <= np.pi) or (target_angle > - np.pi and target_angle <= nxt_angle):
            return True
        else:
            return False
    else:
        raise("OBB with zero width/height!")

def calculate_distance(t_ag, a1, a2, r1, r2):
    """
    给定极坐标下两点(r1, a1), (r2, a2)，计算角度t_ag的射线与这两点确定的直线的交点距极点的距离
    """
    return r1 * r2 * math.sin(a1 - a2) / (r2 * math.sin(t_ag - a2) + r1 * math.sin(a1 - t_ag))


def calculate_intersection_distance(target_angle, neighbor_pts_angle, corner_pts):
    """
    该函数计算target_angle角度对应的边界盒子的交点与原点（中心点）的距离。
    target_angle: 目标角度，弧度制。
    neighbor_pts_angle: ([a_ag, b_ag], [b_ag, c_ag], [c_ag, d_ag], [d_ag, a_ag])
    corner_pts: [a_pt, b_pt, c_pt, d_pt]

    return: target_distance 即为 目标交点与中心点的距离。

    """

    # 对每一个 [pre_a, nxt_a)的范围，如果判断交点属于这一范围，则计算射线与这条直线的交点
    for i, (pre_a, nxt_a) in enumerate(neighbor_pts_angle):
        if between_range(pre_a, nxt_a, target_angle):
            pt_1 = corner_pts[i]
            pt_2 = corner_pts[(i+1) % 4] # 这个不会有问题吧
            pt1_ag = calculate_point_angle(pt_1)
            pt2_ag = calculate_point_angle(pt_2)
            pt1_r  = math.sqrt(pt_1[0]**2 + pt_1[1]**2)
            pt2_r  = math.sqrt(pt_2[0]**2 + pt_2[1]**2)
            dist = calculate_distance(target_angle, pt1_ag, pt2_ag, pt1_r, pt2_r)
            break
    return dist


            
def polar_encode(points_info, n):
    '''
    该函数只处理一张图中一个目标的标注信息，将之从4点标注转为极坐标距离标注
    points_info: 标注角点标注信息的一个字典。
                 其中'pts_4'键对应角点位置信息；'ct'对应中心点位置信息；
                 可以在具体实现中观察其使用方法。
    n: 将0~180度分为几个部分，生成几个标注点，如每隔45度一点的话，n=4。
    return: polar_pts_n表示极坐标系下每隔180/n角度，矩形框边界点与原点（中心点）产生的距离，是一个n长list。
    '''

    pts_4 = points_info['pts_4']
    ct    = points_info['ct']
    theta = points_info['theta']

    bl = pts_4[0,:]
    tl = pts_4[1,:]
    tr = pts_4[2,:]
    br = pts_4[3,:]

    tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2 - ct
    rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2 - ct
    bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2 - ct
    ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2 - ct

    # 转换为了有序的四个象限坐标
    p1, p2, p3, p4 = arrange_order(tt, rr, bb, ll)
    
    # 得到四个角点
    a_pt, b_pt, c_pt, d_pt = p1 + p2, p2 + p3, p3 + p4, p4 + p1

    # 得到四个角点对应的角度，方便计算交点，注意这里角度范围[0, 2*pi)
    a_ag, b_ag, c_ag, d_ag = calculate_point_angle(a_pt), calculate_point_angle(b_pt), \
                             calculate_point_angle(c_pt), calculate_point_angle(d_pt)

    # 接下来给定一个角度，需要计算这个角度射线与边界上某一条边的交点
    # 获取n个标注点，这些点是对应角度射线与包围盒边界的交点

    # 范围是[0, pi)内取n个点
    delta_angle = np.pi / n
    neighbor_pts_angle = ([a_ag, b_ag], [b_ag, c_ag], [c_ag, d_ag], [d_ag, a_ag])
    corner_pts = [a_pt, b_pt, c_pt, d_pt]

    polar_pts_n = []
    for i in range(n):
        target_angle = delta_angle * i
        target_distance = calculate_intersection_distance(target_angle, neighbor_pts_angle, corner_pts)
        polar_pts_n.append(target_distance)

    return polar_pts_n


###########------Encode部分结束------###########


###########--------Decode部分--------###########

def calculate_boundary_points(corner_pts):
    pt1 = corner_pts.pop()
    pt2 = corner_pts.pop()
    pt3 = corner_pts.pop()
    pt4 = corner_pts.pop()
    
    # 采用角度法计算
    pt1_ag = calculate_point_angle(pt1)
    pt2_ag = calculate_point_angle(pt2)
    pt3_ag = calculate_point_angle(pt3)
    pt4_ag = calculate_point_angle(pt4)
    
    unordered_pts    = np.array([pt1,    pt2,    pt3,    pt4   ])
    unordered_angles = np.array([pt1_ag, pt2_ag, pt3_ag, pt4_ag])
    
    indexes = np.argsort(unordered_angles)
    pt1_new = unordered_pts[indexes[0]]
    pt2_new = unordered_pts[indexes[1]]
    pt3_new = unordered_pts[indexes[2]]
    pt4_new = unordered_pts[indexes[3]]
    
    tt = (np.asarray(pt1_new, np.float32) + np.asarray(pt2_new, np.float32)) / 2
    rr = (np.asarray(pt2_new, np.float32) + np.asarray(pt3_new, np.float32)) / 2
    bb = (np.asarray(pt3_new, np.float32) + np.asarray(pt4_new, np.float32)) / 2
    ll = (np.asarray(pt4_new, np.float32) + np.asarray(pt1_new, np.float32)) / 2
    
    return tt, rr, bb, ll
    
    

def polar_decode(wh, scores, clses, xs, ys, thres, n):
    """
    wh, scores, cls，xs, ys即为ctdet_decode中已经转换好的检测结果相关的信息
        其中wh为预测坐标信息，scores用来阈值判断，cls为类别信息，xsys为中心点信息，thres为阈值
    return: detections即为与ctdet_decode中的返回结果一样的东西，在ctdet_decode中可以这样调用该函数：

        return polar_decode(wh, scores, cls, xs, ys, thres)
    sizes:
        wh: (batch, self.K, 8)
        clses: (batch, self.K, 1)
        scores: (batch, self.K, 1)
        xs: (batch, self.K, 1)
        ys: (batch, self.K, 1)
        thres: (1,)
    return:
        detections: (batch, K, 4+(8+8+8+8)+8)
        4:   ctx, cty, score, cls.
        8*4: wh_x, wh_y, wh_sym_x, wh_sym_y.
        8:   corner points of the targets.
    """

    '''
    接下来：
    1. 找到筛选过后的wh
    2. 将距离信息转换为直角坐标信息，且补全整个矩形目标框的信息
    3. 根据补全后的信息，计算最小矩形包围盒
    4. 计算四个边中心点，生成tt_x - ll_y
    5. 把wh外的其它信息cat，筛，再与生成的几个向量cat起来。
    '''
    # 1. 找到筛选后的wh
    index = (scores>thres).squeeze(0).squeeze(1)
    wh = wh[:,index,:]
    # print(wh.size())
    
    # 2. 转为直角坐标并补全
    num_targets = wh.size()[1]
    angles = torch.linspace(0, np.pi, n+1)[:-1].unsqueeze(0).unsqueeze(0).repeat(1,num_targets,1).cuda()
    wh_x = torch.mul(wh, torch.cos(angles))
    wh_y = torch.mul(wh, torch.sin(angles))
    wh_x_symmetry = -1 * wh_x
    wh_y_symmetry = -1 * wh_y
    
    # 3. 计算最小包围盒
    
    target_loc_pts = torch.zeros(1, num_targets, 8).cuda() # ttx tty rrx rry bbx bby llx lly
    
    for i in range(num_targets):
        # 每个目标计算一个MMB(tt, bb, ll, rr来表示)
        target_pts = []
        for j in range(n):
            target_pts.append((wh_x[0, i, j].cpu(),          wh_y[0, i, j].cpu()))
            target_pts.append((wh_x_symmetry[0, i, j].cpu(), wh_y_symmetry[0, i, j].cpu()))
        # 根据边界点计算最小包围盒
        mbb = MinimumBoundingBox(target_pts)
        
        # 4. 计算四个边中心点，生成tt_x - ll_y，赋给检测结果
        
        # 获得角点
        corner_pts = mbb.corner_points
        # 根据无序角点计算各边中点（无序）
        tt, rr, bb, ll = calculate_boundary_points(corner_pts)
        # 把无序变有序
        tt, rr, bb, ll = arrange_order(tt, rr, bb, ll)
        # 将计算得到的各中点赋值给检测结果
        target_loc_pts[0, i, 0] = float(tt[0])
        target_loc_pts[0, i, 1] = float(tt[1])
        target_loc_pts[0, i, 2] = float(rr[0])
        target_loc_pts[0, i, 3] = float(rr[1])
        target_loc_pts[0, i, 4] = float(bb[0])
        target_loc_pts[0, i, 5] = float(bb[1])
        target_loc_pts[0, i, 6] = float(ll[0])
        target_loc_pts[0, i, 7] = float(ll[1])
    
    # print(target_loc_pts)
    # 5. 把wh外的其它信息cat，筛，再与生成的几个向量cat起来。
    detections = torch.cat([xs,                      # cen_x
                            ys,                      # cen_y
                            scores,
                            clses],
                            dim=2)
    detections = detections[:,index,:]

    # detections结构现在是这样的：
    # 序号0~3 ctx, cty, score, cls.
    # 序号[4~4+2N), 边界点x坐标
    # 序号[4+2N, 4+4N), 边界点y坐标
    # 序号[4+4N, -1], MMB边界点的x和y坐标
    detections = torch.cat([detections, wh_x, wh_x_symmetry, wh_y, wh_y_symmetry, target_loc_pts], dim=2) # 5 6 7 8
    
    
    return detections.data.cpu().numpy()

###########------Decode部分结束------###########


###########-------IoU损失函数--------###########

class IoUWeightedSmoothL1Loss(nn.Module):

    def __init__(self):
        super(IoUWeightedSmoothL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _polar_to_bboxes(self, polar_preds):
        """
        将polar_preds: gpu上size为[num_obj,n]的tensor转换为cpu上numpy array类型的[num_obj, 8]的bbox角点表示
        """

        # 1. 转为直角坐标
        num_targets = polar_preds.size()[0]
        n           = polar_preds.size()[1]

        angles = torch.linspace(0, np.pi, n+1)[:-1].unsqueeze(0).repeat(num_targets,1).cuda() # (num_targets, n)
        wh_x = torch.mul(polar_preds, torch.cos(angles))
        wh_y = torch.mul(polar_preds, torch.sin(angles))
        wh_x_symmetry = -1 * wh_x
        wh_y_symmetry = -1 * wh_y

        # 2. 直角坐标转MMB
        target_loc_pts = np.zeros([num_targets, 8]) # trx try brx bry blx bly tlx tly
    
        for i in range(num_targets):
            # 每个目标计算一个MMB
            target_pts = []
            for j in range(n):
                target_pts.append((wh_x[i, j].cpu().detach().numpy(),          wh_y[i, j].cpu().detach().numpy()))
                target_pts.append((wh_x_symmetry[i, j].cpu().detach().numpy(), wh_y_symmetry[i, j].cpu().detach().numpy()))
            # 根据边界点计算最小包围盒
            mbb = MinimumBoundingBox(target_pts)
            
            # 获得角点
            corner_pts = mbb.corner_points
            # 根据无序角点计算各边中点（无序）
            tt, rr, bb, ll = calculate_boundary_points(corner_pts)
            # 把无序变有序
            # tt, rr, bb, ll = arrange_order(tt, rr, bb, ll) # 这里不需要严格顺序
            # 重新生成有序角点
            tr = tt + rr
            br = bb + rr
            bl = bb + ll
            tl = tt + ll
            # 将计算得到有序角点赋值给结果
            target_loc_pts[i, 0] = float(tr[0])
            target_loc_pts[i, 1] = float(tr[1])
            target_loc_pts[i, 2] = float(br[0])
            target_loc_pts[i, 3] = float(br[1])
            target_loc_pts[i, 4] = float(bl[0])
            target_loc_pts[i, 5] = float(bl[1])
            target_loc_pts[i, 6] = float(tl[0])
            target_loc_pts[i, 7] = float(tl[1])

        return target_loc_pts

    def _calculate_ious_one_batch(self, pred_bboxes, target_bboxes):
        """
        根据pred_bboxes: (num_obj, 8)和target_bboxes: (num_obj, 8), 计算ious: (num_obj, )的list
        """

        num_obj = pred_bboxes.shape[0]
        ious_one_batch = []
        for i in range(num_obj):
            iou = polyiou.iou_poly(polyiou.VectorDouble(pred_bboxes[i]), polyiou.VectorDouble(target_bboxes[i]))
            ious_one_batch.append(iou)

        return ious_one_batch

    def _calculate_ious(self, output, mask, ind, target):
        """
        ind可以将网络直接输出的结果output中取出目标中心点处的n个通道值;
        mask可用于从已经“从图转为数组”的数据结构中，挑出那些真正是目标的数据;
        计算这批数据中所有gt中心点出真实目标与预测目标的iou

        return: ious_all 是一个有batch_size个元素的list，其中每个元素是一个[num_obj,]形状的list

        size:
        output: [b, n, w/4, h/4]
        mask: [b, max_obj]
        ind: [b, max_obj]
        target: [b, max_obj, n]
        """

        num_pts = output.size()[1]

        # 1. 特征图数据转为数组结构
        pred = self._tranpose_and_gather_feat(output, ind)  # [b, max_obj, n]

        ious_all = []

        for b in range(pred.size()[0]):

            # 2. 将有效的num_obj个目标信息从max_obj个信息中挑选出来(pred和target都需要)--[num_obj, n]
            cur_pred   = pred[b]     # [max_obj, n]
            cur_target = target[b]   # [max_obj, n]
            cur_mask   = mask[b]     # [max_obj,  ]
            num_obj    = cur_mask.sum()
            if num_obj:
                cur_mask     = cur_mask.unsqueeze(1).expand_as(cur_pred).type(torch.bool) # [max_obj, n]
                valid_pred   = cur_pred.masked_select(cur_mask)       # [num_obj * n]
                valid_target = cur_target.masked_select(cur_mask)     # [num_obj * n]
                valid_pred   = valid_pred.reshape(num_obj, num_pts)   # [num_obj, n]
                valid_target = valid_target.reshape(num_obj, num_pts) # [num_obj, n]

                # 3. 分obj将极坐标信息转为直角坐标；
                pred_bboxes  = self._polar_to_bboxes(valid_pred)   # [num_obj, 8]
                target_bboxes= self._polar_to_bboxes(valid_target) # [num_obj, 8]

                # 4. 计算当前batch所有obj的IoU
                ious = self._calculate_ious_one_batch(pred_bboxes, target_bboxes) # [num_obj, ]
                ious_all.append(ious)

        return ious_all


    def forward(self, output, mask, ind, target):
        """
        计算每个目标的SmoothL1Loss, 并根据对应的IoU进行加权求和，最后得到总的loss
        """
        
        '''
        实现思路：
        1. 利用前面的辅助函数计算每个batch中不同目标处的iou；
        2. 得到所有目标的回归loss
        3. iou对这些loss加权求和
        '''

        # ious_all 是一个list, len(ious_all) = batch_size，其中每个元素是一个[num_obj,]形状的numpy array
        # 1. 得到一个存有所有batch中所有目标的ious的列表
        ious_all_lists = self._calculate_ious(output, mask, ind, target)
        ious_all = []
        for li in ious_all_lists:
          ious_all.extend(li)

        # print(ious_all)


        # 2. 得到所有batch中所有目标中心处的回归结果与GT (num_targets, n)
        n = target.size()[2]
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([b, 500, n])

        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool() # (b, 500, n)
            pred = pred.masked_select(mask).reshape(-1,n)
            target= target.masked_select(mask).reshape(-1,n) 

            # 3. 计算每个目标处的smooth_l1_loss
            losses = []
            for i in range(pred.size()[0]):
              # pred[i] = pred[i] / torch.max(pred[i])
              # target[i] = target[i] / torch.max(target[i])
              # print(pred[i] / torch.max(pred[i]))
              # print(target[i] / torch.max(target[i]))
              losses.append(F.smooth_l1_loss(pred[i], target[i], reduction='mean'))
            # print(losses)
            loss_reg = 0
            loss_iou = 0
            eps = np.finfo(np.float32).eps

            # 4. 加权求和
            for i in range(len(losses)):
              loss_reg += losses[i]
              alpha = - np.log(abs(ious_all[i]) + eps)
              # print("alpha:", alpha)
              loss_iou += alpha * losses[i] / (abs(losses[i].item()) + eps)
              # magnitude = np.clip(-math.log(abs(ious_all[i])), 0, 100)
              # print(magnitude)
              # print( abs(losses[i].item()) )
              # print(magnitude * losses[i] / (abs(losses[i].item())) + eps)
              # loss_iou += (-math.log(abs(ious_all[i]))) * losses[i] / (abs(losses[i].item()))

            loss_reg /= len(losses)
            loss_iou /= len(losses)

            # print()
            # print('*' * 20)
            # print(loss)
            # print(loss2)
            # print('*' * 20)
            
            return loss_reg, loss_iou
        else:
            return 0.

###########-------IoU损失函数结束--------###########
