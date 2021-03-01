import os
import torch
import numpy as np
from datasets.DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast, py_cpu_nms_poly
import time

def decode_prediction(predictions, dsets, args, img_id, down_ratio):
    predictions = predictions[0, :, :]
    ori_image = dsets.load_image(dsets.img_ids.index(img_id))
    h, w, c = ori_image.shape

    pts0 = {cat: [] for cat in dsets.category}
    scores0 = {cat: [] for cat in dsets.category}
    bd_pts0 = {cat: [] for cat in dsets.category}

    for pred in predictions:
        cen_pt = np.asarray([pred[0], pred[1]], np.float32)
        tt = np.asarray([pred[-8], pred[-7]], np.float32)
        rr = np.asarray([pred[-6], pred[-5]], np.float32)
        bb = np.asarray([pred[-4], pred[-3]], np.float32)
        ll = np.asarray([pred[-2], pred[-1]], np.float32)
        tl = tt + ll + cen_pt
        bl = bb + ll + cen_pt
        tr = tt + rr + cen_pt
        br = bb + rr + cen_pt
        score = pred[2]
        clse = pred[3]

        ########################################
        N = 8
        bd_points_list_x = []
        bd_points_list_y = []
        for i in range(2*N):
            x_i = (pred[4+i] + cen_pt[0]) * down_ratio / args.input_w * w
            y_i = (pred[4+2*N+i] + cen_pt[1]) * down_ratio / args.input_h * h
            bd_points_list_x.append(x_i) # 像其他点一样，加上中心点坐标
            bd_points_list_y.append(y_i)
        bd_x = np.asarray(bd_points_list_x)
        bd_y = np.asarray(bd_points_list_y)
        bd_pts = np.asarray([bd_x, bd_y])
        ########################################

        pts = np.asarray([tr, br, bl, tl], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / args.input_w * w
        pts[:, 1] = pts[:, 1] * down_ratio / args.input_h * h

        pts0[dsets.category[int(clse)]].append(pts)
        scores0[dsets.category[int(clse)]].append(score)
        bd_pts0[dsets.category[int(clse)]].append(bd_pts) # (num_targets, 2, 2N)

    return pts0, scores0, bd_pts0


def non_maximum_suppression(pts, scores, bd_pts):
    nms_item = np.concatenate([pts[:, 0:1, 0],
                               pts[:, 0:1, 1],
                               pts[:, 1:2, 0],
                               pts[:, 1:2, 1],
                               pts[:, 2:3, 0],
                               pts[:, 2:3, 1],
                               pts[:, 3:4, 0],
                               pts[:, 3:4, 1],
                               scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)
    bd_pts = np.asarray(bd_pts, np.float64)
    
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.1)

    return nms_item[keep_index], bd_pts[keep_index] # 变成new_nms_item了


def write_results(args,
                  model,
                  dsets,
                  down_ratio,
                  device,
                  decoder,
                  result_path,
                  print_ps=False):
    results = {cat: {img_id: [] for img_id in dsets.img_ids} for cat in dsets.category}
    start_time = time.time()
    for index in range(len(dsets)):
        data_dict = dsets.__getitem__(index)
        image = data_dict['image'].to(device)
        img_id = data_dict['img_id']
        image_w = data_dict['image_w']
        image_h = data_dict['image_h']

        with torch.no_grad():
            pr_decs = model(image)


        decoded_pts = []
        decoded_scores = []
        torch.cuda.synchronize(device)
        predictions = decoder.ctdet_decode(pr_decs)
        pts0, scores0 = decode_prediction(predictions, dsets, args, img_id, down_ratio)
        decoded_pts.append(pts0)
        decoded_scores.append(scores0)

        # nms
        for cat in dsets.category:
            if cat == 'background':
                continue
            pts_cat = []
            scores_cat = []
            for pts0, scores0 in zip(decoded_pts, decoded_scores):
                pts_cat.extend(pts0[cat])
                scores_cat.extend(scores0[cat])
            pts_cat = np.asarray(pts_cat, np.float32)
            scores_cat = np.asarray(scores_cat, np.float32)
            if pts_cat.shape[0]:
                nms_results = non_maximum_suppression(pts_cat, scores_cat)
                results[cat][img_id].extend(nms_results)
        if print_ps:
            print('testing {}/{} data {}'.format(index+1, len(dsets), img_id))
    end_time = time.time()
    duration = end_time - start_time
    print("Average test time per image is: {}s.".format(duration / len(dsets)))
    for cat in dsets.category:
        if cat == 'background':
            continue
        with open(os.path.join(result_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
            for img_id in results[cat]:
                for pt in results[cat][img_id]:
                    f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))
