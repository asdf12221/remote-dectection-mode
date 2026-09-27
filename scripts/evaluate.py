#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a COCO detector on a caller-provided validation split and report
COCO AP plus MR/FAR (IoU 0.5, same-class greedy matching).

Example: python evaluate.py --config cfg.py --ckpt best.pth \
  --val-json annotations_val.json --image-root images/all
"""
import argparse, os, json, pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

CLS = ['HM','LQS','QHS','MS','A1_SU-35','A2_C-130','A3_C-17','A4_C-5','A5_F-16','A6_TU-160','A7_E-3','A8_B-52','A9_P-3C','A10_B-1B','A11_E-8','A12_TU-22','A13_F-15','A14_KC-135','A15_F-22','A16_FA-18','A17_TU-95','A18_KC-10','A19_SU-34','A20_SU-24','FSC']

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)

def greedy_eval(dets, gt_by, thr):
    F = defaultdict(list)
    for img, c, b, s in dets:
        if s >= thr: F[img].append((c, b, s))
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for img_id, gts in gt_by.items():
        pr = F.get(img_id, [])
        matched = [False]*len(gts)
        for pc, pb, ps in sorted(pr, key=lambda x: -x[2]):
            bi, bj = 0, -1
            for j, g in enumerate(gts):
                if matched[j] or g['category_id'] != pc: continue
                gb = g['bbox']; o = iou(pb, [gb[0], gb[1], gb[0]+gb[2], gb[1]+gb[3]])
                if o > bi: bi, bj = o, j
            if bi >= 0.5 and bj >= 0: tp[pc] += 1; matched[bj] = True
            else: fp[pc] += 1
        for j, g in enumerate(gts):
            if not matched[j]: fn[g['category_id']] += 1
    return tp, fp, fn

def mr_far(tp, fp, fn, cl):
    tt = sum(tp.get(c, 0) for c in cl); tfp = sum(fp.get(c, 0) for c in cl); tfn = sum(fn.get(c, 0) for c in cl)
    tgt = tt + tfn
    return (tfn/tgt if tgt else 0, tfp/(tt+tfp) if tt+tfp else 0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--val-json', required=True, help='COCO validation annotations')
    ap.add_argument('--image-root', required=True, help='Directory containing validation images')
    ap.add_argument('--output', default='eval_results.json')
    ap.add_argument('--save-detections', default=None, help='Optional pickle output')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    from mmengine.config import Config
    from mmdet.apis import init_detector, inference_detector
    import cv2
    model = init_detector(Config.fromfile(a.config), a.ckpt, device='cuda:0'); model.eval()
    val = json.load(open(a.val_json, encoding='utf-8'))
    gt_by = defaultdict(list)
    for an in val['annotations']:
        if an['category_id'] == 255: continue
        gt_by[an['image_id']].append(an)
    dets = []
    for im in val['images']:
        r = inference_detector(model, os.path.join(a.image_root, im['file_name']))
        pi = r.pred_instances
        for b, l, s in zip(pi.bboxes, pi.labels, pi.scores):
            x1, y1, x2, y2 = [float(v) for v in b]
            dets.append((im['id'], int(l), [x1, y1, x2, y2], float(s)))
    cats = [{'id': i, 'name': CLS[i]} for i in range(25)]
    report = {'n_imgs': len(val['images']), 'n_gt': sum(len(v) for v in gt_by.values())}
    if a.save_detections:
        with open(a.save_detections, 'wb') as handle:
            pickle.dump(dets, handle)
    # ---- COCO AP (已验证实现,与 pkl2ap 一致) ----
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import pickle as _pk
        _pk.dump(dets, open('/tmp/_eval_dets_tmp.pkl','wb'))
        _GT={'images':val['images'],'annotations':[a for a in val['annotations'] if a['category_id']!=255],'categories':cats}
        json.dump(_GT, open('/tmp/_val25.json','w'))
        coco = COCO('/tmp/_val25.json')
        wh = {im['id']:(im['width'],im['height']) for im in val['images']}
        _res=[]
        for img,c,box,sc in dets:
            w,h=wh[img]
            x1,y1,x2,y2=max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3])
            if x2<=x1 or y2<=y1: continue
            _res.append({'image_id':img,'category_id':int(c),'bbox':[x1,y1,x2-x1,y2-y1],'score':float(sc)})
        json.dump(_res, open('/tmp/_res_tmp.json','w'))
        dt = coco.loadRes('/tmp/_res_tmp.json')
        ev = COCOeval(coco, dt, 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
        pr = ev.eval['precision']
        report['mAP'] = float(ev.stats[0]); report['AP50'] = float(ev.stats[1]); report['AP75'] = float(ev.stats[2])
        report['APs'] = float(ev.stats[3]); report['APm'] = float(ev.stats[4]); report['APl'] = float(ev.stats[5])
        report['per_class_AP'] = {}
        for k in range(25):
            m = pr[:, :, k, 0, 2]
            report['per_class_AP'][CLS[k]] = float(m[m > -1].mean()) if (m > -1).any() else None
    except Exception:
        import traceback; traceback.print_exc()
    for tau in [0.3, 0.5, 0.7]:
        tp, fp, fn = greedy_eval(dets, gt_by, tau)
        o = mr_far(tp, fp, fn, range(25)); sh = mr_far(tp, fp, fn, range(4))
        ac = mr_far(tp, fp, fn, range(4, 24)); fs = mr_far(tp, fp, fn, [24])
        report[f'tau{tau}'] = {'MR': o[0], 'FAR': o[1], 'ship': sh, 'aircraft': ac, 'FSC': fs}
    with open(a.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print('提示: 完整 COCO mAP/AP50/75/s/m/l 与逐类 AP 请参照交付文档《评价指标》使用 pycocotools 计算(本脚本输出 MR/FAR 口径)')
