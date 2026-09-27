#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单图推理(≤800 尺度): 直接送检, τ=0.70, 输出画框图 + JSON。
用法: python infer_single.py --image x.jpg --config cfg.py --ckpt best.pth [--tau 0.70] [--gpu 0]"""
import argparse, os, json, cv2
import warnings; warnings.filterwarnings('ignore')
CLS = ['HM','LQS','QHS','MS','A1_SU-35','A2_C-130','A3_C-17','A4_C-5','A5_F-16','A6_TU-160','A7_E-3','A8_B-52','A9_P-3C','A10_B-1B','A11_E-8','A12_TU-22','A13_F-15','A14_KC-135','A15_F-22','A16_FA-18','A17_TU-95','A18_KC-10','A19_SU-34','A20_SU-24','FSC']
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True); ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='./det_result.png'); ap.add_argument('--tau', type=float, default=0.70); ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    from mmengine.config import Config
    from mmdet.apis import init_detector, inference_detector
    model = init_detector(Config.fromfile(a.config), a.ckpt, device='cuda:0'); model.eval()
    r = inference_detector(model, a.image)
    pi = r.pred_instances
    img = cv2.imread(a.image)
    dets = []
    for b, l, s in zip(pi.bboxes, pi.labels, pi.scores):
        sc = float(s)
        if sc < a.tau: continue
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(img, f'{CLS[int(l)]} {sc:.2f}', (x1, max(25, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        dets.append({'name': CLS[int(l)], 'score': round(sc, 4), 'bbox': [x1, y1, x2-x1, y2-y1]})
    cv2.imwrite(a.out, img)
    json.dump(dets, open(a.out.replace('.png', '.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'检出 {len(dets)} 个 (τ={a.tau}): {[(d["name"], d["score"]) for d in dets]}')
    print(f'保存: {a.out}')
