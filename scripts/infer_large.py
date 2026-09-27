#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大图推理(整景滑窗):1152×1152 切片, overlap 152, stride 1000,
每 tile 下采样到 800×800 送检 → 坐标映射回原图 → 跨 tile 全局 NMS(同类别 IoU=0.5)
→ 输出可视化(画框+置信度)与 COCO JSON 结果。

用法:
    python infer_large.py --image big.png --ckpt best.pth --config cfg.py \
        [--out_dir ./out] [--tau 0.70] [--gpu 0] [--save_vis]

依赖: mmdet 3.3 + 本包 mmdet_custom(见 README 环境准备)
"""
import argparse, json, os, time, cv2, numpy as np, torch
import warnings
warnings.filterwarnings('ignore')

CLS = ['HM','LQS','QHS','MS','A1_SU-35','A2_C-130','A3_C-17','A4_C-5','A5_F-16',
       'A6_TU-160','A7_E-3','A8_B-52','A9_P-3C','A10_B-1B','A11_E-8','A12_TU-22',
       'A13_F-15','A14_KC-135','A15_F-22','A16_FA-18','A17_TU-95','A18_KC-10',
       'A19_SU-34','A20_SU-24','FSC']

TILE, OVERLAP, TARGET = 1152, 152, 800   # 推理策略:1152² overlap152 → 800²
COLLECT_THR = 0.05                        # 低阈收集
NMS_IOU = 0.5                             # 全局同类别 NMS

def load_model(cfg_path, ckpt, device):
    from mmengine.config import Config
    from mmdet.apis import init_detector
    cfg = Config.fromfile(cfg_path)
    model = init_detector(cfg, ckpt, device=device)
    model.eval()
    return model

def global_nms(dets, iou_thr=NMS_IOU):
    """dets: [(cls, x1,y1,x2,y2, score)] → 同类别全局 NMS"""
    if not dets:
        return []
    arr = np.array(dets, dtype=object)
    cls_arr = np.array([d[0] for d in dets])
    out = []
    for c in np.unique(cls_arr):
        sel = np.array([d[1:] for d in dets if d[0] == c], dtype=float)
        order = np.argsort(-sel[:, 4])
        sel = sel[order]
        keep = []
        suppressed = np.zeros(len(sel), dtype=bool)
        for i in range(len(sel)):
            if suppressed[i]:
                continue
            keep.append(sel[i])
            xx1 = np.maximum(sel[i, 0], sel[i+1:, 0]); yy1 = np.maximum(sel[i, 1], sel[i+1:, 1])
            xx2 = np.minimum(sel[i, 2], sel[i+1:, 2]); yy2 = np.minimum(sel[i, 3], sel[i+1:, 3])
            iw = np.maximum(0, xx2-xx1); ih = np.maximum(0, yy2-yy1)
            inter = iw*ih
            area_i = (sel[i,2]-sel[i,0])*(sel[i,3]-sel[i,1])
            area_j = (sel[i+1:,2]-sel[i+1:,0])*(sel[i+1:,3]-sel[i+1:,1])
            iou = inter/(area_i+area_j-inter+1e-10)
            suppressed[i+1:] |= iou > iou_thr
        out.extend([(int(c), *k) for k in keep])
    return out

def infer_large(model, image_path, device, tau, save_vis, out_dir):
    from mmdet.apis import inference_detector
    big = cv2.imread(image_path)
    H, W = big.shape[:2]
    stride = TILE - OVERLAP
    xs = list(range(0, W - TILE + 1, stride))
    if xs[-1] + TILE < W: xs.append(W - TILE)
    ys = list(range(0, H - TILE + 1, stride))
    if ys[-1] + TILE < H: ys.append(H - TILE)
    n = len(xs)*len(ys)
    print(f'影像 {W}x{H} | tiles {len(xs)}x{len(ys)}={n} | {TILE}² overlap{OVERLAP} → {TARGET}²')
    torch.backends.cudnn.benchmark = True
    # warmup
    t0 = time.perf_counter()
    raw = []
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            tile = big[y0:y0+TILE, x0:x0+TILE]
            tile800 = cv2.resize(tile, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
            r = inference_detector(model, tile800)
            pi = r.pred_instances
            if len(pi.scores) == 0:
                continue
            scl = TILE / TARGET          # 800 坐标 → 1152 坐标
            for b, l, s in zip(pi.bboxes, pi.labels, pi.scores):
                sc = float(s)
                if sc < COLLECT_THR: continue
                x1, y1, x2, y2 = [float(v) for v in b]
                raw.append((int(l), x0+x1*scl, y0+y1*scl, x0+x2*scl, y0+y2*scl, sc))
            if (yi*len(xs)+xi+1) % 25 == 0:
                print(f'  {yi*len(xs)+xi+1}/{n} tiles', flush=True)
    dt = time.perf_counter() - t0
    print(f'模型推理完成: {n} tiles, {dt:.1f}s ({dt/n*1000:.0f} ms/tile)')
    # 全局 NMS
    t1 = time.perf_counter()
    keep = global_nms(raw, NMS_IOU)
    print(f'全局 NMS: {len(raw)} → {len(keep)} (IoU {NMS_IOU}) {time.perf_counter()-t1:.2f}s')
    # τ 后处理
    final = [d for d in keep if d[5] >= tau]
    print(f'τ={tau} 后保留 {len(final)} 个检测')
    os.makedirs(out_dir, exist_ok=True)
    # JSON(COCO 格式, 类别从 1 计数, 与 mmdet 推断一致起见保留 0-based 字段 label)
    js = [{'label': d[0], 'category_id': d[0]+1, 'name': CLS[d[0]],
           'bbox': [round(d[1],1), round(d[2],1), round(d[3]-d[1],1), round(d[4]-d[2],1)],
           'score': round(float(d[5]), 4)} for d in final]
    json.dump(js, open(f'{out_dir}/detections.json', 'w'), ensure_ascii=False, indent=1)
    if save_vis:
        vis = big.copy()
        for d in final:
            x1, y1, x2, y2 = int(d[1]), int(d[2]), int(d[3]), int(d[4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 4)
            cv2.putText(vis, f'{CLS[d[0]]} {d[5]:.2f}', (x1, max(28, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
        out_img = f'{out_dir}/vis_result.jpg'
        cv2.imwrite(out_img, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'可视化: {out_img}')
    return final, dt

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--config', required=True, help='模型配置(finaltrain_abl_a0_crt_config.py 等)')
    ap.add_argument('--ckpt', required=True, help='best_coco_bbox_mAP_epoch_1.pth(C1)')
    ap.add_argument('--out_dir', default='./infer_out')
    ap.add_argument('--tau', type=float, default=0.70)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--save_vis', action='store_true')
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    model = load_model(a.config, a.ckpt, 'cuda:0')
    final, t = infer_large(model, a.image, 'cuda:0', a.tau, a.save_vis, a.out_dir)
    print(f'DONE: 整景推理 {t:.1f}s, 检出 {len(final)} (见 {a.out_dir}/detections.json)')
