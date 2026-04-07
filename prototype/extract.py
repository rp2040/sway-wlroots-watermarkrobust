# -*- coding: utf-8 -*-
import json
from pathlib import Path

import cv2
import numpy as np


INPUT_IMAGE = 'test_mix.png'
PAYLOAD_JSON = 'payload.json'
DEBUG_FFT = 'debug_fft.png'
DEBUG_DETECT = 'debug_detect.png'

DETECT_WIN = 64
DETECT_STEP = 8
DETECT_SCORE_THRESH = 200.0
DEBUG_QUAD = 'debug_quad.png'
DEBUG_WARP1 = 'debug_warp1.png'

WARP1_PAD = 32
ANGLE_THRESH_DEG = 35.0
LENGTH_RATIO_MAX = 1.6

DEBUG_EDGE = 'debug_edge.png'
DEBUG_REFINE = 'debug_refine.png'
DEBUG_WARP2 = 'debug_warp2.png'

EDGE_DETECT_STEP = 4
EDGE_SCORE_THRESH = 200.0

WARP1_CONTEXT = 80
CROP_MARGIN = 96

REFINE_SEARCH_BOX = 120
REFINE_STEP = 2


def majority_vote(bits, n=3):
    out = []
    for i in range(0, len(bits), n):
        chunk = bits[i:i+n]
        out.append(1 if sum(chunk) >= (len(chunk) / 2.0) else 0)
    return out


def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v



def detect_template_centers(img_gray: np.ndarray, tpl: np.ndarray, top_k: int, score_thresh: float):
    img_f = img_gray.astype(np.float32)
    tpl_f = tpl.astype(np.float32)

    
    resp = cv2.matchTemplate(img_f, tpl_f, cv2.TM_CCOEFF_NORMED)

    h, w = tpl.shape
    ys, xs = np.where(resp >= score_thresh)

    candidates = []
    for (yy, xx) in zip(ys.tolist(), xs.tolist()):
        score = float(resp[yy, xx])
        cx = xx + w / 2.0
        cy = yy + h / 2.0
        candidates.append((cx, cy, score))

    candidates.sort(key=lambda t: t[2], reverse=True)
    candidates = suppress_close_points(candidates, min_dist=max(h, w) * 0.5)
    candidates = candidates[:top_k]

    return candidates, resp



def build_rois_from_grid(grid_x, grid_y):
    rois = []
    for row in range(2):
        for col in range(2):
            x0 = int(round(grid_x[col]))
            y0 = int(round(grid_y[row]))
            x1 = int(round(grid_x[col + 1]))
            y1 = int(round(grid_y[row + 1]))

            rois.append((x0, y0, x1 - x0, y1 - y0))
    return rois

def suppress_close_points(points, min_dist):
    kept = []
    for p in points:
        x = p[0]
        y = p[1]

        ok = True
        for kp in kept:
            kx = kp[0]
            ky = kp[1]
            if (x - kx) ** 2 + (y - ky) ** 2 < min_dist * min_dist:
                ok = False
                break

        if ok:
            kept.append(p)

    return kept

def draw_detection_debug(img_gray, vertex_candidates, out_path):
    debug = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for (x, y, score, energy_v, energy_e) in vertex_candidates:
        cv2.circle(debug, (int(round(x)), int(round(y))), 5, (0, 255, 0), 2)

    cv2.imwrite(str(out_path), debug)

def draw_edge_debug(img_gray, edge_candidates, out_path):
    debug = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for (x, y, score, energy_v, energy_e) in edge_candidates:
        cv2.circle(debug, (int(round(x)), int(round(y))), 4, (255, 0, 0), 2)

    cv2.imwrite(str(out_path), debug)

def refine_sync_point_fft(img_gray: np.ndarray, cx: float, cy: float, win_size: int, radii, mode: str):
    h, w = img_gray.shape
    best = None

    half = REFINE_SEARCH_BOX // 2

    x_start = int(round(cx - half))
    x_end = int(round(cx + half))
    y_start = int(round(cy - half))
    y_end = int(round(cy + half))

    for ccy in range(y_start, y_end + 1, REFINE_STEP):
        for ccx in range(x_start, x_end + 1, REFINE_STEP):
            x0 = int(round(ccx - win_size / 2))
            y0 = int(round(ccy - win_size / 2))
            x1 = x0 + win_size
            y1 = y0 + win_size

            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue

            patch = img_gray[y0:y1, x0:x1].astype(np.float32)
            patch = patch - patch.mean()

            spectrum = np.fft.fftshift(np.fft.fft2(patch))
            mag = np.abs(spectrum)

            if mode == 'V':
                score, ev, ee = compute_vertex_score(mag, radii)
            elif mode == 'E':
                score, ev, ee = compute_edge_score(mag, radii)
            else:
                raise ValueError('mode must be V or E')

            cand = (float(ccx), float(ccy), float(score), float(ev), float(ee))
            if best is None or cand[2] > best[2]:
                best = cand

    return best
  
def draw_refine_debug(img_gray, quad_pts, refined_v_points, edge_candidates, refined_e_points, out_path):
    debug = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # 原始四边形（红）
    pts = quad_pts.astype(np.int32)
    cv2.line(debug, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[1]), tuple(pts[3]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[3]), tuple(pts[2]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[2]), tuple(pts[0]), (0, 0, 255), 2)

    # 粗 E 候选（蓝）
    for (x, y, score, ev, ee) in edge_candidates:
        cv2.circle(debug, (int(round(x)), int(round(y))), 4, (255, 0, 0), 1)

    # 精 V 点（绿）
    for p in refined_v_points:
        if p is None:
            continue
        x, y, score, ev, ee = p
        cv2.circle(debug, (int(round(x)), int(round(y))), 5, (0, 255, 0), 2)

    # 精 E 点（黄）
    for p in refined_e_points:
        if p is None:
            continue
        x, y, score, ev, ee = p
        cv2.circle(debug, (int(round(x)), int(round(y))), 5, (0, 255, 255), 2)

    cv2.imwrite(str(out_path), debug)  
   
def compute_vertex_score(mag: np.ndarray, radii):
    h, w = mag.shape
    cy = h // 2
    cx = w // 2

    def sample_energy(angles_deg):
        vals = []
        for r in radii:
            for deg in angles_deg:
                theta = np.deg2rad(deg)
                x = int(round(cx + r * np.cos(theta)))
                y = int(round(cy + r * np.sin(theta)))

                x1 = max(0, x - 1)
                y1 = max(0, y - 2)
                x2 = min(w, x + 2)
                y2 = min(h, y + 2)

                if x1 < x2 and y1 < y2:
                    vals.append(float(np.max(mag[y1:y2, x1:x2])))

        if not vals:
            return 0.0
        return float(np.sum(vals))

    # V: Q1
    energy_v = sample_energy([31, 38, 45, 52, 59])

    # E: Q2
    energy_e = sample_energy([121, 128, 135, 142, 149])

    score = energy_v - energy_e
    return score, energy_v, energy_e 

def compute_edge_score(mag: np.ndarray, radii):
    h, w = mag.shape
    cy = h // 2
    cx = w // 2

    def sample_energy(angles_deg):
        vals = []
        for r in radii:
            for deg in angles_deg:
                theta = np.deg2rad(deg)
                x = int(round(cx + r * np.cos(theta)))
                y = int(round(cy + r * np.sin(theta)))

                x1 = max(0, x - 1)
                y1 = max(0, y - 1)
                x2 = min(w, x + 2)
                y2 = min(h, y + 2)

                if x1 < x2 and y1 < y2:
                    vals.append(float(np.max(mag[y1:y2, x1:x2])))

        if not vals:
            return 0.0
        return float(np.sum(vals))

    # V: Q1
    energy_v = sample_energy([31, 38, 45, 52, 59])

    # E: Q2
    energy_e = sample_energy([121, 128, 135, 142, 149])

    score = energy_e - energy_v
    return score, energy_v, energy_e

def detect_edges_fft(img_gray: np.ndarray, win_size: int, radii, step: int, score_thresh: float):
    h, w = img_gray.shape
    candidates = []

    for y in range(0, h - win_size + 1, step):
        for x in range(0, w - win_size + 1, step):
            patch = img_gray[y:y + win_size, x:x + win_size].astype(np.float32)
            patch = patch - patch.mean()

            spectrum = np.fft.fftshift(np.fft.fft2(patch))
            mag = np.abs(spectrum)

            score, energy_v, energy_e = compute_edge_score(mag, radii)

            if score >= score_thresh:
                cx = x + win_size / 2.0
                cy = y + win_size / 2.0
                candidates.append((cx, cy, score, energy_v, energy_e))

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates


def detect_vertices_fft(img_gray: np.ndarray, win_size: int, radii, step: int, score_thresh: float):
    h, w = img_gray.shape
    candidates = []

    for y in range(0, h - win_size + 1, step):
        for x in range(0, w - win_size + 1, step):
            patch = img_gray[y:y + win_size, x:x + win_size].astype(np.float32)
            patch = patch - patch.mean()

            spectrum = np.fft.fftshift(np.fft.fft2(patch))
            mag = np.abs(spectrum)

            score, energy_v, energy_e = compute_vertex_score(mag, radii)

            if score >= score_thresh:
                cx = x + win_size / 2.0
                cy = y + win_size / 2.0
                candidates.append((cx, cy, score, energy_v, energy_e))

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates


def extract_encoded_bits(roi: np.ndarray, nbits: int, radii):
    h, w = roi.shape
    assert h == w
    size = h
    center = size // 2

    spectrum = np.fft.fftshift(np.fft.fft2(roi.astype(np.float32)))
    mag = np.abs(spectrum)
    mag_log = np.log1p(mag)

    mag_u8 = (255 * mag_log / (mag_log.max() + 1e-8)).astype(np.uint8)
    mag_u8 = np.ascontiguousarray(mag_u8)

  
    values = []
    ring_ids = []
    sample_points = []

   
    ring_colors = [64, 128, 192, 255]

    for j in range(nbits):
        ring_id = j % len(radii)
        r = radii[ring_id]
        theta = np.pi * j / nbits

        x = int(round(center + r * np.cos(theta)))
        y = int(round(center + r * np.sin(theta)))

        x1 = max(0, x - 1)
        y1 = max(0, y - 1)
        x2 = min(size, x + 2)
        y2 = min(size, y + 2)

        v = float(np.max(mag[y1:y2, x1:x2]))
        values.append(v)
        ring_ids.append(ring_id)
        sample_points.append((x, y))

        color = int(ring_colors[ring_id % len(ring_colors)])
        cv2.rectangle(mag_u8, (x1, y1), (x2 - 1, y2 - 1), color, 1)

    values = np.array(values, dtype=np.float32)
    ring_ids = np.array(ring_ids, dtype=np.int32)

   
    bits = np.zeros(nbits, dtype=np.uint8)
    ring_thresholds = []

    for rid in range(len(radii)):
        mask = (ring_ids == rid)
        ring_vals = values[mask]
        th = float(ring_vals.mean() + 0.35 * ring_vals.std())
        ring_thresholds.append(th)
        bits[mask] = (ring_vals >= th).astype(np.uint8)

    debug_info = {
        'values': values,
        'ring_ids': ring_ids,
        'sample_points': sample_points,
        'ring_thresholds': ring_thresholds,
    }
    return bits.tolist(), mag_u8, ring_thresholds, debug_info

def decode_one_roi(roi: np.ndarray, radii, repeat, encoded_bits_len):
    roi = roi.astype(np.float32)
    roi = roi - roi.mean()

    encoded_bits, fft_debug, ring_thresholds, debug_info = extract_encoded_bits(
        roi, encoded_bits_len, radii
    )

    bits64 = majority_vote(encoded_bits, repeat)
    device_id = bits_to_int(bits64[:32])
    timeslot = bits_to_int(bits64[32:48])
    crc16 = bits_to_int(bits64[48:64])

    import zlib
    body = device_id.to_bytes(4, 'big') + timeslot.to_bytes(2, 'big')
    expected_crc = zlib.crc32(body) & 0xFFFF
    crc_ok = (expected_crc == crc16)

    return {
        'device_id': device_id,
        'timeslot': timeslot,
        'crc16': crc16,
        'expected_crc': expected_crc,
        'crc_ok': crc_ok,
        'fft_debug': fft_debug,
        'ring_thresholds': ring_thresholds,
        'debug_info': debug_info,
    }

def point_dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return float(np.sqrt(dx * dx + dy * dy))



def build_quad_candidates_axis(vertex_candidates):
    quads = []
    angle_thresh = np.deg2rad(35.0)

    pts = vertex_candidates[:12]

    for i, pa in enumerate(pts):
        ax, ay, ascore, _, _ = pa

        horiz = []
        vert = []

        for j, pb in enumerate(pts):
            if i == j:
                continue

            bx, by, bscore, _, _ = pb
            dx = bx - ax
            dy = by - ay
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                continue

            angle = np.arctan2(abs(dy), abs(dx))

            # 近水平
            if angle <= angle_thresh:
                horiz.append((dist, pb))

            # 近垂直
            if abs(np.pi / 2 - angle) <= angle_thresh:
                vert.append((dist, pb))

        horiz.sort(key=lambda t: t[0])
        vert.sort(key=lambda t: t[0])

        if not horiz or not vert:
            continue

        pb = horiz[0][1]
        pc = vert[0][1]

        bx, by, bscore, _, _ = pb
        cx, cy, cscore, _, _ = pc

        dx_pred = bx + cx - ax
        dy_pred = by + cy - ay

        best_pd = None
        best_pd_err = None

        for k, pd in enumerate(pts):
            if k == i:
                continue
            if pd is pb or pd is pc:
                continue

            dx, dy, dscore, _, _ = pd
            err = np.sqrt((dx - dx_pred) ** 2 + (dy - dy_pred) ** 2)

            if best_pd is None or err < best_pd_err:
                best_pd = pd
                best_pd_err = err

        if best_pd is None:
            continue

        dx, dy, dscore, _, _ = best_pd

        edge_h = point_dist((ax, ay), (bx, by))
        edge_v = point_dist((ax, ay), (cx, cy))
        edge_ref = 0.5 * (edge_h + edge_v)

        if best_pd_err > 0.35 * edge_ref:
            continue

        ratio = max(edge_h, edge_v) / max(1e-6, min(edge_h, edge_v))
        if ratio > LENGTH_RATIO_MAX:
            continue

        quad_pts = np.array([
            [ax, ay],
            [bx, by],
            [cx, cy],
            [dx, dy],
        ], dtype=np.float32)

        s = quad_pts[:, 0] + quad_pts[:, 1]
        diff = quad_pts[:, 0] - quad_pts[:, 1]

        tl = quad_pts[np.argmin(s)]
        br = quad_pts[np.argmax(s)]
        tr = quad_pts[np.argmax(diff)]
        bl = quad_pts[np.argmin(diff)]

        ordered = np.array([tl, tr, bl, br], dtype=np.float32)

        
        quad_score = max(ascore, bscore, cscore, dscore)

        quads.append({
            'points': ordered,
            'score': quad_score,
            'raw_points': [pa, pb, pc, best_pd],
            'mode': 'axis',
        })

    # 去重：同一个 cell 的四个角点如果几乎一样，只保留一个
    unique_quads = []
    for q in quads:
        pts = q['points']

        found_same = False
        for uq in unique_quads:
            upts = uq['points']
            d = np.mean(np.sqrt(np.sum((pts - upts) ** 2, axis=1)))
            if d < 12.0:
                found_same = True
                break

        if not found_same:
            unique_quads.append(q)

    unique_quads.sort(key=lambda q: q['score'], reverse=True)
    return unique_quads


def build_quad_candidates_relaxed(vertex_candidates):
    quads = []

    # 只使用当前顶点候选的前若干个高分点
    pts = vertex_candidates[:12]

    for i, pa in enumerate(pts):
        ax, ay, ascore, _, _ = pa

    # 不再假设“近水平/近垂直”，
    # 直接从最近邻里找两个不同方向的邻点
        dist_list = []

        for j, pb in enumerate(pts):
            if i == j:
                continue

            bx, by, bscore, _, _ = pb
            dx = bx - ax
            dy = by - ay
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                continue

            dist_list.append((dist, pb))
 
        dist_list.sort(key=lambda t: t[0])

        if len(dist_list) < 2:
            continue

        pb = dist_list[0][1]
        pc = None

        # 第二个邻点不能和第一个邻点几乎共线
        bx, by, _, _, _ = pb
        v1x = bx - ax
        v1y = by - ay
        norm1 = np.sqrt(v1x * v1x + v1y * v1y)

        for _, cand in dist_list[1:]:
            cx, cy, _, _, _ = cand
            v2x = cx - ax
            v2y = cy - ay
            norm2 = np.sqrt(v2x * v2x + v2y * v2y)
            if norm2 < 1e-6:
                continue

            cosang = abs((v1x * v2x + v1y * v2y) / (norm1 * norm2))
            # cosang 接近 1 表示几乎共线，跳过
            if cosang < 0.85:
                pc = cand
                break

        if pc is None:
            continue

        bx, by, bscore, _, _ = pb
        cx, cy, cscore, _, _ = pc

        # 预测对角点 pd ≈ pb + pc - pa
        dx_pred = bx + cx - ax
        dy_pred = by + cy - ay

        best_pd = None
        best_pd_err = None

        for k, pd in enumerate(pts):
            if k == i:
                continue
            if pd is pb or pd is pc:
                continue

            dx, dy, dscore, _, _ = pd
            err = np.sqrt((dx - dx_pred) ** 2 + (dy - dy_pred) ** 2)

            if best_pd is None or err < best_pd_err:
                best_pd = pd
                best_pd_err = err

        if best_pd is None:
            continue

        dx, dy, dscore, _, _ = best_pd

        # 几何约束 1：对角点不能离预测点太远
        edge_h = point_dist((ax, ay), (bx, by))
        edge_v = point_dist((ax, ay), (cx, cy))
        edge_ref = 0.5 * (edge_h + edge_v)

        if best_pd_err > 0.35 * edge_ref:
            continue

        # 几何约束 2：横竖边长度不能差太离谱
        ratio = max(edge_h, edge_v) / max(1e-6, min(edge_h, edge_v))
        if ratio > LENGTH_RATIO_MAX:
            continue

        # 四点集合：先按 TL,TR,BL,BR 近似排序
        quad_pts = np.array([
            [ax, ay],
            [bx, by],
            [cx, cy],
            [dx, dy],
        ], dtype=np.float32)

        # 用 x+y / x-y 近似排序成 tl,tr,bl,br
        s = quad_pts[:, 0] + quad_pts[:, 1]
        diff = quad_pts[:, 0] - quad_pts[:, 1]

        tl = quad_pts[np.argmin(s)]
        br = quad_pts[np.argmax(s)]
        tr = quad_pts[np.argmax(diff)]
        bl = quad_pts[np.argmin(diff)]

        ordered = np.array([tl, tr, bl, br], dtype=np.float32)

        quad_score = float((ascore + bscore + cscore + dscore) / 4.0)

        quads.append({
            'points': ordered,
            'score': float(quad_score),
            'raw_points': [pa, pb, pc, best_pd],
        })

    # 按可信度排序，优先尝试高响应四元组
    quads.sort(key=lambda q: q['score'], reverse=True)
    return quads

def build_quad_candidates_from_vertices(vertex_candidates):
    return build_quad_candidates_axis(vertex_candidates)

def warp_from_quad(img_gray: np.ndarray, quad_pts: np.ndarray, block_size: int, context: int, crop_margin: int):
    h, w = img_gray.shape

    x_min = max(0, int(np.floor(np.min(quad_pts[:, 0]) - crop_margin)))
    y_min = max(0, int(np.floor(np.min(quad_pts[:, 1]) - crop_margin)))
    x_max = min(w, int(np.ceil(np.max(quad_pts[:, 0]) + crop_margin)))
    y_max = min(h, int(np.ceil(np.max(quad_pts[:, 1]) + crop_margin)))

    cropped = img_gray[y_min:y_max, x_min:x_max].copy()

    quad_local = quad_pts.copy().astype(np.float32)
    quad_local[:, 0] -= x_min
    quad_local[:, 1] -= y_min

    out_size = block_size + 2 * context

    dst = np.array([
        [context, context],
        [context + block_size, context],
        [context, context + block_size],
        [context + block_size, context + block_size],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(quad_local, dst)
    warped = cv2.warpPerspective(cropped, H, (out_size, out_size))

    crop_box = (x_min, y_min, x_max - x_min, y_max - y_min)
    return warped, H, crop_box
    
def warp_from_refined_vertices(img_gray: np.ndarray, refined_v_points, block_size: int):
    # refined_v_points 顺序约定：
    # 0 = TL, 1 = TR, 2 = BL, 3 = BR
    src = np.array([
        [refined_v_points[0][0], refined_v_points[0][1]],
        [refined_v_points[1][0], refined_v_points[1][1]],
        [refined_v_points[2][0], refined_v_points[2][1]],
        [refined_v_points[3][0], refined_v_points[3][1]],
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [block_size, 0],
        [0, block_size],
        [block_size, block_size],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_gray, H, (block_size, block_size))
    return warped, H
 
def warp_from_refined_sync_points(img_gray: np.ndarray, refined_v_points, refined_e_points, block_size: int):
    # refined_v_points 顺序：
    # 0 = TL, 1 = TR, 2 = BL, 3 = BR
    #
    # refined_e_points 顺序：
    # 0 = TOP, 1 = BOTTOM, 2 = LEFT, 3 = RIGHT

    src = np.array([
        [refined_v_points[0][0], refined_v_points[0][1]],  # TL
        [refined_v_points[1][0], refined_v_points[1][1]],  # TR
        [refined_v_points[2][0], refined_v_points[2][1]],  # BL
        [refined_v_points[3][0], refined_v_points[3][1]],  # BR
        [refined_e_points[0][0], refined_e_points[0][1]],  # TOP
        [refined_e_points[1][0], refined_e_points[1][1]],  # BOTTOM
        [refined_e_points[2][0], refined_e_points[2][1]],  # LEFT
        [refined_e_points[3][0], refined_e_points[3][1]],  # RIGHT
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],                              # TL
        [block_size, 0],                     # TR
        [0, block_size],                     # BL
        [block_size, block_size],            # BR
        [block_size / 2.0, 0],               # TOP
        [block_size / 2.0, block_size],      # BOTTOM
        [0, block_size / 2.0],               # LEFT
        [block_size, block_size / 2.0],      # RIGHT
    ], dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError('findHomography failed for refined V/E points')

    warped = cv2.warpPerspective(img_gray, H, (block_size, block_size))
    return warped, H
   
def try_decode_from_quad(img_gray, quad_pts, sync_radii, bs, here):
    # 1) 第一次粗校正
    warped1, H1, crop_box = warp_from_quad(
        img_gray=img_gray,
        quad_pts=quad_pts,
        block_size=bs,
        context=WARP1_CONTEXT,
        crop_margin=CROP_MARGIN,
    )

    # 2) 在粗校正图里检测 E
    edge_candidates = detect_edges_fft(
        img_gray=warped1,
        win_size=DETECT_WIN,
        radii=sync_radii,
        step=EDGE_DETECT_STEP,
        score_thresh=EDGE_SCORE_THRESH,
    )

    edge_candidates = suppress_close_points(
        edge_candidates,
        min_dist=DETECT_WIN * 1.0
    )

    if edge_candidates:
        max_e_score = max(p[2] for p in edge_candidates)
        edge_candidates = [
            p for p in edge_candidates
            if p[2] > max_e_score * 0.5
        ]

    # 3) warp1 中的理论 V / E 位置
    approx_v = [
        (WARP1_CONTEXT, WARP1_CONTEXT),                   # TL
        (WARP1_CONTEXT + bs, WARP1_CONTEXT),             # TR
        (WARP1_CONTEXT, WARP1_CONTEXT + bs),             # BL
        (WARP1_CONTEXT + bs, WARP1_CONTEXT + bs),        # BR
    ]

    approx_e = [
        (WARP1_CONTEXT + bs / 2, WARP1_CONTEXT),                 # top
        (WARP1_CONTEXT + bs / 2, WARP1_CONTEXT + bs),           # bottom
        (WARP1_CONTEXT, WARP1_CONTEXT + bs / 2),                # left
        (WARP1_CONTEXT + bs, WARP1_CONTEXT + bs / 2),           # right
    ]

    # 4) 精定位 V
    refined_v = []
    for (cx, cy) in approx_v:
        best = refine_sync_point_fft(
            img_gray=warped1,
            cx=cx,
            cy=cy,
            win_size=DETECT_WIN,
            radii=sync_radii,
            mode='V'
        )
        refined_v.append(best)

    # 5) 精定位 E
    refined_e = []
    for (cx, cy) in approx_e:
        best = refine_sync_point_fft(
            img_gray=warped1,
            cx=cx,
            cy=cy,
            win_size=DETECT_WIN,
            radii=sync_radii,
            mode='E'
        )
        refined_e.append(best)

    # 6) 几何筛选：E 应该接近四条边中点
    expected_e = approx_e
    for i, p in enumerate(refined_e):
        if p is None:
            raise RuntimeError('refined E point missing')

        ex, ey = expected_e[i]
        px, py = p[0], p[1]
        d = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        if d > 24:
            raise RuntimeError('refined E point too far from expected midpoint')

    # 7) 第二次精透视校正：用 refined V + refined E 一起估计更稳的 H
    warped2, H2 = warp_from_refined_sync_points(
        img_gray=warped1,
        refined_v_points=refined_v,
        refined_e_points=refined_e,
        block_size=bs,
    )

    # 8) 消息解码
    meta = json.loads((here / PAYLOAD_JSON).read_text(encoding='utf-8'))
    radii = list(meta['radii'])
    repeat = int(meta['repeat'])
    encoded_bits_len = int(meta['encoded_bits_len'])

    result = decode_one_roi(
        roi=warped2,
        radii=radii,
        repeat=repeat,
        encoded_bits_len=encoded_bits_len,
    )

    result['warp1'] = warped1
    result['warp2'] = warped2
    result['edge_candidates'] = edge_candidates
    result['refined_v'] = refined_v
    result['refined_e'] = refined_e
    result['crop_box'] = crop_box
    result['quad_pts'] = quad_pts
    return result

def draw_quad_debug(img_gray: np.ndarray, quad_pts: np.ndarray, out_path):
    debug = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    pts = quad_pts.astype(np.int32)

    # 画四条边
    cv2.line(debug, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[1]), tuple(pts[3]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[3]), tuple(pts[2]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[2]), tuple(pts[0]), (0, 0, 255), 2)

    # 画四个角点
    for i, p in enumerate(pts):
        cv2.circle(debug, tuple(p), 5, (0, 255, 0), 2)
        cv2.putText(
            debug,
            str(i),
            (int(p[0]) + 6, int(p[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(str(out_path), debug)

def main():
    here = Path(__file__).resolve().parent

    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(INPUT_IMAGE)

    meta = json.loads((here / PAYLOAD_JSON).read_text(encoding='utf-8'))
    sync_radii = list(meta['sync_radii'])
    bs = int(meta['block_size'])

    # 第一步：盲检测 V 候选点
    vertex_candidates = detect_vertices_fft(
        img_gray=img,
        win_size=DETECT_WIN,
        radii=sync_radii,
        step=DETECT_STEP,
        score_thresh=DETECT_SCORE_THRESH,
    )

    vertex_candidates = suppress_close_points(
        vertex_candidates,
        min_dist=DETECT_WIN * 1.5
    )

    if not vertex_candidates:
        raise RuntimeError('no vertex candidates found')

    max_score = max(p[2] for p in vertex_candidates)
    vertex_candidates = [
        p for p in vertex_candidates
        if p[2] > max_score * 0.5
    ]

    draw_detection_debug(
        img_gray=img,
        vertex_candidates=vertex_candidates,
        out_path=here / DEBUG_DETECT,
    )

    print('vertex candidates:')
    for i, (x, y, score, energy_v, energy_e) in enumerate(vertex_candidates, start=1):
        print(
            '  V%d: (%.2f, %.2f) score=%.4f ev=%.4f ee=%.4f'
            % (i, x, y, score, energy_v, energy_e)
        )

    # 第二步：拼四元组
    quads = build_quad_candidates_from_vertices(vertex_candidates)
    if not quads:
        raise RuntimeError('no valid quad candidate built from vertex points')

    print('quad candidates:', len(quads))

    best_result = None
    best_quad_idx = None
    best_crc_ok_result = None
    best_crc_ok_quad_idx = None
    last_exc = None

    # 第三步：按置信度从高到低尝试四元组
    for qi, quad in enumerate(quads, start=1):
        quad_pts = quad['points']
        print('trying quad %d score=%.4f' % (qi, quad['score']))

        try:
            result = try_decode_from_quad(
                img_gray=img,
                quad_pts=quad_pts,
                sync_radii=sync_radii,
                bs=bs,
                here=here,
            )

            # 保留第一组完整跑通的结果，供最终兜底调试
            if best_result is None:
                best_result = result
                best_quad_idx = qi

            # 关键：只要 CRC 正确，立即认为成功并返回该结果
            if result['crc_ok']:
                best_crc_ok_result = result
                best_crc_ok_quad_idx = qi

                draw_quad_debug(
                    img_gray=img,
                    quad_pts=quad_pts,
                    out_path=here / DEBUG_QUAD,
                )
                cv2.imwrite(str(here / DEBUG_WARP1), result['warp1'])
                cv2.imwrite(str(here / DEBUG_WARP2), result['warp2'])

                draw_edge_debug(
                    img_gray=result['warp1'],
                    edge_candidates=result['edge_candidates'],
                    out_path=here / DEBUG_EDGE,
                )

                draw_refine_debug(
                    img_gray=result['warp1'],
                    quad_pts=np.array([
                        [WARP1_CONTEXT, WARP1_CONTEXT],
                        [WARP1_CONTEXT + bs, WARP1_CONTEXT],
                        [WARP1_CONTEXT, WARP1_CONTEXT + bs],
                        [WARP1_CONTEXT + bs, WARP1_CONTEXT + bs],
                    ], dtype=np.float32),
                    refined_v_points=result['refined_v'],
                    edge_candidates=result['edge_candidates'],
                    refined_e_points=result['refined_e'],
                    out_path=here / DEBUG_REFINE,
                )

                best_result = best_crc_ok_result
                best_quad_idx = best_crc_ok_quad_idx
                break

            else:
                print('  quad %d decoded but crc failed' % qi)

        except Exception as e:
            last_exc = e
            print('  quad %d failed: %s' % (qi, str(e)))
            continue

    if best_crc_ok_result is not None:
        best_result = best_crc_ok_result
        best_quad_idx = best_crc_ok_quad_idx
    elif best_result is None:
        if last_exc is not None:
            raise RuntimeError('all quad attempts failed, last error: %s' % str(last_exc))
        raise RuntimeError('all quad attempts failed')
    else:
        print('warning: no quad passed crc, showing best non-crc result')

    print('chosen quad index:', best_quad_idx)
    print('crop box:', best_result['crop_box'])

    print('refined V points:')
    for i, p in enumerate(best_result['refined_v'], start=1):
        if p is None:
            print('  V%d: None' % i)
        else:
            print('  V%d: (%.2f, %.2f) score=%.4f' % (i, p[0], p[1], p[2]))

    print('refined E points:')
    for i, p in enumerate(best_result['refined_e'], start=1):
        if p is None:
            print('  E%d: None' % i)
        else:
            print('  E%d: (%.2f, %.2f) score=%.4f' % (i, p[0], p[1], p[2]))

    print('decoded deviceid: 0x%08x' % best_result['device_id'])
    print('decoded timeslot:', best_result['timeslot'])
    print('decoded crc16: 0x%04x' % best_result['crc16'])
    print('expected crc16: 0x%04x' % best_result['expected_crc'])
    print('crc ok:', best_result['crc_ok'])

    print('wrote', DEBUG_DETECT)
    print('wrote', DEBUG_QUAD)
    print('wrote', DEBUG_WARP1)
    print('wrote', DEBUG_EDGE)
    print('wrote', DEBUG_REFINE)
    print('wrote', DEBUG_WARP2)


if __name__ == '__main__':
    main()
