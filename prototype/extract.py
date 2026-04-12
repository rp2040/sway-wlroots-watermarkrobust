# -*- coding: utf-8 -*-
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
from scipy.signal import wiener


PAYLOAD_JSON = 'payload.json'

DEBUG_NOISE = 'debug_noise.png'
DEBUG_DETECT = 'debug_detect.png'
DEBUG_QUAD = 'debug_quad.png'
DEBUG_WARP1 = 'debug_warp1.png'
DEBUG_EDGE = 'debug_edge.png'
DEBUG_REFINE = 'debug_refine.png'
DEBUG_WARP2 = 'debug_warp2.png'

DETECT_WIN = 64
DETECT_STEP = 8
DETECT_SCORE_THRESH = 120.0

EDGE_DETECT_STEP = 4
EDGE_SCORE_THRESH = 120.0

SCALE_CANDIDATES = [1.0, 0.9, 0.8, 0.7, 0.6]

WARP1_CONTEXT = 80
CROP_MARGIN = 96

REFINE_SEARCH_BOX = 120
REFINE_STEP = 2

LENGTH_RATIO_MAX = 1.8


def to_u8_for_save(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.copy()
        arr[~np.isfinite(arr)] = 0.0
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax - vmin < 1e-8:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = (arr - vmin) / (vmax - vmin)
        arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)

    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)

    raise TypeError(f'unsupported dtype: {arr.dtype}')


def safe_imwrite(path, img: np.ndarray):
    ok = cv2.imwrite(str(path), to_u8_for_save(img))
    if not ok:
        raise RuntimeError(f'failed to write image: {path}')


def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def bits_to_bytes(bits):
    if len(bits) % 8 != 0:
        raise ValueError('bits length must be multiple of 8')
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | int(b)
        out.append(byte)
    return bytes(out)


def bytes_to_bits(buf: bytes):
    out = []
    for x in buf:
        for i in range(7, -1, -1):
            out.append((x >> i) & 1)
    return out


def parse_raw_message_bits(raw_bits, year_base: int):
    if len(raw_bits) != 64:
        raise ValueError(f'raw message bits must be 64, got {len(raw_bits)}')

    device_id = bits_to_int(raw_bits[0:32])
    year_offset = bits_to_int(raw_bits[32:38])
    month = bits_to_int(raw_bits[38:42])
    day = bits_to_int(raw_bits[42:47])
    hour = bits_to_int(raw_bits[47:52])
    minute = bits_to_int(raw_bits[52:58])
    second = bits_to_int(raw_bits[58:64])

    year = year_base + year_offset

    valid = True
    if not (1 <= month <= 12):
        valid = False
    if not (1 <= day <= 31):
        valid = False
    if not (0 <= hour <= 23):
        valid = False
    if not (0 <= minute <= 59):
        valid = False
    if not (0 <= second <= 59):
        valid = False

    return {
        'device_id': device_id,
        'year': year,
        'year_offset': year_offset,
        'month': month,
        'day': day,
        'hour': hour,
        'minute': minute,
        'second': second,
        'valid_datetime': valid,
    }


def decode_bits_with_bch(code_bits, meta):
    try:
        import bchlib
    except ImportError as e:
        raise RuntimeError('need bchlib: pip3 install bchlib') from e

    data_bytes = int(meta['data_bytes'])
    ecc_bytes = int(meta['ecc_bytes'])
    bch_m = int(meta['bch_m'])
    bch_t = int(meta['bch_t'])
    raw_bits_len = int(meta['raw_bits_len'])

    total_bytes = data_bytes + ecc_bytes
    expected_code_bits_len = total_bytes * 8
    if len(code_bits) != expected_code_bits_len:
        raise ValueError(
            f'code bits len mismatch, expect {expected_code_bits_len}, got {len(code_bits)}'
        )

    code_buf = bits_to_bytes(code_bits)
    packet_data = bytearray(code_buf[:data_bytes])
    packet_ecc = bytearray(code_buf[data_bytes:])

    bch = bchlib.BCH(m=bch_m, t=bch_t)

    bitflips = bch.decode(data=packet_data, recv_ecc=packet_ecc)

    if bitflips < 0:
        return {
            'decode_ok': False,
            'bitflips': bitflips,
            'raw_bits': None,
            'raw_bytes': None,
        }

    bch.correct(data=packet_data, ecc=packet_ecc)
    raw_bytes = bytes(packet_data)
    raw_bits = bytes_to_bits(raw_bytes)[:raw_bits_len]

    return {
        'decode_ok': True,
        'bitflips': bitflips,
        'raw_bits': raw_bits,
        'raw_bytes': raw_bytes,
    }


def build_noise_image(img_gray: np.ndarray):
    img_f = img_gray.astype(np.float32)
    den = wiener(img_f, (5, 5))
    den = np.asarray(den, dtype=np.float32)
    noise = img_f - den
    noise = np.clip(noise, -64.0, 64.0)
    return noise.astype(np.float32)
    
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
                y1 = max(0, y - 1)
                x2 = min(w, x + 2)
                y2 = min(h, y + 2)
                if x1 < x2 and y1 < y2:
                    vals.append(float(np.max(mag[y1:y2, x1:x2])))
        return float(np.sum(vals)) if vals else 0.0

    energy_v = sample_energy([31, 38, 45, 52, 59])
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
        return float(np.sum(vals)) if vals else 0.0

    energy_v = sample_energy([31, 38, 45, 52, 59])
    energy_e = sample_energy([121, 128, 135, 142, 149])
    score = energy_e - energy_v
    return score, energy_v, energy_e


def detect_vertices_fft_with_response(img_gray: np.ndarray, win_size: int, radii, step: int, score_thresh: float):
    h, w = img_gray.shape
    candidates = []
    all_scores = []

    for y in range(0, h - win_size + 1, step):
        for x in range(0, w - win_size + 1, step):
            patch = img_gray[y:y + win_size, x:x + win_size].astype(np.float32)
            patch = patch - patch.mean()

            spectrum = np.fft.fftshift(np.fft.fft2(patch))
            mag = np.abs(spectrum)

            score, energy_v, energy_e = compute_vertex_score(mag, radii)
            all_scores.append(score)

            if score >= score_thresh:
                cx = x + win_size / 2.0
                cy = y + win_size / 2.0
                candidates.append((cx, cy, score, energy_v, energy_e))

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates, np.array(all_scores, dtype=np.float32)


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


def select_best_scale_for_vertices(noise_img: np.ndarray, sync_radii):
    best = None

    for s in SCALE_CANDIDATES:
        if s == 1.0:
            scaled = noise_img
        else:
            scaled = cv2.resize(
                noise_img,
                None,
                fx=s,
                fy=s,
                interpolation=cv2.INTER_AREA
            )

        candidates, all_scores = detect_vertices_fft_with_response(
            img_gray=scaled,
            win_size=DETECT_WIN,
            radii=sync_radii,
            step=DETECT_STEP,
            score_thresh=DETECT_SCORE_THRESH,
        )

        sigma_s = float(all_scores.std()) if all_scores.size else 0.0
        mean_s = float(all_scores.mean()) if all_scores.size else 0.0
        Rs = sigma_s * (s ** 2)

        item = {
            'scale': s,
            'image': scaled,
            'candidates': candidates,
            'all_scores': all_scores,
            'sigma_s': sigma_s,
            'mean_s': mean_s,
            'Rs': Rs,
        }

        if best is None or item['Rs'] > best['Rs']:
            best = item

    return best


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
    debug = cv2.cvtColor(to_u8_for_save(img_gray), cv2.COLOR_GRAY2BGR)
    for (x, y, score, energy_v, energy_e) in vertex_candidates:
        cv2.circle(debug, (int(round(x)), int(round(y))), 5, (0, 255, 0), 2)
    safe_imwrite(out_path, debug)


def draw_edge_debug(img_gray, edge_candidates, out_path):
    debug = cv2.cvtColor(to_u8_for_save(img_gray), cv2.COLOR_GRAY2BGR)
    for (x, y, score, energy_v, energy_e) in edge_candidates:
        cv2.circle(debug, (int(round(x)), int(round(y))), 4, (255, 0, 0), 2)
    safe_imwrite(out_path, debug)
    
def point_dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return float(np.sqrt(dx * dx + dy * dy))


def build_quad_candidates_from_vertices(vertex_candidates):
    quads = []
    pts = vertex_candidates[:12]

    for i, pa in enumerate(pts):
        ax, ay, ascore, _, _ = pa
        dist_list = []

        for j, pb in enumerate(pts):
            if i == j:
                continue
            bx, by, _, _, _ = pb
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
            if cosang < 0.85:
                pc = cand
                break

        if pc is None:
            continue

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

        quad_score = float((ascore + bscore + cscore + dscore) / 4.0)

        quads.append({
            'points': ordered,
            'score': quad_score,
            'raw_points': [pa, pb, pc, best_pd],
        })

    unique_quads = []
    for q in quads:
        pts_q = q['points']
        found_same = False
        for uq in unique_quads:
            upts = uq['points']
            d = np.mean(np.sqrt(np.sum((pts_q - upts) ** 2, axis=1)))
            if d < 12.0:
                found_same = True
                break
        if not found_same:
            unique_quads.append(q)

    unique_quads.sort(key=lambda q: q['score'], reverse=True)
    return unique_quads


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
            else:
                score, ev, ee = compute_edge_score(mag, radii)

            cand = (float(ccx), float(ccy), float(score), float(ev), float(ee))
            if best is None or cand[2] > best[2]:
                best = cand

    return best


def warp_from_refined_sync_points(img_gray: np.ndarray, refined_v_points, refined_e_points, block_size: int):
    src = np.array([
        [refined_v_points[0][0], refined_v_points[0][1]],
        [refined_v_points[1][0], refined_v_points[1][1]],
        [refined_v_points[2][0], refined_v_points[2][1]],
        [refined_v_points[3][0], refined_v_points[3][1]],
        [refined_e_points[0][0], refined_e_points[0][1]],
        [refined_e_points[1][0], refined_e_points[1][1]],
        [refined_e_points[2][0], refined_e_points[2][1]],
        [refined_e_points[3][0], refined_e_points[3][1]],
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [block_size, 0],
        [0, block_size],
        [block_size, block_size],
        [block_size / 2.0, 0],
        [block_size / 2.0, block_size],
        [0, block_size / 2.0],
        [block_size, block_size / 2.0],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError('findHomography failed for refined sync points')

    warped = cv2.warpPerspective(img_gray, H, (block_size, block_size))
    return warped, H

def extract_encoded_bits(roi: np.ndarray, nbits: int, radii, k3: float):
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
    ring_thresholds = []
    ring_stats = []

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

        cv2.rectangle(mag_u8, (x1, y1), (x2 - 1, y2 - 1), 255, 1)

    values = np.array(values, dtype=np.float32)
    ring_ids = np.array(ring_ids, dtype=np.int32)

    bits = np.zeros(nbits, dtype=np.uint8)

    for rid, r in enumerate(radii):
        ring_band_vals = []
        yy, xx = np.indices((size, size))
        rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        mask_band = ((rr >= (r - 2)) & (rr <= (r + 2)))
        band = mag[mask_band]

        if band.size == 0:
            mean_ring = 0.0
            std_ring = 1.0
        else:
            mean_ring = float(band.mean())
            std_ring = float(band.std())

        th = mean_ring + k3 * std_ring
        ring_thresholds.append(th)
        ring_stats.append({
            'ring_id': rid,
            'radius': r,
            'mean': mean_ring,
            'std': std_ring,
            'threshold': th,
        })

        mask = (ring_ids == rid)
        bits[mask] = (values[mask] >= th).astype(np.uint8)

    debug_info = {
        'values': values,
        'ring_ids': ring_ids,
        'sample_points': sample_points,
        'ring_thresholds': ring_thresholds,
        'ring_stats': ring_stats,
    }

    return bits.tolist(), mag_u8, ring_thresholds, debug_info


def decode_one_roi(roi: np.ndarray, meta, k3: float):
    roi = roi.astype(np.float32)
    roi = roi - roi.mean()

    radii = list(meta['radii'])
    code_bits_len = int(meta['code_bits_len'])
    year_base = int(meta.get('year_base', 2024))
    expected_device_id = int(meta['device_id'])

    code_bits, fft_debug, ring_thresholds, debug_info = extract_encoded_bits(
        roi=roi,
        nbits=code_bits_len,
        radii=radii,
        k3=k3,
    )

    dec = decode_bits_with_bch(code_bits, meta)
    if not dec['decode_ok']:
        return {
            'decode_ok': False,
            'reason': 'bch_decode_failed',
            'bitflips': dec.get('bitflips', -1),
            'fft_debug': fft_debug,
            'ring_thresholds': ring_thresholds,
            'debug_info': debug_info,
        }

    raw_bits = dec['raw_bits']
    parsed = parse_raw_message_bits(raw_bits, year_base=year_base)

    device_ok = (parsed['device_id'] == expected_device_id)
    semantic_ok = parsed['valid_datetime']

    parsed.update({
        'decode_ok': bool(device_ok and semantic_ok),
        'device_ok': device_ok,
        'semantic_ok': semantic_ok,
        'bitflips': dec['bitflips'],
        'raw_bits': raw_bits,
        'code_bits': code_bits,
        'fft_debug': fft_debug,
        'ring_thresholds': ring_thresholds,
        'debug_info': debug_info,
    })
    return parsed


def draw_quad_debug(img_gray: np.ndarray, quad_pts: np.ndarray, out_path):
    debug = cv2.cvtColor(to_u8_for_save(img_gray), cv2.COLOR_GRAY2BGR)
    pts = quad_pts.astype(np.int32)

    cv2.line(debug, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[1]), tuple(pts[3]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[3]), tuple(pts[2]), (0, 0, 255), 2)
    cv2.line(debug, tuple(pts[2]), tuple(pts[0]), (0, 0, 255), 2)

    for i, p in enumerate(pts):
        cv2.circle(debug, tuple(p), 5, (0, 255, 0), 2)
        cv2.putText(debug, str(i), (int(p[0]) + 6, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    safe_imwrite(out_path, debug)


def draw_refine_debug(img_gray, refined_v_points, edge_candidates, refined_e_points, out_path):
    debug = cv2.cvtColor(to_u8_for_save(img_gray), cv2.COLOR_GRAY2BGR)

    for (x, y, score, ev, ee) in edge_candidates:
        cv2.circle(debug, (int(round(x)), int(round(y))), 4, (255, 0, 0), 1)

    for p in refined_v_points:
        x, y, score, ev, ee = p
        cv2.circle(debug, (int(round(x)), int(round(y))), 5, (0, 255, 0), 2)

    for p in refined_e_points:
        x, y, score, ev, ee = p
        cv2.circle(debug, (int(round(x)), int(round(y))), 5, (0, 255, 255), 2)

    safe_imwrite(out_path, debug)


def try_decode_from_quad(img_gray, quad_pts, sync_radii, bs, meta):
    warped1, H1, crop_box = warp_from_quad(
        img_gray=img_gray,
        quad_pts=quad_pts,
        block_size=bs,
        context=WARP1_CONTEXT,
        crop_margin=CROP_MARGIN,
    )

    edge_candidates = detect_edges_fft(
        img_gray=warped1,
        win_size=DETECT_WIN,
        radii=sync_radii,
        step=EDGE_DETECT_STEP,
        score_thresh=EDGE_SCORE_THRESH,
    )

    edge_candidates = suppress_close_points(edge_candidates, min_dist=DETECT_WIN * 1.0)
    if edge_candidates:
        max_e_score = max(p[2] for p in edge_candidates)
        edge_candidates = [p for p in edge_candidates if p[2] > max_e_score * 0.5]

    approx_v = [
        (WARP1_CONTEXT, WARP1_CONTEXT),
        (WARP1_CONTEXT + bs, WARP1_CONTEXT),
        (WARP1_CONTEXT, WARP1_CONTEXT + bs),
        (WARP1_CONTEXT + bs, WARP1_CONTEXT + bs),
    ]

    approx_e = [
        (WARP1_CONTEXT + bs / 2, WARP1_CONTEXT),
        (WARP1_CONTEXT + bs / 2, WARP1_CONTEXT + bs),
        (WARP1_CONTEXT, WARP1_CONTEXT + bs / 2),
        (WARP1_CONTEXT + bs, WARP1_CONTEXT + bs / 2),
    ]

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
        if best is None:
            raise RuntimeError('refined V point missing')
        refined_v.append(best)

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
        if best is None:
            raise RuntimeError('refined E point missing')
        refined_e.append(best)

    for i, p in enumerate(refined_e):
        ex, ey = approx_e[i]
        px, py = p[0], p[1]
        d = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        if d > 24:
            raise RuntimeError('refined E point too far from expected midpoint')

    warped2, H2 = warp_from_refined_sync_points(
        img_gray=warped1,
        refined_v_points=refined_v,
        refined_e_points=refined_e,
        block_size=bs,
    )

    result = decode_one_roi(
        roi=warped2,
        meta=meta,
        k3=float(meta.get('k3', 1.2)),
    )

    result['warp1'] = warped1
    result['warp2'] = warped2
    result['edge_candidates'] = edge_candidates
    result['refined_v'] = refined_v
    result['refined_e'] = refined_e
    result['crop_box'] = crop_box
    result['quad_pts'] = quad_pts
    return result


def main():
    here = Path(__file__).resolve().parent

    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
        if not input_path.is_absolute():
            input_path = here / input_path
    else:
        input_path = here / 'embedded.png'

    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(input_path))

    meta = json.loads((here / PAYLOAD_JSON).read_text(encoding='utf-8'))
    sync_radii = list(meta['sync_radii'])
    bs = int(meta['block_size'])

    noise_img = build_noise_image(img)

    noise_vis = noise_img.copy()
    noise_vis = noise_vis - noise_vis.min()
    noise_vis = noise_vis / (noise_vis.max() + 1e-8)
    noise_vis = (noise_vis * 255.0).astype(np.uint8)
    safe_imwrite(here / DEBUG_NOISE, noise_vis)

    best_scale = select_best_scale_for_vertices(noise_img, sync_radii)
    scaled_img = best_scale['image']
    scale_factor = best_scale['scale']

    print('input:', input_path.name)
    print('best scale:', scale_factor)
    print('best scale Rs:', best_scale['Rs'])
    print('best scale sigma:', best_scale['sigma_s'])

    vertex_candidates = best_scale['candidates']
    vertex_candidates = suppress_close_points(vertex_candidates, min_dist=DETECT_WIN * 1.5)

    if not vertex_candidates:
        raise RuntimeError('no vertex candidates found')

    max_score = max(p[2] for p in vertex_candidates)
    vertex_candidates = [p for p in vertex_candidates if p[2] > max_score * 0.5]

    draw_detection_debug(scaled_img, vertex_candidates, here / DEBUG_DETECT)

    quads = build_quad_candidates_from_vertices(vertex_candidates)
    if not quads:
        raise RuntimeError('no valid quad candidate built from vertex points')

    print('quad candidates:', len(quads))

    best_result = None
    best_quad_idx = None
    last_exc = None

    for qi, quad in enumerate(quads, start=1):
        quad_pts = quad['points']
        print('trying quad %d score=%.4f' % (qi, quad['score']))

        try:
            result = try_decode_from_quad(
                img_gray=scaled_img,
                quad_pts=quad_pts,
                sync_radii=sync_radii,
                bs=bs,
                meta=meta,
            )

            # 无论成功失败，先把当前 quad 的调试结果写出来
            draw_quad_debug(scaled_img, quad_pts, here / f'fail_quad_{qi}.png')
            safe_imwrite(here / f'fail_warp1_{qi}.png', result['warp1'])
            safe_imwrite(here / f'fail_warp2_{qi}.png', result['warp2'])
            draw_edge_debug(result['warp1'], result['edge_candidates'], here / f'fail_edge_{qi}.png')
            draw_refine_debug(
                result['warp1'],
                result['refined_v'],
                result['edge_candidates'],
                result['refined_e'],
                here / f'fail_refine_{qi}.png',
            )

            if result.get('decode_ok', False):
                best_result = result
                best_quad_idx = qi

                # 成功时再额外写一份固定命名的“最终结果”
                draw_quad_debug(scaled_img, quad_pts, here / DEBUG_QUAD)
                safe_imwrite(here / DEBUG_WARP1, result['warp1'])
                safe_imwrite(here / DEBUG_WARP2, result['warp2'])
                draw_edge_debug(result['warp1'], result['edge_candidates'], here / DEBUG_EDGE)
                draw_refine_debug(
                    result['warp1'],
                    result['refined_v'],
                    result['edge_candidates'],
                    result['refined_e'],
                    here / DEBUG_REFINE,
                )
                break
            else:
                if best_result is None:
                    best_result = result
                    best_quad_idx = qi
                print('  quad %d decoded but semantic/BCH check failed' % qi)

        except Exception as e:
            last_exc = e
            print('  quad %d failed: %s' % (qi, str(e)))
            continue

    if best_result is None:
        if last_exc is not None:
            raise RuntimeError('all quad attempts failed, last error: %s' % str(last_exc))
        raise RuntimeError('all quad attempts failed')

    print('chosen quad index:', best_quad_idx)
    print('crop box:', best_result.get('crop_box'))

    print('decoded deviceid: 0x%08x' % best_result.get('device_id', 0))
    print('expected deviceid: 0x%08x' % int(meta['device_id']))
    print('device ok:', best_result.get('device_ok', False))

    if all(k in best_result for k in ['year', 'month', 'day', 'hour', 'minute', 'second']):
        print(
            'decoded datetime: %04d-%02d-%02d %02d:%02d:%02d'
            % (
                best_result['year'],
                best_result['month'],
                best_result['day'],
                best_result['hour'],
                best_result['minute'],
                best_result['second'],
            )
        )
        print('datetime semantic ok:', best_result.get('semantic_ok', False))
        if best_result.get('decode_ok', False):
            try:
                dt_utc = datetime(
                    best_result['year'],
                    best_result['month'],
                    best_result['day'],
                    best_result['hour'],
                    best_result['minute'],
                    best_result['second'],
                    tzinfo=timezone.utc
                )
                print('decoded utc:', dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC'))
            except Exception:
                print('decoded utc: invalid')
    else:
        print('decoded datetime: unavailable')

    print('bch decode ok:', best_result.get('decode_ok', False))
    print('bch bitflips:', best_result.get('bitflips', -1))

    print('wrote', DEBUG_NOISE)
    print('wrote', DEBUG_DETECT)
    print('wrote', DEBUG_QUAD)
    print('wrote', DEBUG_WARP1)
    print('wrote', DEBUG_EDGE)
    print('wrote', DEBUG_REFINE)
    print('wrote', DEBUG_WARP2)


if __name__ == '__main__':
    main()
    
