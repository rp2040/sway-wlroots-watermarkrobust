# -*- coding: utf-8 -*-
from pathlib import Path
import json
import math
import random

import cv2
import numpy as np

INPUT_IMAGE = 'embedded.png'
SEED = 20260412


def apply_jpeg(img: np.ndarray, quality: int = 75) -> np.ndarray:
    ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('jpeg encode failed')
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        raise RuntimeError('jpeg decode failed')
    return dec


def apply_blur(img: np.ndarray, ksize=(3, 3), sigma=0.8) -> np.ndarray:
    return cv2.GaussianBlur(img, ksize, sigma)


def apply_scale_roundtrip(img: np.ndarray, scale: float = 0.85) -> np.ndarray:
    h, w = img.shape[:2]
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def apply_gamma(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    y = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return y


def apply_illum_gradient(img: np.ndarray, mode: str = 'lr', strength: float = 0.10) -> np.ndarray:
    h, w = img.shape[:2]
    if mode == 'lr':
        grad = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)
        grad = np.tile(grad[None, :], (h, 1))
    elif mode == 'tb':
        grad = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)
        grad = np.tile(grad[:, None], (1, w))
    else:
        raise ValueError('mode must be lr or tb')

    out = img.astype(np.float32)
    out[:, :, 0] *= grad
    out[:, :, 1] *= grad
    out[:, :, 2] *= grad
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_median(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


def apply_crop_resize(img: np.ndarray, crop_ratio: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    dx = int(round(w * crop_ratio))
    dy = int(round(h * crop_ratio))
    cropped = img[dy:h - dy, dx:w - dx]
    if cropped.size == 0:
        return img.copy()
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_rotate_crop_resize(img: np.ndarray, angle_deg: float = 3.0) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rotated

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    crop = rotated[y0:y1, x0:x1]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_perspective_random(img: np.ndarray, strength: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    h, w = img.shape[:2]
    d = int(round(min(w, h) * strength))

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1],
    ])

    dst = np.float32([
        [rng.randint(0, d), rng.randint(0, d)],
        [w - 1 - rng.randint(0, d), rng.randint(0, d)],
        [rng.randint(0, d), h - 1 - rng.randint(0, d)],
        [w - 1 - rng.randint(0, d), h - 1 - rng.randint(0, d)],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


def save_case(here: Path, name: str, img: np.ndarray, meta_list: list):
    out_path = here / name
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f'failed to write {out_path}')
    meta_list.append({'file': name})


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    here = Path(__file__).resolve().parent
    img = cv2.imread(str(here / INPUT_IMAGE), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(INPUT_IMAGE)

    records = []

    # A. 基础组
    cases = {}
    cases['baseline.png'] = img.copy()
    cases['jpeg_q80.jpg'] = apply_jpeg(img, 80)
    cases['jpeg_q70.jpg'] = apply_jpeg(img, 70)
    cases['jpeg_q60.jpg'] = apply_jpeg(img, 60)
    cases['blur_g3.png'] = apply_blur(img, (3, 3), 0.8)
    cases['scale_085.png'] = apply_scale_roundtrip(img, 0.85)

    # B. 几何组
    cases['persp_light_1.png'] = apply_perspective_random(img, 0.035, seed=101)
    cases['persp_light_2.png'] = apply_perspective_random(img, 0.045, seed=102)
    cases['persp_mid_1.png'] = apply_perspective_random(img, 0.060, seed=201)
    cases['persp_mid_2.png'] = apply_perspective_random(img, 0.075, seed=202)
    cases['persp_strong_1.png'] = apply_perspective_random(img, 0.100, seed=301)
    cases['rot_crop_p3.png'] = apply_rotate_crop_resize(img, 3.0)
    cases['rot_crop_n3.png'] = apply_rotate_crop_resize(img, -3.0)
    cases['crop5_resize.png'] = apply_crop_resize(img, 0.05)

    # C. 成像退化组
    cases['gamma_085.png'] = apply_gamma(img, 0.85)
    cases['gamma_120.png'] = apply_gamma(img, 1.20)
    cases['illum_lr.png'] = apply_illum_gradient(img, 'lr', 0.10)
    cases['illum_tb.png'] = apply_illum_gradient(img, 'tb', 0.10)
    cases['noise_s3.png'] = apply_gaussian_noise(img, 3.0)
    cases['noise_s5.png'] = apply_gaussian_noise(img, 5.0)
    cases['median_3.png'] = apply_median(img, 3)

    # D. 复合组
    tmp = apply_perspective_random(img, 0.04, seed=401)
    cases['mix_persp_light_j80.jpg'] = apply_jpeg(tmp, 80)

    tmp = apply_perspective_random(img, 0.07, seed=402)
    cases['mix_persp_mid_j70.jpg'] = apply_jpeg(tmp, 70)

    tmp = apply_perspective_random(img, 0.07, seed=403)
    tmp = apply_blur(tmp, (3, 3), 0.8)
    cases['mix_persp_mid_blur_j70.jpg'] = apply_jpeg(tmp, 70)

    tmp = apply_perspective_random(img, 0.04, seed=404)
    tmp = apply_gamma(tmp, 1.20)
    cases['mix_persp_light_gamma_j75.jpg'] = apply_jpeg(tmp, 75)

    tmp = apply_crop_resize(img, 0.05)
    cases['mix_crop5_resize_j70.jpg'] = apply_jpeg(tmp, 70)

    tmp = apply_rotate_crop_resize(img, 3.0)
    cases['mix_rot3_j75.jpg'] = apply_jpeg(tmp, 75)

    for name, out in cases.items():
        ok = cv2.imwrite(str(here / name), out)
        if not ok:
            raise RuntimeError(f'failed to write {name}')
        records.append({'file': name})

    meta = {
        'input': INPUT_IMAGE,
        'seed': SEED,
        'count': len(records),
        'files': records,
    }
    (here / 'attack_cases.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('generated', len(records), 'attack images')
    print('wrote attack_cases.json')


if __name__ == '__main__':
    main()