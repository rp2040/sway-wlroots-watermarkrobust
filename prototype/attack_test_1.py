from pathlib import Path

import cv2
import numpy as np

INPUT_IMAGE = 'embedded.png'


def apply_blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def apply_scale_roundtrip(img: np.ndarray, scale: float = 0.85) -> np.ndarray:
    h, w = img.shape[:2]
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def apply_gamma(img: np.ndarray, gamma: float = 1.25) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    y = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return y


def apply_jpeg(img: np.ndarray, quality: int = 70) -> np.ndarray:
    ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('jpeg encode failed')
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        raise RuntimeError('jpeg decode failed')
    return dec


def main():
    here = Path(__file__).resolve().parent
    img = cv2.imread(str(here / INPUT_IMAGE), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(INPUT_IMAGE)

    out_blur = apply_blur(img)
    out_scale = apply_scale_roundtrip(img, 0.85)
    out_gamma = apply_gamma(img, 1.25)

    blur_name = 'embedded_blur.png'
    scale_name = 'embedded_scale.png'
    gamma_name = 'embedded_gamma.png'

    cv2.imwrite(str(here / blur_name), out_blur)
    cv2.imwrite(str(here / scale_name), out_scale)
    cv2.imwrite(str(here / gamma_name), out_gamma)

    print('wrote', blur_name)
    print('wrote', scale_name)
    print('wrote', gamma_name)

    jpeg_qualities = [90, 80, 70, 60, 50, 40]
    for q in jpeg_qualities:
       out_jpeg = apply_jpeg(img, q)
       jpeg_name = f'embedded_jpeg_q{q}.jpg'
       cv2.imwrite(str(here / jpeg_name), out_jpeg)
       print('wrote', jpeg_name)


if __name__ == '__main__':
    main()