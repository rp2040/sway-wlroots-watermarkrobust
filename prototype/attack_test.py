# -*- coding: utf-8 -*-
import cv2
import numpy as np

INPUT_IMAGE = 'embedded.png'


def perspective_transform(img, dx_ratio=0.0):
    h, w = img.shape[:2]

    dx = int(w * dx_ratio)

    src = np.float32([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ])

    dst = np.float32([
        [dx, 0],
        [w - dx, 0],
        [0, h],
        [w, h]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped


def jpeg_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def main():
    img = cv2.imread(INPUT_IMAGE)

    # baseline
    cv2.imwrite('test_baseline.png', img)

    # 15°„
    img_p15 = perspective_transform(img, dx_ratio=0.08)
    cv2.imwrite('test_p15.png', img_p15)

    # 30°„
    img_p30 = perspective_transform(img, dx_ratio=0.15)
    cv2.imwrite('test_p30.png', img_p30)

    # JPEG75
    img_j75 = jpeg_compress(img, 75)
    cv2.imwrite('test_j75.png', img_j75)
    
    # JPEG70
    img_j70 = jpeg_compress(img, 70)
    cv2.imwrite('test_j70.png', img_j70)

    # JPEG65
    img_j65 = jpeg_compress(img, 65)
    cv2.imwrite('test_j65.png', img_j65)

    # JPEG60
    img_j60 = jpeg_compress(img, 60)
    cv2.imwrite('test_j60.png', img_j60)

    # Õ∏ ” + JPEG
    img_mix = jpeg_compress(img_p15, 75)
    cv2.imwrite('test_mix.png', img_mix)

    print("generated test images done.")


if __name__ == '__main__':
    main()