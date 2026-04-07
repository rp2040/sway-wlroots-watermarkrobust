import json
import zlib
from pathlib import Path

import cv2
import numpy as np

DEVICE_ID = 0x1A2B3C4D
TIMESLOT = 1234
BLOCK_SIZE = 256
RADII = [28, 37, 46, 53]
REPEAT = 3
TEMPLATE_PNG = 'template.png'
TEMPLATE_NPY = 'template.npy'
PAYLOAD_JSON = 'payload.json'

SYNC_SIZE = 64
SYNC_RADII = [10, 15, 19]

SYNC_VERTEX_PNG = 'sync_vertex.png'
SYNC_VERTEX_NPY = 'sync_vertex.npy'
SYNC_EDGE_PNG = 'sync_edge.png'
SYNC_EDGE_NPY = 'sync_edge.npy'


def u32_to_bits(v: int):
    return [(v >> i) & 1 for i in range(31, -1, -1)]


def u16_to_bits(v: int):
    return [(v >> i) & 1 for i in range(15, -1, -1)]


def bits_to_u16(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def build_payload_bits(device_id: int, timeslot: int):
    body = device_id.to_bytes(4, 'big') + timeslot.to_bytes(2, 'big')
    crc = zlib.crc32(body) & 0xFFFF
    bits = u32_to_bits(device_id) + u16_to_bits(timeslot) + u16_to_bits(crc)
    return bits, crc


def repetition_encode(bits, n=3):
    out = []
    for b in bits:
        out.extend([b] * n)
    return out

def spec_to_spatial(spec: np.ndarray):
    spatial = np.fft.ifft2(np.fft.ifftshift(spec)).real
    spatial -= spatial.mean()
    spatial /= (np.max(np.abs(spatial)) + 1e-8)
    return spatial.astype(np.float32)


def build_message_template(bits, size: int, radii):
    spec = np.zeros((size, size), dtype=np.complex64)
    center = size // 2
    nbits = len(bits)
    for j, bit in enumerate(bits):
        r = radii[j % len(radii)]
        theta = np.pi * j / nbits
        x = int(round(center + r * np.cos(theta)))
        y = int(round(center + r * np.sin(theta)))
        xm = int(round(center - r * np.cos(theta)))
        ym = int(round(center - r * np.sin(theta)))
        if 0 <= x < size and 0 <= y < size and 0 <= xm < size and 0 <= ym < size:
            # 1 bit -> strong positive frequency coefficient, 0 bit -> keep zero
            if bit == 1:
                spec[y, x] = 1.0 + 0.0j
                spec[ym, xm] = 1.0 + 0.0j
    # move DC to origin for ifft2
    return spec_to_spatial(spec)
    
def build_vertex_sync_template(size: int, radii):
    spec = np.zeros((size, size), dtype=np.complex64)
    center = size // 2

    # Pv
    angles_deg = [31, 38, 45, 52, 59]

    for r in radii:
        for deg in angles_deg:
            theta = np.deg2rad(deg)
            x = int(round(center + r * np.cos(theta)))
            y = int(round(center + r * np.sin(theta)))
            xm = int(round(center - r * np.cos(theta)))
            ym = int(round(center - r * np.sin(theta)))

            if 0 <= x < size and 0 <= y < size and 0 <= xm < size and 0 <= ym < size:
                spec[y, x] = 1.6 + 0.0j
                spec[ym, xm] = 1.6 + 0.0j

    return spec_to_spatial(spec)


def build_edge_sync_template(size: int, radii):
    spec = np.zeros((size, size), dtype=np.complex64)
    center = size // 2

    #  Pe
    angles_deg = [121, 128, 135, 142, 149]

    for r in radii:
        for deg in angles_deg:
            theta = np.deg2rad(deg)
            x = int(round(center + r * np.cos(theta)))
            y = int(round(center + r * np.sin(theta)))
            xm = int(round(center - r * np.cos(theta)))
            ym = int(round(center - r * np.sin(theta)))

            if 0 <= x < size and 0 <= y < size and 0 <= xm < size and 0 <= ym < size:
                spec[y, x] = 1.6 + 0.0j
                spec[ym, xm] = 1.6 + 0.0j

    return spec_to_spatial(spec)

def main():
    here = Path(__file__).resolve().parent
    payload_bits, crc = build_payload_bits(DEVICE_ID, TIMESLOT)
    encoded_bits = repetition_encode(payload_bits, REPEAT)

    tm = build_message_template(encoded_bits, BLOCK_SIZE, RADII)

    sync_vertex = build_vertex_sync_template(SYNC_SIZE, SYNC_RADII)
    sync_edge = build_edge_sync_template(SYNC_SIZE, SYNC_RADII)

    np.save(here / TEMPLATE_NPY, tm)
    tm_u8 = np.clip((tm * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(here / TEMPLATE_PNG), tm_u8)

    np.save(here / SYNC_VERTEX_NPY, sync_vertex)
    sync_vertex_u8 = np.clip((sync_vertex * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(here / SYNC_VERTEX_PNG), sync_vertex_u8)

    np.save(here / SYNC_EDGE_NPY, sync_edge)
    sync_edge_u8 = np.clip((sync_edge * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(here / SYNC_EDGE_PNG), sync_edge_u8)

    meta = {
        'device_id': DEVICE_ID,
        'timeslot': TIMESLOT,
        'crc16': crc,
        'block_size': BLOCK_SIZE,
        'radii': RADII,
        'repeat': REPEAT,
        'payload_bits_len': len(payload_bits),
        'encoded_bits_len': len(encoded_bits),
        'sync_size': SYNC_SIZE,
        'sync_radii': SYNC_RADII,
        'sync_templates': ['vertex', 'edge'],
    }
    (here / PAYLOAD_JSON).write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('wrote', TEMPLATE_NPY, TEMPLATE_PNG)
    print('wrote', SYNC_VERTEX_NPY, SYNC_VERTEX_PNG)
    print('wrote', SYNC_EDGE_NPY, SYNC_EDGE_PNG)
    print('wrote', PAYLOAD_JSON)


if __name__ == '__main__':
    main()
