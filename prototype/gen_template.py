# -*- coding: utf-8 -*-
from pathlib import Path
import json
from datetime import datetime, timezone, timedelta

import numpy as np

BLOCK_SIZE = 256

# Use 4 rings (more stable than 2 rings for 100+ bits)
RADII = [28, 36, 46, 53]

SYNC_SIZE = 64
SYNC_RADII = [10, 15, 19]

YEAR_BASE = 2024

# BCH params (compatible with your bchlib)
BCH_M = 7
BCH_T = 5   # reduced redundancy

OUT_MSG = 'template_msg.npy'
OUT_V = 'template_sync_v.npy'
OUT_E = 'template_sync_e.npy'
OUT_META = 'payload.json'


def build_message_bits(device_id: int, dt: datetime):
    year_offset = dt.year - YEAR_BASE
    if not (0 <= year_offset < 64):
        raise ValueError('year out of range')

    bits = []

    def append(value, nbits):
        for i in range(nbits - 1, -1, -1):
            bits.append((value >> i) & 1)

    append(device_id, 32)
    append(year_offset, 6)
    append(dt.month, 4)
    append(dt.day, 5)
    append(dt.hour, 5)
    append(dt.minute, 6)
    append(dt.second, 6)

    return bits


def bits_to_bytes(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for bit in bits[i:i+8]:
            b = (b << 1) | bit
        out.append(b)
    return bytes(out)


def bytes_to_bits(buf: bytes):
    bits = []
    for x in buf:
        for i in range(7, -1, -1):
            bits.append((x >> i) & 1)
    return bits


def bch_encode(bits):
    import bchlib

    data = bits_to_bytes(bits)
    bch = bchlib.BCH(m=BCH_M, t=BCH_T)

    print("BCH:", "m=", bch.m, "t=", bch.t, "ecc_bytes=", bch.ecc_bytes)

    ecc = bch.encode(data)
    packet = data + ecc

    return bytes_to_bits(packet), len(data), len(ecc)


def build_message_template(bits, size, radii):
    spec = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    nbits = len(bits)

    for j, bit in enumerate(bits):
        if bit == 0:
            continue

        r = radii[j % len(radii)]
        theta = np.pi * j / nbits

        x = int(round(center + r * np.cos(theta)))
        y = int(round(center + r * np.sin(theta)))

        spec[y, x] += 1.0
        spec[(size - y) % size, (size - x) % size] += 1.0

    spatial = np.fft.ifft2(np.fft.ifftshift(spec)).real
    spatial -= spatial.mean()
    spatial /= (np.max(np.abs(spatial)) + 1e-8)
    return spatial.astype(np.float32)


def build_sync_template(size, radii, angles):
    spec = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    for r in radii:
        for deg in angles:
            theta = np.deg2rad(deg)
            x = int(round(center + r * np.cos(theta)))
            y = int(round(center + r * np.sin(theta)))

            spec[y, x] += 1.0
            spec[(size - y) % size, (size - x) % size] += 1.0

    spatial = np.fft.ifft2(np.fft.ifftshift(spec)).real
    spatial -= spatial.mean()
    spatial /= (np.max(np.abs(spatial)) + 1e-8)
    return spatial.astype(np.float32)


def main():
    here = Path(__file__).resolve().parent

    device_id = 0x1A2B3C4D
    now = datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0)

    raw_bits = build_message_bits(device_id, now)
    code_bits, data_bytes, ecc_bytes = bch_encode(raw_bits)

    print("raw bits:", len(raw_bits))
    print("code bits:", len(code_bits))

    msg_tpl = build_message_template(code_bits, BLOCK_SIZE, RADII)

    sync_v = build_sync_template(
        SYNC_SIZE, SYNC_RADII, [31, 38, 45, 52, 59]
    )
    sync_e = build_sync_template(
        SYNC_SIZE, SYNC_RADII, [121, 128, 135, 142, 149]
    )

    np.save(here / OUT_MSG, msg_tpl)
    np.save(here / OUT_V, sync_v)
    np.save(here / OUT_E, sync_e)

    meta = {
        "device_id": device_id,

        "block_size": BLOCK_SIZE,
        "radii": RADII,

        "sync_size": SYNC_SIZE,
        "sync_radii": SYNC_RADII,

        "year_base": YEAR_BASE,
        "raw_bits_len": 64,

        "coding": "bch",
        "data_bytes": data_bytes,
        "ecc_bytes": ecc_bytes,
        "bch_m": BCH_M,
        "bch_t": BCH_T,

        "code_bits_len": len(code_bits),

        "k3": 1.5,

    
        "generated_utc": now.strftime("%Y-%m-%d %H:%M:%S %z"),
        "datetime_fields": {
           "year": now.year,
           "month": now.month,
           "day": now.day,
           "hour": now.hour,
           "minute": now.minute,
           "second": now.second
        }
    }

    (here / OUT_META).write_text(json.dumps(meta, indent=2))

    print("done")


if __name__ == "__main__":
    main()