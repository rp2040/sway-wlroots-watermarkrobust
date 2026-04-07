import json
from pathlib import Path

import cv2
import numpy as np

ALPHA = 18.0
INPUT_IMAGE = 'sample.png'
TEMPLATE_NPY = 'template.npy'
PAYLOAD_JSON = 'payload.json'
OUTPUT_IMAGE = 'embedded.png'
OUTPUT_DEBUG = 'embedded_roi_box.png'

SYNC_VERTEX_NPY = 'sync_vertex.npy'
SYNC_EDGE_NPY = 'sync_edge.npy'

SYNC_ALPHA = 18.0



def blend_template(gray: np.ndarray, tpl: np.ndarray, x0: int, y0: int, alpha: float):
    h, w = gray.shape
    th, tw = tpl.shape
    if x0 < 0 or y0 < 0 or x0 + tw > w or y0 + th > h:
        return False

    roi = gray[y0:y0 + th, x0:x0 + tw]
    blended = np.clip(roi + alpha * tpl, 0, 255)
    gray[y0:y0 + th, x0:x0 + tw] = blended
    return True

def blend_template_centered(gray: np.ndarray, tpl: np.ndarray, cx: int, cy: int, alpha: float):
    th, tw = tpl.shape
    x0 = int(round(cx - tw / 2))
    y0 = int(round(cy - th / 2))
    return blend_template(gray, tpl, x0, y0, alpha)
    
def get_message_block_positions(w: int, h: int, bs: int):
    total_w = 2 * bs
    total_h = 2 * bs

    x0 = (w - total_w) // 2
    y0 = (h - total_h) // 2

    positions = [
        (x0, y0),             # roi1 
        (x0 + bs, y0),        # roi2 
        (x0, y0 + bs),        # roi3 
        (x0 + bs, y0 + bs),   # roi4 
    ]
    return positions

def get_vertex_grid_positions(msg_positions, bs: int):
    
    x0, y0 = msg_positions[0]

    vertices = []
    for row in range(3):
        for col in range(3):
            vx = x0 + col * bs
            vy = y0 + row * bs
            vertices.append((vx, vy))
    return vertices
    
def get_edge_midpoint_positions(msg_positions, bs: int):
    x0, y0 = msg_positions[0]

    edge_points = []

    
    for row in range(3):
        for col in range(2):
            ex = x0 + col * bs + bs // 2
            ey = y0 + row * bs
            edge_points.append((ex, ey))

    
    for row in range(2):
        for col in range(3):
            ex = x0 + col * bs
            ey = y0 + row * bs + bs // 2
            edge_points.append((ex, ey))

    return edge_points

def get_block_corners(x0: int, y0: int, bs: int):
    return {
        'tl': (x0, y0),
        'tr': (x0 + bs, y0),
        'bl': (x0, y0 + bs),
        'br': (x0 + bs, y0 + bs),
    }

def main():
    here = Path(__file__).resolve().parent
    img = cv2.imread(str(here / INPUT_IMAGE), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(INPUT_IMAGE)

    tm = np.load(here / TEMPLATE_NPY)
    sync_vertex = np.load(here / SYNC_VERTEX_NPY)
    sync_edge = np.load(here / SYNC_EDGE_NPY)

    meta = json.loads((here / PAYLOAD_JSON).read_text(encoding='utf-8'))
    bs = int(meta['block_size'])
    ss = int(meta['sync_size'])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    positions = get_message_block_positions(w, h, bs)
    vertex_points = get_vertex_grid_positions(positions, bs)
    edge_points = get_edge_midpoint_positions(positions, bs)

    all_sync_boxes = []

   
    for (x0, y0) in positions:
        if x0 < 0 or y0 < 0 or x0 + bs > w or y0 + bs > h:
            raise ValueError('sample image too small for packed 2x2 block layout')

        ok = blend_template(gray, tm, x0, y0, ALPHA)
        if not ok:
            raise ValueError('failed to embed message block')


    for (vx, vy) in vertex_points:
        if blend_template_centered(gray, sync_vertex, vx, vy, SYNC_ALPHA):
            all_sync_boxes.append((
                int(round(vx - ss / 2)),
                int(round(vy - ss / 2)),
                ss, ss, 'V'
            ))


    for (ex, ey) in edge_points:
        if blend_template_centered(gray, sync_edge, ex, ey, SYNC_ALPHA):
            all_sync_boxes.append((
                int(round(ex - ss / 2)),
                int(round(ey - ss / 2)),
                ss, ss, 'E'
            ))

    out = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(here / OUTPUT_IMAGE), out)
 
    debug = out.copy()


    for (x0, y0) in positions:
       cv2.rectangle(debug, (x0, y0), (x0 + bs, y0 + bs), (0, 0, 255), 2)


    sync_colors = {
        'V': (0, 255, 0),      # G
        'E': (255, 0, 0),      # B
    }

    for (sx, sy, sw, sh, tag) in all_sync_boxes:
        color = sync_colors.get(tag, (255, 255, 255))
        cv2.rectangle(debug, (sx, sy), (sx + sw, sy + sh), color, 2)

    cv2.imwrite(str(here / OUTPUT_DEBUG), debug)

    print('wrote', OUTPUT_IMAGE, OUTPUT_DEBUG)
    for i, (x0, y0) in enumerate(positions, start=1):
        print('msg roi%d:' % i, x0, y0, bs, bs)

    for i, (sx, sy, sw, sh, tag) in enumerate(all_sync_boxes, start=1):
        print('sync%d:' % i, tag, sx, sy, sw, sh)


if __name__ == '__main__':
    main()
