"""Numba-compiled inner loops for the pipeline's hot spots.

Pixel-level work NumPy can't vectorize -- union-find labeling, the sequential
small-region merge, and the bilateral filter's per-pixel weighting -- runs
here as compiled code. Arrays are passed flattened (row-major) with explicit
``height``/``width``.

Kernels compile on first call and are cached on disk (``cache=True``), so only
the first run after an install pays the compile cost; ``warm_up`` pays it
ahead of time.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:  # path compression
        up = parent[i]
        parent[i] = root
        i = up
    return root


@njit(cache=True, nogil=True)
def _union(parent, a, b):
    # Always keep the smaller pixel index as root, so a component's root is
    # its first pixel in raster order.
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra < rb:
        parent[rb] = ra
    elif rb < ra:
        parent[ra] = rb


@njit(cache=True, nogil=True)
def label_components(labels, height, width, num_colors, out_ids):
    """8-connected components of equal values in ``labels``.

    Writes each pixel's region id to ``out_ids`` (-1 where the label is
    outside ``[0, num_colors)``). Ids are ordered by label value, then by the
    first 2x2 pixel block (in raster order of blocks) the component touches.
    That is the numbering produced by running ``cv2.connectedComponents`` on
    each label's mask in turn, whose 8-connectivity algorithms scan 2x2 blocks.
    A block can't hold two components of one label (its pixels are all
    8-adjacent), so the order is well defined.

    Returns ``(region_color, areas)``, indexed by region id.
    """
    n = height * width
    parent = np.empty(n, np.int32)
    for p in range(n):
        parent[p] = p

    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            c = labels[p]
            if c < 0 or c >= num_colors:
                continue
            if y > 0 and labels[p - width] == c:
                # The pixel above is 8-adjacent to the left, upper-left and
                # upper-right pixels too, so any of those with this label
                # are already in its component.
                _union(parent, p, p - width)
                continue
            if x > 0 and labels[p - 1] == c:
                _union(parent, p, p - 1)
            elif y > 0 and x > 0 and labels[p - width - 1] == c:
                _union(parent, p, p - width - 1)
            if y > 0 and x + 1 < width and labels[p - width + 1] == c:
                _union(parent, p, p - width + 1)

    count_per_label = np.zeros(num_colors, np.int64)
    for p in range(n):
        c = labels[p]
        if 0 <= c < num_colors and _find(parent, p) == p:
            count_per_label[c] += 1
    next_id = np.empty(num_colors, np.int64)
    total = 0
    for c in range(num_colors):
        next_id[c] = total
        total += count_per_label[c]

    # Number each component (stored at its root pixel) when its first block is
    # reached, then give every pixel its root's id.
    region_color = np.empty(total, np.int32)
    out_ids[:] = -1
    for by in range(0, height, 2):
        for bx in range(0, width, 2):
            for y in range(by, min(by + 2, height)):
                for x in range(bx, min(bx + 2, width)):
                    p = y * width + x
                    c = labels[p]
                    if c < 0 or c >= num_colors:
                        continue
                    root = _find(parent, p)
                    if out_ids[root] < 0:
                        out_ids[root] = next_id[c]
                        region_color[next_id[c]] = c
                        next_id[c] += 1

    areas = np.zeros(total, np.int64)
    for p in range(n):
        c = labels[p]
        if c < 0 or c >= num_colors:
            continue
        rid = out_ids[_find(parent, p)]
        out_ids[p] = rid
        areas[rid] += 1
    return region_color, areas


@njit(cache=True, nogil=True)
def _sift_down(heap, i, size):
    item = heap[i]
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and heap[child + 1] < heap[child]:
            child += 1
        if heap[child] >= item:
            break
        heap[i] = heap[child]
        i = child
    heap[i] = item


@njit(cache=True, nogil=True)
def _sift_up(heap, i):
    item = heap[i]
    while i > 0:
        parent = (i - 1) >> 1
        if heap[parent] <= item:
            break
        heap[i] = heap[parent]
        i = parent
    heap[i] = item


@njit(cache=True, nogil=True)
def merge_small_regions(ids, height, width, areas, min_area_px):
    """Merge regions smaller than ``min_area_px`` into a neighbor, in place.

    Repeatedly takes the smallest undersized region (lowest id on ties) and
    relabels it to the neighbor owning the most pixels of its 8-connected
    outer ring (lowest id on ties); a region with no neighbor is left alone.

    Each merge visits only the merged region's own pixels, kept as per-region
    linked lists. A region always merges into one at least as large, so a
    pixel moves at most ~log2(min_area_px) times: O(pixels * log(min_area)).
    """
    num_regions = areas.shape[0]
    if min_area_px <= 0 or num_regions <= 1:
        return
    n = height * width

    head = np.full(num_regions, -1, np.int32)
    tail = np.full(num_regions, -1, np.int32)
    next_pixel = np.full(n, -1, np.int32)
    for p in range(n):
        r = ids[p]
        if r < 0:
            continue
        if head[r] < 0:
            head[r] = p
        else:
            next_pixel[tail[r]] = p
        tail[r] = p

    # Min-heap of (area << 32 | id). Entries go stale when a region grows or
    # is merged away; those are skipped when popped.
    heap = np.empty(2 * num_regions + 1, np.int64)
    size = 0
    for r in range(num_regions):
        if areas[r] < min_area_px:
            heap[size] = (areas[r] << 32) | r
            size += 1
    for i in range(size // 2 - 1, -1, -1):
        _sift_down(heap, i, size)

    ring_stamp = np.full(n, -1, np.int32)
    ring_counts = np.zeros(num_regions, np.int32)
    neighbors = np.empty(num_regions, np.int32)
    active = np.ones(num_regions, np.bool_)
    visit = 0

    while size > 0:
        key = heap[0]
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            _sift_down(heap, 0, size)
        r = key & 0xFFFFFFFF
        if not active[r] or areas[r] != (key >> 32):
            continue

        # Count, per neighbor region, the distinct ring pixels it owns.
        visit += 1
        num_neighbors = 0
        p = head[r]
        while p >= 0:
            y = p // width
            x = p - y * width
            for yy in range(max(y - 1, 0), min(y + 2, height)):
                for xx in range(max(x - 1, 0), min(x + 2, width)):
                    q = yy * width + xx
                    s = ids[q]
                    if s >= 0 and s != r and ring_stamp[q] != visit:
                        ring_stamp[q] = visit
                        if ring_counts[s] == 0:
                            neighbors[num_neighbors] = s
                            num_neighbors += 1
                        ring_counts[s] += 1
            p = next_pixel[p]

        if num_neighbors == 0:
            active[r] = False
            continue

        target = -1
        best = -1
        for k in range(num_neighbors):
            s = neighbors[k]
            if ring_counts[s] > best or (ring_counts[s] == best and s < target):
                best = ring_counts[s]
                target = s
            ring_counts[s] = 0

        p = head[r]
        while p >= 0:
            ids[p] = target
            p = next_pixel[p]
        next_pixel[tail[target]] = head[r]
        tail[target] = tail[r]
        head[r] = -1
        tail[r] = -1
        areas[target] += areas[r]
        areas[r] = 0
        active[r] = False
        if areas[target] < min_area_px:
            heap[size] = (areas[target] << 32) | target
            _sift_up(heap, size)
            size += 1


@njit(cache=True, nogil=True)
def region_bounds(ids, height, width, num_regions):
    """Per region id: inclusive bounding box ``(x0, y0, x1, y1)`` and pixel count."""
    bounds = np.empty((num_regions, 4), np.int32)
    bounds[:, 0] = width
    bounds[:, 1] = height
    bounds[:, 2] = -1
    bounds[:, 3] = -1
    areas = np.zeros(num_regions, np.int64)
    for y in range(height):
        row = y * width
        for x in range(width):
            r = ids[row + x]
            if r < 0:
                continue
            if x < bounds[r, 0]:
                bounds[r, 0] = x
            if x > bounds[r, 2]:
                bounds[r, 2] = x
            if y < bounds[r, 1]:
                bounds[r, 1] = y
            bounds[r, 3] = y
            areas[r] += 1
    return bounds, areas


@njit(cache=True, nogil=True)
def bilateral_rows(padded, out, y_start, y_stop, width, radius, offsets, space_weights, color_weights):
    """Bilateral-filter rows ``[y_start, y_stop)`` of a BGR image into ``out``.

    ``padded`` is the image with a ``radius``-pixel border, flattened; ``out``
    is the flattened (unpadded) output. Each output pixel is the average of
    the taps at ``offsets`` (flat byte offsets to a tap's B channel), weighted
    by ``space_weights`` times ``color_weights[|dB| + |dG| + |dR|]`` -- the
    weighting of ``cv2.bilateralFilter`` for 8-bit color images.
    """
    stride = (width + 2 * radius) * 3
    num_taps = offsets.shape[0]
    half = np.float32(0.5)
    for y in range(y_start, y_stop):
        center = (y + radius) * stride + radius * 3
        dst = y * width * 3
        for _x in range(width):
            b0 = np.int32(padded[center])
            g0 = np.int32(padded[center + 1])
            r0 = np.int32(padded[center + 2])
            weight_sum = np.float32(0.0)
            b_sum = np.float32(0.0)
            g_sum = np.float32(0.0)
            r_sum = np.float32(0.0)
            for k in range(num_taps):
                tap = center + offsets[k]
                b = np.int32(padded[tap])
                g = np.int32(padded[tap + 1])
                r = np.int32(padded[tap + 2])
                weight = space_weights[k] * color_weights[abs(b - b0) + abs(g - g0) + abs(r - r0)]
                weight_sum += weight
                b_sum += weight * b
                g_sum += weight * g
                r_sum += weight * r
            out[dst] = np.uint8(b_sum / weight_sum + half)
            out[dst + 1] = np.uint8(g_sum / weight_sum + half)
            out[dst + 2] = np.uint8(r_sum / weight_sum + half)
            center += 3
            dst += 3


def warm_up() -> None:
    """Compile (or load from cache) every kernel using tiny inputs.

    The argument types match real calls exactly, so no recompiling later.
    """
    labels = np.array([0, 0, 1, 0, 1, 1, 2, 2, 1], dtype=np.int32)
    ids = np.empty(9, dtype=np.int32)
    _region_color, areas = label_components(labels, 3, 3, 3, ids)
    merge_small_regions(ids, 3, 3, areas, 3)
    region_bounds(ids, 3, 3, int(ids.max()) + 1)

    padded = np.zeros(5 * 5 * 3, dtype=np.uint8)
    out = np.empty(3 * 3 * 3, dtype=np.uint8)
    offsets = np.array([0], dtype=np.int64)
    weights = np.ones(1, dtype=np.float32)
    bilateral_rows(padded, out, 0, 3, 3, 1, offsets, weights, np.ones(766, dtype=np.float32))
