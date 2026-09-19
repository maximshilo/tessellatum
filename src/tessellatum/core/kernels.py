"""Numba-compiled inner loops for the pipeline's hot spots.

Pixel-level work NumPy can't vectorize -- union-find labeling, the sequential
small-region merge, the same-color union that follows it, the greedy coloring
that groups regions for the width measurement, the walk that turns the
boundaries between regions into one path each, the search for the nearest
core a thin part can reach without crossing line art's ink, and the bilateral
filter's per-pixel weighting -- runs here as compiled code. Arrays are passed
flattened (row-major) with explicit ``height``/``width``.

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
    # Always keep the smaller index as root: a pixel component's root is then
    # its first pixel in raster order, and a group of regions keeps its
    # lowest id.
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
def merge_same_color_neighbors(ids, height, width, region_color, areas):
    """Union 8-adjacent regions of the same color into one region, in place.

    Components start out one color each, so only ``merge_small_regions`` can
    leave two neighbors sharing a color: a small region merges into whichever
    neighbor shares the most boundary, whatever its color, and the grown
    region can end up touching another region of its own color. Those pairs
    would be drawn with a line between them that no painter should see.

    A group of regions connected by such contacts becomes one region, keeping
    the group's lowest id (so its color and its place in the numbering are
    unchanged). ``areas`` is updated to match: the surviving id gains the
    group's pixels, the others drop to 0. One pass suffices -- the union-find
    closes chains of contacts transitively.
    """
    num_regions = areas.shape[0]
    if num_regions <= 1:
        return
    parent = np.empty(num_regions, np.int32)
    for r in range(num_regions):
        parent[r] = r

    # Each pixel looks right and along the row below, so every 8-adjacent
    # pair of pixels is visited exactly once.
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            r = ids[p]
            if r < 0:
                continue
            c = region_color[r]
            if x + 1 < width:
                s = ids[p + 1]
                if s >= 0 and s != r and region_color[s] == c:
                    _union(parent, r, s)
            if y + 1 < height:
                below = row + width
                for xx in range(max(x - 1, 0), min(x + 2, width)):
                    s = ids[below + xx]
                    if s >= 0 and s != r and region_color[s] == c:
                        _union(parent, r, s)

    # _union keeps the lower id as the root, so every group already survives
    # under its lowest id.
    changed = False
    for r in range(num_regions):
        root = _find(parent, r)
        if root != r:
            areas[root] += areas[r]
            areas[r] = 0
            changed = True
    if not changed:
        return
    for p in range(height * width):
        r = ids[p]
        if r >= 0:
            ids[p] = _find(parent, r)


@njit(cache=True, nogil=True)
def edge_adjacency_classes(ids, height, width, num_regions, out_classes):
    """Split the regions into classes in which no two share a pixel edge, in place.

    Writes each pixel's class to ``out_classes`` (-1 where it is in no
    region) and returns the number of classes used. A greedy coloring of the
    region adjacency graph, most-connected region first, which takes 5-6
    classes on real pages.

    The region stage measures how wide a region is by how far its pixels lie
    from the nearest pixel of another region. One distance transform per
    class answers that for every region in the class at once: the nearest
    pixel of another region always shares an edge with this one (a step from
    it towards the region's inside lands there), so it is never in the same
    class.
    """
    n = height * width
    degree = np.zeros(num_regions, np.int64)
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            r = ids[p]
            if r < 0:
                continue
            if x + 1 < width:
                s = ids[p + 1]
                if s >= 0 and s != r:
                    degree[r] += 1
                    degree[s] += 1
            if y + 1 < height:
                s = ids[p + width]
                if s >= 0 and s != r:
                    degree[r] += 1
                    degree[s] += 1

    # Adjacency as one flat array, a region's neighbors at
    # start[r]:start[r + 1]. A pair is stored once per shared pixel edge;
    # repeats only make the scan below a little longer.
    start = np.empty(num_regions + 1, np.int64)
    total = 0
    for r in range(num_regions):
        start[r] = total
        total += degree[r]
    start[num_regions] = total
    fill = start[:num_regions].copy()
    neighbors = np.empty(total, np.int32)
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            r = ids[p]
            if r < 0:
                continue
            if x + 1 < width:
                s = ids[p + 1]
                if s >= 0 and s != r:
                    neighbors[fill[r]] = s
                    fill[r] += 1
                    neighbors[fill[s]] = r
                    fill[s] += 1
            if y + 1 < height:
                s = ids[p + width]
                if s >= 0 and s != r:
                    neighbors[fill[r]] = s
                    fill[r] += 1
                    neighbors[fill[s]] = r
                    fill[s] += 1

    order = np.argsort(-degree)  # most neighbors first: the usual greedy coloring order
    region_class = np.full(num_regions, -1, np.int32)
    seen = np.zeros(num_regions + 1, np.int64)  # seen[c] == stamp: class c is taken by a neighbor
    num_classes = 0
    for k in range(num_regions):
        r = order[k]
        stamp = k + 1
        for i in range(start[r], start[r + 1]):
            c = region_class[neighbors[i]]
            if c >= 0:
                seen[c] = stamp
        c = 0
        while seen[c] == stamp:
            c += 1
        region_class[r] = c
        if c >= num_classes:
            num_classes = c + 1

    for p in range(n):
        r = ids[p]
        out_classes[p] = region_class[r] if r >= 0 else -1
    return num_classes


@njit(cache=True, nogil=True)
def _crack_edge(right, down, stride, corner, direction):
    """Is there a crack edge leaving ``corner`` in ``direction`` (0 right, 1 down, 2 left, 3 up)?

    The edge left of a corner is the one right of its left neighbor, and the
    edge above it the one below the corner above; both are absent in the
    column and row where those neighbors would fall off the grid, which
    ``right`` and ``down`` mark as absent anyway (see
    ``boundaries.crack_edges``).
    """
    if direction == 0:
        return right[corner]
    if direction == 1:
        return down[corner]
    if direction == 2:
        return corner >= 1 and right[corner - 1]
    return corner >= stride and down[corner - stride]


@njit(cache=True, nogil=True)
def _use_crack_edge(used, stride, corner, direction):
    """Mark the crack edge leaving ``corner`` in ``direction`` as drawn."""
    if direction == 0:
        used[corner] |= 1
    elif direction == 1:
        used[corner] |= 2
    elif direction == 2:
        used[corner - 1] |= 1
    else:
        used[corner - stride] |= 2


@njit(cache=True, nogil=True)
def _crack_edge_used(used, stride, corner, direction):
    if direction == 0:
        return (used[corner] & 1) != 0
    if direction == 1:
        return (used[corner] & 2) != 0
    if direction == 2:
        return corner >= 1 and (used[corner - 1] & 1) != 0
    return corner >= stride and (used[corner - stride] & 2) != 0


@njit(cache=True, nogil=True)
def trace_boundary_paths(right, down, degree, stride, num_edges):
    """Walk the crack graph into one path per boundary between two regions.

    The graph lives on the grid of pixel corners: corner ``i * stride + j``
    is the point ``(x, y) = (j - 0.5, i - 0.5)``, and ``right[c]`` / ``down[c]``
    say whether a crack edge joins it to the corner on its right / below it
    (see ``boundaries.crack_edges``). ``degree[c]`` counts the crack edges
    around a corner, which is 0, 2, 3 or 4: a corner with three or more is a
    junction, where boundaries meet.

    Every other corner has exactly two edges, which separate the same two
    regions, so following them from junction to junction gives the whole
    boundary between one pair of regions as a single path. Edges left over
    belong to boundaries with no junction at all -- a region lying inside
    another, or the page edge around a page of one region -- and come back as
    closed paths that repeat their first corner.

    Returns ``(corners, starts)``: every path's corners in order, and where
    each path begins in that array, with its length last. Each of the
    ``num_edges`` crack edges is walked exactly once, so each boundary gets
    exactly one line.
    """
    used = np.zeros(right.size, np.uint8)
    # A path of k edges has k + 1 corners, and there are at most num_edges paths.
    corners = np.empty(2 * num_edges + 2, np.int32)
    starts = np.empty(num_edges + 2, np.int32)
    on_graph = np.flatnonzero(degree > 0)
    n = 0
    paths = 0

    # Junctions first, so that the paths between them are traced end to end;
    # whatever is left over after that is a boundary that meets no junction.
    for junctions_first in range(2):
        for k in range(on_graph.size):
            corner = on_graph[k]
            if (degree[corner] >= 3) != (junctions_first == 0):
                continue
            for direction in range(4):
                if not _crack_edge(right, down, stride, corner, direction):
                    continue
                if _crack_edge_used(used, stride, corner, direction):
                    continue
                starts[paths] = n
                paths += 1
                c, d = corner, direction
                corners[n] = c
                n += 1
                while True:
                    _use_crack_edge(used, stride, c, d)
                    if d == 0:
                        c += 1
                    elif d == 1:
                        c += stride
                    elif d == 2:
                        c -= 1
                    else:
                        c -= stride
                    corners[n] = c
                    n += 1
                    if degree[c] != 2:
                        break  # a junction: the boundary to the next one is another path
                    back = (d + 2) & 3
                    nd = -1
                    for e in range(4):
                        if e != back and _crack_edge(right, down, stride, c, e):
                            nd = e
                            break
                    if nd < 0 or _crack_edge_used(used, stride, c, nd):
                        break  # back where this path started: a closed boundary
                    d = nd

    starts[paths] = n
    return corners[:n], starts[: paths + 1]


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


@njit(cache=True, nogil=True)
def nearest_seed_within(seeds, passable, height, width):
    """For every pixel, the label of the seed nearest to it along a path through ``passable`` pixels.

    ``seeds`` holds a label (>= 0) at each seed pixel and -1 elsewhere; a seed
    on a pixel that isn't passable is ignored. A path steps between
    8-neighbors, a straight step counting 5 and a diagonal one 7: a chamfer
    distance within a few percent of the straight-line one, measured around
    whatever isn't passable rather than through it (Dijkstra's algorithm).
    Ties go to the wave that reaches a pixel first, in order of distance and
    then raster position, so the result is deterministic. Returns -1 where no
    seed can be reached, and on pixels that aren't passable.
    """
    n = height * width
    unreached = np.int64(1) << 60
    dist = np.full(n, unreached, np.int64)
    out = np.full(n, -1, np.int32)
    # Min-heap of (distance << 32 | pixel); a pixel is pushed again whenever it gets nearer, and stale entries are
    # skipped when popped. It grows as needed.
    heap = np.empty(max(64, n), np.int64)
    size = 0
    for p in range(n):
        if seeds[p] >= 0 and passable[p]:
            dist[p] = 0
            out[p] = seeds[p]
            heap[size] = p
            size += 1
    for i in range(size // 2 - 1, -1, -1):
        _sift_down(heap, i, size)

    while size > 0:
        key = heap[0]
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            _sift_down(heap, 0, size)
        d = key >> 32
        p = key & 0xFFFFFFFF
        if d != dist[p]:
            continue
        y = p // width
        x = p - y * width
        for dy in range(-1, 2):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for dx in range(-1, 2):
                xx = x + dx
                if (dy == 0 and dx == 0) or xx < 0 or xx >= width:
                    continue
                q = yy * width + xx
                if not passable[q]:
                    continue
                nd = d + (7 if dy != 0 and dx != 0 else 5)
                if nd < dist[q]:
                    dist[q] = nd
                    out[q] = out[p]
                    if size == heap.size:
                        grown = np.empty(2 * heap.size, np.int64)
                        grown[:size] = heap[:size]
                        heap = grown
                    heap[size] = (nd << 32) | q
                    _sift_up(heap, size)
                    size += 1
    return out


def warm_up() -> None:
    """Compile (or load from cache) every kernel using tiny inputs.

    The argument types match real calls exactly, so no recompiling later.
    """
    labels = np.array([0, 0, 1, 0, 1, 1, 2, 2, 1], dtype=np.int32)
    ids = np.empty(9, dtype=np.int32)
    region_color, areas = label_components(labels, 3, 3, 3, ids)
    merge_small_regions(ids, 3, 3, areas, 3)
    merge_same_color_neighbors(ids, 3, 3, region_color, areas)
    edge_adjacency_classes(ids, 3, 3, int(ids.max()) + 1, np.empty(9, dtype=np.int8))
    region_bounds(ids, 3, 3, int(ids.max()) + 1)
    nearest_seed_within(labels - 1, labels >= 0, 3, 3)

    from tessellatum.core.boundaries import crack_edges  # imported here: boundaries imports this module

    right, down, degree, num_edges = crack_edges(ids.reshape(3, 3))
    trace_boundary_paths(right, down, degree, 4, num_edges)

    padded = np.zeros(5 * 5 * 3, dtype=np.uint8)
    out = np.empty(3 * 3 * 3, dtype=np.uint8)
    offsets = np.array([0], dtype=np.int64)
    weights = np.ones(1, dtype=np.float32)
    bilateral_rows(padded, out, 0, 3, 3, 1, offsets, weights, np.ones(766, dtype=np.float32))
