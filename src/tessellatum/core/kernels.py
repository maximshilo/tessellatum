"""Numba-compiled inner loops for the pipeline's hot spots.

Pixel-level work NumPy can't vectorize -- union-find labeling, the sequential
small-region merge, the same-color union that follows it, the greedy coloring
that groups regions for the width measurement, the walk that turns the
boundaries between regions into one path each, the search for the nearest
core a thin part can reach without crossing line art's ink, and the bilateral
filter's per-pixel weighting -- runs here as compiled code. So do line art's
region steps (see ``regions.look_through_hatching`` to ``regions.split_areas``):
their searches along paths, and the passes over the page that compare each
pixel with its neighbors, which NumPy would make one whole-page array per
neighbor for. Those give exactly what the NumPy code they replace gave
(``tests/reference_line_art.py`` keeps it). Arrays are passed flattened
(row-major) with explicit ``height``/``width``.

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

    # A parent always lies before its child (see _union), so one pass in raster order points every pixel at its root.
    count_per_label = np.zeros(num_colors, np.int64)
    for p in range(n):
        up = parent[p]
        if up != p:
            parent[p] = parent[up]
        else:
            c = labels[p]
            if 0 <= c < num_colors:
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
                    root = parent[p]
                    if out_ids[root] < 0:
                        out_ids[root] = next_id[c]
                        region_color[next_id[c]] = c
                        next_id[c] += 1

    areas = np.zeros(total, np.int64)
    for p in range(n):
        c = labels[p]
        if c < 0 or c >= num_colors:
            continue
        rid = out_ids[parent[p]]
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
def merge_small_regions(ids, height, width, areas, min_area_px, diagonals):
    """Merge regions smaller than ``min_area_px`` into a neighbor, in place.

    Repeatedly takes the smallest undersized region (lowest id on ties) and
    relabels it to the neighbor owning the most pixels of its 8-connected
    outer ring (lowest id on ties); a region with no neighbor is left alone.
    Without ``diagonals`` the ring is 4-connected: a region touching another
    only at a corner is not its neighbor.

    Each merge visits only the merged region's own pixels, kept as per-region
    linked lists. A region always merges into one at least as large, so a
    pixel moves at most ~log2(min_area_px) times: O(pixels * log(min_area)).
    Only a region starting below ``min_area_px`` is ever merged away (a
    region only grows), so only those keep a list: a region at least that
    large takes the pixels merged into it without one.
    """
    num_regions = areas.shape[0]
    if min_area_px <= 0 or num_regions <= 1:
        return
    n = height * width

    head = np.full(num_regions, -1, np.int32)
    tail = np.full(num_regions, -1, np.int32)
    listed = areas < min_area_px
    next_pixel = np.empty(n, np.int32)  # written for every listed pixel before it is read
    for p in range(n):
        r = ids[p]
        if r < 0 or not listed[r]:
            continue
        next_pixel[p] = -1
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
                    if not diagonals and yy != y and xx != x:
                        continue
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
        if listed[target]:  # one that isn't is too large ever to merge away: it needs no list
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
def merge_same_color_neighbors(ids, height, width, region_color, areas, diagonals):
    """Union 8-adjacent regions of the same color into one region, in place.

    Without ``diagonals``, only 4-adjacent ones: two regions touching at a
    corner stay apart.

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
                    if not diagonals and xx != x:
                        continue
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
    whatever isn't passable rather than through it. Returns -1 where no seed
    can be reached, and on pixels that aren't passable.

    The distances are propagated in raster passes, forward and back, until a
    pass changes nothing: each pass carries them along every path that runs
    its way, and a path that doubles back takes another pair. A pixel keeps
    the first label that reaches it at its shortest distance, so the result is
    deterministic.
    """
    return _nearest_seed(seeds, passable, height, width, False)


@njit(cache=True, nogil=True)
def nearest_seed_in_groups(seeds, groups, height, width):
    """``nearest_seed_within``, with a path stepping only between pixels of one group.

    ``groups`` holds each pixel's group (>= 0), or -1 where it is passable by
    none. Each group is searched as if it were the only passable part of the
    page: what ``nearest_seed_within`` gives for it, run on it alone, is what
    this gives for it, since a pass over one group never reads another.
    """
    return _nearest_seed(seeds, groups, height, width, -1)


@njit(cache=True, nogil=True)
def _nearest_seed(seeds, groups, height, width, none):
    """The raster passes of ``nearest_seed_within``, skipping only what they could not change.

    A pixel is passable where ``groups`` isn't ``none``, and a step joins two
    pixels of one group. Seeds and pixels no path enters never change, so a
    pass visits only the others, in raster order: the order the passes over
    the whole page visit them in. A forward pass reads a pixel's own row and
    the row above, so a row where neither has changed since the forward pass
    last went over it keeps what it has; a pass back reads the row below
    instead, and skips a row the same way. Neither skip changes what any pass
    does, so the result is exactly that of passing over every pixel.
    """
    n = height * width
    unreached = np.int64(1) << 60
    dist = np.full(n, unreached, np.int64)
    out = np.full(n, -1, np.int32)
    todo = np.empty(n, np.int32)  # the pixels a pass can change, row by row: those of row y at row_start[y]:row_start[y + 1]
    row_start = np.empty(height + 1, np.int64)
    k = 0
    for y in range(height):
        row_start[y] = k
        for p in range(y * width, (y + 1) * width):
            if groups[p] == none:
                continue
            if seeds[p] >= 0:
                dist[p] = 0
                out[p] = seeds[p]
            else:
                todo[k] = p
                k += 1
    row_start[height] = k

    # Pass numbers: the last pass that changed a pixel of each row, and the last forward and back pass over it.
    changed_in = np.zeros(height, np.int64)
    forward_over = np.full(height, -1, np.int64)
    back_over = np.full(height, -1, np.int64)
    step = 0
    changed = True
    while changed:
        changed = False
        step += 1
        for y in range(height):  # forward: from the pixels above and to the left
            if row_start[y] == row_start[y + 1]:
                continue
            if changed_in[y] <= forward_over[y] and (y == 0 or changed_in[y - 1] <= forward_over[y]):
                continue
            row = y * width
            row_changed = False
            for i in range(row_start[y], row_start[y + 1]):
                p = todo[i]
                x = p - row
                g = groups[p]
                best = dist[p]
                label = out[p]
                if y > 0:
                    q = p - width
                    if x > 0 and groups[q - 1] == g and dist[q - 1] + 7 < best:
                        best = dist[q - 1] + 7
                        label = out[q - 1]
                    if groups[q] == g and dist[q] + 5 < best:
                        best = dist[q] + 5
                        label = out[q]
                    if x + 1 < width and groups[q + 1] == g and dist[q + 1] + 7 < best:
                        best = dist[q + 1] + 7
                        label = out[q + 1]
                if x > 0 and groups[p - 1] == g and dist[p - 1] + 5 < best:
                    best = dist[p - 1] + 5
                    label = out[p - 1]
                if best < dist[p]:
                    dist[p] = best
                    out[p] = label
                    row_changed = True
            forward_over[y] = step
            if row_changed:
                changed_in[y] = step
                changed = True
        step += 1
        for y in range(height - 1, -1, -1):  # back: from the pixels below and to the right
            if row_start[y] == row_start[y + 1]:
                continue
            if changed_in[y] <= back_over[y] and (y == height - 1 or changed_in[y + 1] <= back_over[y]):
                continue
            row = y * width
            row_changed = False
            for i in range(row_start[y + 1] - 1, row_start[y] - 1, -1):
                p = todo[i]
                x = p - row
                g = groups[p]
                best = dist[p]
                label = out[p]
                if y + 1 < height:
                    q = p + width
                    if x + 1 < width and groups[q + 1] == g and dist[q + 1] + 7 < best:
                        best = dist[q + 1] + 7
                        label = out[q + 1]
                    if groups[q] == g and dist[q] + 5 < best:
                        best = dist[q] + 5
                        label = out[q]
                    if x > 0 and groups[q - 1] == g and dist[q - 1] + 7 < best:
                        best = dist[q - 1] + 7
                        label = out[q - 1]
                if x + 1 < width and groups[p + 1] == g and dist[p + 1] + 5 < best:
                    best = dist[p + 1] + 5
                    label = out[p + 1]
                if best < dist[p]:
                    dist[p] = best
                    out[p] = label
                    row_changed = True
            back_over[y] = step
            if row_changed:
                changed_in[y] = step
                changed = True
    return out


@njit(cache=True, nogil=True)
def region_color_sums(ids, pixels, asked, own):
    """Per region id ``asked`` about, the sums of its pixels' colors and their count: of its pixels in ``own`` too.

    ``ids`` is the flat region map (-1 for no region), ``pixels`` the flat
    Nx3 image, ``asked`` a bool per region id and ``own`` a bool per pixel.
    Returns ``(sums, counts, own_sums, own_counts)``, float64, a row per
    region id; the rows of the regions not asked about are 0.
    """
    count = asked.shape[0]
    sums = np.zeros((count, 3), np.float64)
    counts = np.zeros(count, np.float64)
    own_sums = np.zeros((count, 3), np.float64)
    own_counts = np.zeros(count, np.float64)
    for p in range(ids.shape[0]):
        r = ids[p]
        if r < 0 or not asked[r]:
            continue
        counts[r] += 1
        for c in range(3):
            sums[r, c] += pixels[p, c]
        if own[p]:
            own_counts[r] += 1
            for c in range(3):
                own_sums[r, c] += pixels[p, c]
    return sums, counts, own_sums, own_counts


@njit(cache=True, nogil=True)
def white_pieces(ids, height, width, out_piece):
    """4-connected runs of one id in ``ids`` (-1 for none): each pixel's piece to ``out_piece``, -1 where it has no id.

    Pieces are numbered in raster order of their first pixel, which is how
    ``cv2.connectedComponents`` numbers the 4-connected parts of one mask: the
    pieces of one id come in the order that numbering gives them in any box
    holding them. Returns the id of each piece.
    """
    n = height * width
    parent = np.empty(n, np.int32)
    for p in range(n):
        parent[p] = p
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            r = ids[p]
            if r < 0:
                continue
            if x > 0 and ids[p - 1] == r:
                _union(parent, p, p - 1)
            if y > 0 and ids[p - width] == r:
                _union(parent, p, p - width)
    # _union keeps the lower index as the root, so a piece's root is its first pixel in raster order, and every parent
    # lies before its child, in the same piece: in raster order, a pixel's parent is numbered already.
    piece_region = np.empty(n, np.int32)
    count = 0
    for p in range(n):
        if ids[p] < 0:
            out_piece[p] = -1
            continue
        up = parent[p]
        if up == p:
            out_piece[p] = count
            piece_region[count] = ids[p]
            count += 1
        else:
            out_piece[p] = out_piece[up]
    return piece_region[:count].copy()


@njit(cache=True, nogil=True)
def nearest_core_color(nearest, fits, ids, region_color, out_color):
    """Each pixel's color in ``out_color``: that of the region of its nearest core pixel.

    ``nearest`` comes from a distance transform with a label per pixel over the
    pixels outside the cores (``fits`` False): the label of the nearest core
    pixel, which is its own on a core pixel.
    """
    most = 0
    for p in range(nearest.shape[0]):
        if nearest[p] > most:
            most = nearest[p]
    color_of_label = np.zeros(most + 1, np.int32)
    for p in range(nearest.shape[0]):
        if fits[p]:
            color_of_label[nearest[p]] = region_color[ids[p]]
    for p in range(nearest.shape[0]):
        out_color[p] = color_of_label[nearest[p]]


@njit(cache=True, nogil=True)
def across_to_nearest_core(nearest, fits, compartment, outside, out_across):
    """Mark in ``out_across`` the pixels not ``outside`` whose nearest core pixel lies in another ``compartment``.

    ``nearest`` is as in ``nearest_core_color``. Returns, per compartment,
    whether it holds such a pixel.
    """
    most = 0
    most_compartment = 0
    for p in range(nearest.shape[0]):
        if nearest[p] > most:
            most = nearest[p]
        if compartment[p] > most_compartment:
            most_compartment = compartment[p]
    compartment_of_label = np.zeros(most + 1, np.int32)
    for p in range(nearest.shape[0]):
        if fits[p]:
            compartment_of_label[nearest[p]] = compartment[p]
    wanted = np.zeros(most_compartment + 1, np.bool_)
    for p in range(nearest.shape[0]):
        out_across[p] = not outside[p] and compartment_of_label[nearest[p]] != compartment[p]
        if out_across[p]:
            wanted[compartment[p]] = True
    return wanted


@njit(cache=True, nogil=True)
def white_of(ids, printed, asked, out_white):
    """Each pixel's region id in ``out_white`` where it is unprinted and ``asked`` marks its region, else -1."""
    for p in range(ids.shape[0]):
        r = ids[p]
        out_white[p] = r if r >= 0 and not printed[p] and asked[r] else -1


@njit(cache=True, nogil=True)
def split_seeds(ids, piece, paintable, splitting, out_groups, out_seeds):
    """The claim search's input for the regions ``splitting`` marks (see ``regions._split_all``).

    ``out_groups``: each pixel's region where it is splitting, else -1.
    ``out_seeds``: its piece where that piece is ``paintable``, else -1.
    """
    for p in range(ids.shape[0]):
        r = ids[p]
        if r >= 0 and splitting[r]:
            out_groups[p] = r
            k = piece[p]
            out_seeds[p] = k if k >= 0 and paintable[k] else -1
        else:
            out_groups[p] = -1
            out_seeds[p] = -1


@njit(cache=True, nogil=True)
def apply_split(ids, claim, rank, printed, groups, base, height, width, out_corners):
    """Split the regions ``groups`` marks in place in ``ids``, one region per area, with walls where two areas meet.

    ``claim`` holds, where a region splits, the area claiming each pixel (a
    piece, see ``white_pieces``) or -1, and becomes each pixel's rank there:
    ``rank`` numbers a region's areas 0, 1, ... . The first keeps the
    region's id, and area k > 0 of region r gets ``base[r] + k - 1``. Of two
    8-adjacent pixels of one region claimed by different areas, one is a wall
    (see ``claim_walls``) and leaves every region (-1); ``out_corners`` marks
    the unprinted walls, which the page prints.
    """
    for p in range(ids.shape[0]):
        c = claim[p]
        if c >= 0:
            claim[p] = rank[c]
    wall = np.empty(height * width, np.bool_)
    claim_walls(claim, printed, groups, height, width, wall)
    for p in range(ids.shape[0]):
        out_corners[p] = wall[p] and not printed[p]
        if wall[p]:
            ids[p] = -1
        elif claim[p] > 0:
            ids[p] = base[groups[p]] + claim[p] - 1


@njit(cache=True, nogil=True)
def painted_labels(ids, colors, num_colors, out_labels):
    """Each pixel's color in ``out_labels``: its region's in ``colors``, ``num_colors`` where it is in no region."""
    for p in range(ids.shape[0]):
        r = ids[p]
        out_labels[p] = colors[r] if r >= 0 else num_colors


@njit(cache=True, nogil=True)
def regions_joined(old, new, count):
    """For each of ``count`` regions of ``new``, whether it holds pixels of two regions of ``old`` or more."""
    first = np.full(count, -1, np.int32)  # the first region of old seen in each region of new
    joined = np.zeros(count, np.bool_)
    for p in range(old.shape[0]):
        o = old[p]
        r = new[p]
        if o < 0 or r < 0:
            continue
        if first[r] < 0:
            first[r] = o
        elif first[r] != o:
            joined[r] = True
    return joined


@njit(cache=True, nogil=True)
def follow_white(ids, white, merged, count, out_ids):
    """Every region of ``ids`` where its white went in ``merged``: each pixel's new id to ``out_ids``.

    ``white`` is ``ids`` less the printed pixels (-1), ``merged`` the same
    after merging, in which every pixel of a region moved together. A region
    with no white stays itself. Returns, per region, where it went.
    """
    target = np.arange(count).astype(np.int32)
    for p in range(ids.shape[0]):
        if white[p] >= 0:
            target[white[p]] = merged[p]
    for p in range(ids.shape[0]):
        r = ids[p]
        out_ids[p] = target[r] if r >= 0 else -1
    return target


@njit(cache=True, nogil=True)
def seams_round(ids, printed, which, height, width, out_ids, out_taken):
    """``regions._seams_round``: keep the regions ``which`` marks from touching another, taking a pixel of the two away.

    Of two 8-adjacent pixels of different regions, one of them a region
    ``which`` marks, one leaves its region (-1 in ``out_ids``): the printed
    one if only one is, else the one with the higher id. Two white pixels
    across a corner count only where neither pixel beside them joins the two
    regions edge to edge, and the one that leaves is marked in ``out_taken``.
    Two white pixels sharing an edge are left as they are.

    Only a pixel of a marked region, or beside one, can leave, so only the
    boxes round the marked regions, a pixel wider, are looked at.
    """
    num_regions = which.shape[0]
    box = np.empty((num_regions, 4), np.int64)  # x0, y0, x1, y1 of each marked region
    box[:, 0] = width
    box[:, 1] = height
    box[:, 2] = -1
    box[:, 3] = -1
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            r = ids[p]
            out_ids[p] = r
            out_taken[p] = False
            if r >= 0 and which[r]:
                box[r, 0] = min(box[r, 0], x)
                box[r, 1] = min(box[r, 1], y)
                box[r, 2] = max(box[r, 2], x)
                box[r, 3] = max(box[r, 3], y)
    for k in range(num_regions):
        if box[k, 2] < 0:
            continue
        for y in range(max(box[k, 1] - 1, 0), min(box[k, 3] + 2, height)):
            for x in range(max(box[k, 0] - 1, 0), min(box[k, 2] + 2, width)):
                _seam_pixel(ids, printed, which, height, width, y, x, out_ids, out_taken)


@njit(cache=True, nogil=True)
def _seam_pixel(ids, printed, which, height, width, y, x, out_ids, out_taken):
    """``seams_round`` for the pixel at ``(x, y)``: whether it leaves its region, from ``ids`` alone."""
    row = y * width
    p = row + x
    r = ids[p]
    if r < 0:
        return
    marked = which[r]
    drop = False
    for dy in range(-1, 2):
        yy = y + dy
        if yy < 0 or yy >= height:
            continue
        for dx in range(-1, 2):
            xx = x + dx
            if (dy == 0 and dx == 0) or xx < 0 or xx >= width:
                continue
            q = yy * width + xx
            o = ids[q]
            if o < 0 or o == r or not (marked or which[o]):
                continue
            if dy != 0 and dx != 0:
                a = yy * width + x  # the two pixels beside both, sharing an edge with each
                b = row + xx
                edge_to_edge = (not printed[a] and (ids[a] == r or ids[a] == o)) or (
                    not printed[b] and (ids[b] == r or ids[b] == o)
                )
                if printed[p]:
                    drop = drop or not printed[q] or o < r
                elif not printed[q] and not edge_to_edge and o < r:
                    drop = True
            elif printed[p] and (not printed[q] or o < r):
                drop = True
    if drop:
        out_ids[p] = -1
        out_taken[p] = not printed[p]


@njit(cache=True, nogil=True)
def region_census(ids, printed, count, height, width):
    """Per region id below ``count``: its pixels, its unprinted pixels, and whether another region touches it (8-adjacent)."""
    areas = np.zeros(count, np.int64)
    unprinted = np.zeros(count, np.int64)
    touched = np.zeros(count, np.bool_)
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            r = ids[p]
            if r < 0:
                continue
            areas[r] += 1
            if not printed[p]:
                unprinted[r] += 1
            # Each pair once: the pixel to the right and the three below.
            if x + 1 < width:
                s = ids[p + 1]
                if s >= 0 and s != r:
                    touched[r] = True
                    touched[s] = True
            if y + 1 < height:
                for xx in range(max(x - 1, 0), min(x + 2, width)):
                    s = ids[row + width + xx]
                    if s >= 0 and s != r:
                        touched[r] = True
                        touched[s] = True
    return areas, unprinted, touched


@njit(cache=True, nogil=True)
def unreached_by_brush(ids, to_brush, radius, out_unreached):
    """``out_unreached``: the pixels in a region farther than ``radius`` from where a brush fits (``to_brush``, float32).

    Squared distances are whole numbers, which the float32 roots only
    approximate: they are rounded back and compared, as the benchmark does.
    """
    limit = radius * radius
    for p in range(ids.shape[0]):
        d = np.float64(to_brush[p])
        out_unreached[p] = ids[p] >= 0 and np.rint(d * d) > limit


@njit(cache=True, nogil=True)
def pocket_contacts(pocket, ids, walls, count, height, width):
    """For each pocket (``pocket`` > 0, numbered below ``count``), its contacts and how many of them are ``walls``.

    A contact is a pixel 8-adjacent to one of the pocket's that is neither in
    the pocket nor in that pixel's region; off the page is a contact too, and
    no wall. Each pair of pixels counts once.
    """
    contacts = np.zeros(count, np.int64)
    on_walls = np.zeros(count, np.int64)
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            k = pocket[p]
            if k == 0:
                continue
            r = ids[p]
            for dy in range(-1, 2):
                yy = y + dy
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    xx = x + dx
                    if yy < 0 or yy >= height or xx < 0 or xx >= width:
                        contacts[k] += 1
                        continue
                    q = yy * width + xx
                    if pocket[q] != k and ids[q] != r:
                        contacts[k] += 1
                        if walls[q]:
                            on_walls[k] += 1
    return contacts, on_walls


@njit(cache=True, nogil=True)
def thin_ink_to_nearest(ids, thin, nearest, distance, max_width, height, width, out_ids, out_joined):
    """Give each ``thin`` ink pixel the region nearest it, within ``max_width``, and put seams where two regions meet.

    ``nearest`` and ``distance`` come from a distance transform with a label
    per pixel over the pixels in no region (``ids`` < 0): the label of the
    nearest pixel in a region, and how far it is. ``out_joined`` marks the
    pixels given a region; ``out_ids`` is ``ids`` with them in it, less every
    joined pixel 8-adjacent to another region's pixel that was a region's own
    already, or joined too and of a lower id: one pass leaves no two regions
    touching where either pixel was joined.
    """
    n = height * width
    most = 0
    for p in range(n):
        if nearest[p] > most:
            most = nearest[p]
    region_of_label = np.full(most + 1, -1, np.int32)
    for p in range(n):
        if ids[p] >= 0:
            region_of_label[nearest[p]] = ids[p]
    for p in range(n):
        out_joined[p] = False
        out_ids[p] = ids[p]
        if thin[p]:
            r = region_of_label[nearest[p]]
            if r >= 0 and distance[p] <= max_width:
                out_joined[p] = True
                out_ids[p] = r
    drop = np.zeros(n, np.bool_)  # every pixel is tested against the joined map before any leaves it
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            if not out_joined[p]:
                continue
            r = out_ids[p]
            for yy in range(max(y - 1, 0), min(y + 2, height)):
                for xx in range(max(x - 1, 0), min(x + 2, width)):
                    q = yy * width + xx
                    o = out_ids[q]
                    if o >= 0 and o != r and (not out_joined[q] or o < r):
                        drop[p] = True
    for p in range(n):
        if drop[p]:
            out_ids[p] = -1


@njit(cache=True, nogil=True)
def keep_anchored(ids, pieces, own, out_ids):
    """``out_ids``: ``ids`` less each piece (``pieces``, 8-connected runs of one region) holding none of its ``own`` pixels."""
    most = -1
    for p in range(ids.shape[0]):
        if pieces[p] > most:
            most = pieces[p]
    anchored = np.zeros(most + 1, np.bool_)
    for p in range(ids.shape[0]):
        if pieces[p] >= 0 and own[p]:
            anchored[pieces[p]] = True
    for p in range(ids.shape[0]):
        k = pieces[p]
        out_ids[p] = -1 if k >= 0 and not anchored[k] else ids[p]


@njit(cache=True, nogil=True)
def paper_rings(piece, ids, paper, count, height, width):
    """For each piece of paper (``piece`` >= 0, below ``count``), the lowest and highest id in its ring.

    A piece's ring is the pixels 8-adjacent to it that aren't paper, off the
    page left out: a region's id, or -1 for ink. With no ring, the lowest is
    the largest int32 and the highest -2.
    """
    lowest = np.full(count, np.iinfo(np.int32).max, np.int64)
    highest = np.full(count, -2, np.int64)
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            k = piece[p]
            if k < 0:
                continue
            for yy in range(max(y - 1, 0), min(y + 2, height)):
                for xx in range(max(x - 1, 0), min(x + 2, width)):
                    q = yy * width + xx
                    if paper[q]:
                        continue
                    o = ids[q]
                    if o < lowest[k]:
                        lowest[k] = o
                    if o > highest[k]:
                        highest[k] = o
    return lowest, highest


@njit(cache=True, nogil=True)
def ink_is_main_neighbor(ids, asked, height, width):
    """For each patch ``asked`` about, whether the ink (-1 in ``ids``) owns at least as much of its outer ring as any one patch.

    A patch's ring is the pixels 8-adjacent to it that aren't its own, each
    counted once, off the page left out, as ``merge_small_regions`` counts it;
    a tie goes to the ink, and a patch with no ink in its ring is False.
    """
    count = asked.shape[0]
    n = height * width
    start = np.zeros(count + 1, np.int64)  # the asked patches' pixels, patch by patch: patch k's at start[k]:start[k + 1]
    for p in range(n):
        r = ids[p]
        if r >= 0 and asked[r]:
            start[r + 1] += 1
    for k in range(count):
        start[k + 1] += start[k]
    pixels = np.empty(start[count], np.int64)
    fill = start[:count].copy()
    for p in range(n):
        r = ids[p]
        if r >= 0 and asked[r]:
            pixels[fill[r]] = p
            fill[r] += 1
    result = np.zeros(count, np.bool_)
    stamp = np.zeros(n, np.int32)
    tally = np.zeros(count, np.int64)
    owners = np.empty(count, np.int32)
    for k in range(count):
        if start[k] == start[k + 1]:
            continue
        by_ink = 0
        num_owners = 0
        for i in range(start[k], start[k + 1]):
            p = pixels[i]
            y = p // width
            x = p - y * width
            for yy in range(max(y - 1, 0), min(y + 2, height)):
                for xx in range(max(x - 1, 0), min(x + 2, width)):
                    q = yy * width + xx
                    o = ids[q]
                    if o == k or stamp[q] == k + 1:
                        continue
                    stamp[q] = k + 1
                    if o < 0:
                        by_ink += 1
                    else:
                        if tally[o] == 0:
                            owners[num_owners] = o
                            num_owners += 1
                        tally[o] += 1
        most = 0
        for j in range(num_owners):
            o = owners[j]
            if tally[o] > most:
                most = tally[o]
            tally[o] = 0
        result[k] = by_ink > 0 and by_ink >= most
    return result


@njit(cache=True, nogil=True)
def claim_walls(claim, ink, groups, height, width, out_wall):
    """Mark, in ``out_wall``, the pixels that keep two claims apart: of every two 8-adjacent pixels of different claims, one.

    ``claim`` is each pixel's claim (>= 0), or -1 for none; ``ink`` a bool per
    pixel. Of two pixels of different claims, the ink one is the wall if only
    one of them is ink, else the one of the higher claim. With ``groups``
    (a group per pixel, as in ``nearest_seed_in_groups``) not empty, only two
    pixels of one group are compared: each group's claims are its own.
    """
    grouped = groups.shape[0] > 0
    for y in range(height):
        row = y * width
        for x in range(width):
            p = row + x
            c = claim[p]
            out_wall[p] = False
            if c < 0:
                continue
            for yy in range(max(y - 1, 0), min(y + 2, height)):
                for xx in range(max(x - 1, 0), min(x + 2, width)):
                    q = yy * width + xx
                    o = claim[q]
                    if o < 0 or o == c or (grouped and groups[q] != groups[p]):
                        continue
                    if ink[p] != ink[q]:
                        if ink[p]:
                            out_wall[p] = True
                    elif c > o:
                        out_wall[p] = True


@njit(cache=True, nogil=True)
def settle_gaps(piece, claim, paintable):
    """Give each piece no brush fits in, in place in ``claim``, wholly to the claim most of its pixels are in.

    ``piece`` is each pixel's piece (-1 for none), ``paintable`` a bool per
    piece, ``claim`` each pixel's claim: a paintable piece (-1 for none). Of a
    piece that isn't paintable, only its claimed pixels count, and only they
    take the claim, which goes to the lowest on a tie.
    """
    num_pieces = paintable.shape[0]
    counts = np.zeros(num_pieces + 1, np.int64)
    for p in range(piece.shape[0]):
        k = piece[p]
        if k >= 0 and not paintable[k] and claim[p] >= 0:
            counts[k + 1] += 1
    for k in range(num_pieces):  # counts[k]: where piece k's claims start in ``claims``
        counts[k + 1] += counts[k]
    claims = np.empty(counts[num_pieces], np.int32)
    fill = counts[:num_pieces].copy()
    for p in range(piece.shape[0]):
        k = piece[p]
        if k >= 0 and not paintable[k] and claim[p] >= 0:
            claims[fill[k]] = claim[p]
            fill[k] += 1
    best = np.full(num_pieces, -1, np.int32)
    tally = np.zeros(num_pieces, np.int64)
    for k in range(num_pieces):
        start, stop = counts[k], counts[k + 1]
        if start == stop:
            continue
        most = 0
        for i in range(start, stop):
            c = claims[i]
            tally[c] += 1
            if tally[c] > most or (tally[c] == most and c < best[k]):
                most = tally[c]
                best[k] = c
        for i in range(start, stop):
            tally[claims[i]] = 0
    for p in range(piece.shape[0]):
        k = piece[p]
        if k >= 0 and not paintable[k] and claim[p] >= 0:
            claim[p] = best[k]


def warm_up() -> None:
    """Compile (or load from cache) every kernel using tiny inputs.

    The argument types match real calls exactly, so no recompiling later.
    """
    labels = np.array([0, 0, 1, 0, 1, 1, 2, 2, 1], dtype=np.int32)
    ids = np.empty(9, dtype=np.int32)
    region_color, areas = label_components(labels, 3, 3, 3, ids)
    merge_small_regions(ids, 3, 3, areas, 3, True)
    merge_same_color_neighbors(ids, 3, 3, region_color, areas, True)
    edge_adjacency_classes(ids, 3, 3, int(ids.max()) + 1, np.empty(9, dtype=np.int8))
    region_bounds(ids, 3, 3, int(ids.max()) + 1)
    nearest_seed_within(labels - 1, labels >= 0, 3, 3)
    region_color_sums(ids, np.zeros((9, 3), dtype=np.uint8), np.ones(int(ids.max()) + 1, dtype=np.bool_), labels >= 0)

    # Line art's region steps.
    count = int(ids.max()) + 1
    flags = np.ones(count, dtype=np.bool_)
    printed = labels == 2
    ids32 = np.empty(9, dtype=np.int32)
    taken = np.empty(9, dtype=np.bool_)
    nearest_seed_in_groups(labels - 1, ids, 3, 3)
    white_of(ids, printed, flags, ids32)
    piece_region = white_pieces(ids, 3, 3, ids32)
    split_seeds(ids, ids32, np.ones(piece_region.size, dtype=np.bool_), flags, np.empty(9, dtype=np.int32), labels - 1)
    settle_gaps(ids32, labels - 1, np.zeros(piece_region.size, dtype=np.bool_))
    claim_walls(labels - 1, printed, ids, 3, 3, taken)
    claim_walls(labels - 1, printed, np.zeros(0, dtype=np.int32), 3, 3, taken)
    apply_split(ids.copy(), labels.copy(), labels, printed, ids, np.zeros(count, dtype=np.int64), 3, 3, taken)
    painted_labels(ids, region_color, 3, ids32)
    regions_joined(ids, ids, count)
    follow_white(ids, ids, ids, count, ids32)
    seams_round(ids, printed, flags, 3, 3, ids32, taken)
    region_census(ids, printed, count, 3, 3)
    unreached_by_brush(ids, np.ones(9, dtype=np.float32), 1.5, taken)
    pocket_contacts(labels, ids, printed, 3, 3, 3)
    thin_ink_to_nearest(ids, printed, labels, np.ones(9, dtype=np.float32), 1.5, 3, 3, ids32, taken)
    keep_anchored(ids, ids, printed, ids32)
    paper_rings(labels - 1, ids, printed, 3, 3, 3)
    ink_is_main_neighbor(ids, flags, 3, 3)
    nearest_core_color(labels, printed, ids, region_color, ids32)
    across_to_nearest_core(labels, printed, labels, printed, taken)

    from tessellatum.core.boundaries import crack_edges  # imported here: boundaries imports this module

    right, down, degree, num_edges = crack_edges(ids.reshape(3, 3))
    trace_boundary_paths(right, down, degree, 4, num_edges)

    padded = np.zeros(5 * 5 * 3, dtype=np.uint8)
    out = np.empty(3 * 3 * 3, dtype=np.uint8)
    offsets = np.array([0], dtype=np.int64)
    weights = np.ones(1, dtype=np.float32)
    bilateral_rows(padded, out, 0, 3, 3, 1, offsets, weights, np.ones(766, dtype=np.float32))
