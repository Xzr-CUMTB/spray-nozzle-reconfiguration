"""Voxel traversal for trajectory segments."""
import numpy as np
from numba import njit

@njit(cache=True)
def mark_segment(a, b, origin, h, covered):
    """DDA by intervals between grid crossings; zero-measure touches add no cell.

    Cells are half-open. Segments lying in a grid plane belong to its positive
    side. A terminal face touch does not mark the cell beyond that face.
    """
    delta = b-a
    if np.sum(delta*delta) < 1e-30:
        return
    t = 0.
    max_intervals = sum(covered.shape)+4
    for _ in range(max_intervals):
        tnext = 1.
        probe = min(1., t+1e-11)
        for d in range(3):
            if abs(delta[d]) < 1e-15:
                continue
            v = (a[d]+probe*delta[d]-origin[d])/h
            k = np.floor(v)
            boundary = origin[d]+(k+1 if delta[d] > 0 else k)*h
            crossing = (boundary-a[d])/delta[d]
            if crossing > t+1e-13:
                tnext = min(tnext, crossing)
        if tnext > t+1e-14:
            p = a+((t+tnext)*.5)*delta
            ix = int(np.floor((p[0]-origin[0])/h))
            iy = int(np.floor((p[1]-origin[1])/h))
            iz = int(np.floor((p[2]-origin[2])/h))
            if 0 <= ix < covered.shape[0] and 0 <= iy < covered.shape[1] and 0 <= iz < covered.shape[2]:
                covered[ix,iy,iz] = True
        if tnext >= 1.-1e-13:
            return
        t = tnext
    raise RuntimeError('DDA iteration bound exceeded')
