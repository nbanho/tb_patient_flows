import os, argparse
import numpy as np
import pandas as pd
import time
from read_tracking_linked_data import read_linked_tracking_data, pad_track_data
from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor

base_path = 'data-clean/tracking/'
save_path = os.path.join(base_path, 'occupancy/')
linked_tb_path = 'data-clean/tracking/linked-tb/'

START_SEC = 6 * 3600
TOTAL_SECONDS = 12 * 3600 + 1

EVAL_COORDS = None      # unused in fast path but we keep it for compatibility
CELL_AREA   = None
BANDWIDTH   = None
MASK        = None
GRID_META   = None      # (y0, x0, cell_size, H, W)
SIGMA_PIX   = None

# create fixed spatial domain
def create_spatial_domain(cell_size=0.5):
    image_extent = (0, 51, -0.02, 14.214)
    x1, x2, y1, y2 = image_extent
    x_grid = np.arange(x1, x2, cell_size)
    y_grid = np.arange(y1, y2, cell_size)
    xx, yy = np.meshgrid(x_grid, y_grid)

    waiting_area_1 = [[8, 0.5], [47.5, 0.5], [47.5, 6.3], [8, 6.3]]
    waiting_area_2 = [[0, 6.3], [51, 6.3], [51, 8.9], [0, 8.9]]
    waiting_area_3 = [[0, 8.9], [8, 8.9], [8, 12.8], [0, 12.8]]
    polygons = [Polygon(p) for p in (waiting_area_1, waiting_area_2, waiting_area_3)]

    mask = np.zeros(xx.shape, dtype=bool)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            pt = Point(xx[i, j], yy[i, j])
            mask[i, j] = any(poly.contains(pt) for poly in polygons)
    return mask, xx, yy, cell_size

def _to_grid_idx(x, y, grid_meta):
    y0, x0, cell_sz, H, W = grid_meta
    j = np.floor((x - x0) / cell_sz + 0.5).astype(np.int32)  # round to nearest
    i = np.floor((y - y0) / cell_sz + 0.5).astype(np.int32)
    # clip to grid
    np.clip(i, 0, H-1, out=i)
    np.clip(j, 0, W-1, out=j)
    return i, j


# helpers
def gini(p):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return np.nan
    ps = np.sort(p)            
    n = ps.size
    # G = (2 * sum(i * ps_i) / (n * s)) - (n + 1)/n
    return float((2.0 * np.dot(np.arange(1, n + 1, dtype=np.float64), ps) / (n * s)) - ((n + 1.0) / n))


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return np.nan
    p = p / s
    p = np.clip(p, 1e-300, 1.0)
    return float(-np.sum(p * np.log(p)))

def spatial_kde(x, y):
    pts = np.column_stack((x, y))
    kde = KernelDensity(bandwidth=float(BANDWIDTH), kernel='gaussian')
    kde.fit(pts)
    log_d = kde.score_samples(EVAL_COORDS)
    d = np.exp(log_d)
    Z = d.sum() * CELL_AREA
    p = (d * CELL_AREA) / Z
    return gini(p), entropy(p)

def process_date(date):
    # load and pad data
    df = read_linked_tracking_data(date)
    df = pad_track_data(df)
    
    # compute density metrics per second
    t0 = time.time()
    times = np.arange(TOTAL_SECONDS, dtype=np.int32)

    # quick empty-day path
    if df.empty:
        res = pd.DataFrame({
            'time': times,
            'N': np.zeros_like(times),
            'gini_kde': np.full_like(times, 0.0, dtype=float),
            'entropy_kde': np.full_like(times, 0.0, dtype=float)
        })
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, f'{date}.csv')
        res.to_csv(path, index=False)
        return path

    # work as NumPy to avoid groupby overhead
    t_arr = df['time'].to_numpy(np.int32, copy=False)
    x_arr = df['position_x'].to_numpy(np.float64, copy=False)
    y_arr = df['position_y'].to_numpy(np.float64, copy=False)
    id_arr = df['new_track_id'].to_numpy(np.int32, copy=False)

    # get unique seconds and slices
    order = np.argsort(t_arr, kind='mergesort')
    t_sorted = t_arr[order]
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    id_sorted = id_arr[order]

    uniq_t, idx_start, counts = np.unique(t_sorted, return_index=True, return_counts=True)
    idx_end = idx_start + counts

    H, W = MASK.shape
    cell_area = CELL_AREA
    sigma_pix = SIGMA_PIX

    # pre-alloc arrays for result; default zeros for missing seconds
    N_per_sec = np.zeros(TOTAL_SECONDS, dtype=np.int32)
    G_per_sec = np.zeros(TOTAL_SECONDS, dtype=np.float64)
    E_per_sec = np.zeros(TOTAL_SECONDS, dtype=np.float64)

    # flat mask view (for quick extraction)
    mask_flat = MASK.ravel()

    # main loop over seconds that have data
    for pos, t in enumerate(uniq_t):
        sl = slice(idx_start[pos], idx_end[pos])

        # unique people that second (you already pre-averaged to 1Hz)
        # if duplicates could still exist, do a tiny unique on ids and slice
        N = len(np.unique(id_sorted[sl]))
        N_per_sec[t] = N
        if N < 2:
            # keep defaults: G=0, E=0 for N in {0,1} (adjust if you prefer NaN)
            continue

        # snap to grid
        ii, jj = _to_grid_idx(x_sorted[sl], y_sorted[sl], GRID_META)

        # build occupancy image (H x W), only on valid floor: mask out invalid hits
        # optional: filter to valid mask to avoid adding where MASK=False
        valid_hits = MASK[ii, jj]
        ii = ii[valid_hits]; jj = jj[valid_hits]

        occ = np.zeros((H, W), dtype=np.float32)
        # count multiple people falling into the same cell
        np.add.at(occ, (ii, jj), 1.0)

        # smooth with Gaussian (bandwidth in pixels)
        # mode='constant' keeps zeros outside domain
        dens = gaussian_filter(occ, sigma=sigma_pix, mode='constant', cval=0.0)

        # restrict to valid domain and normalize to probability mass
        d = dens.ravel()[mask_flat].astype(np.float64)
        S = d.sum() * cell_area
        if S <= 0.0:
            # degenerate; keep zeros
            continue
        p = (d * cell_area) / S

        # evenness metrics
        G_per_sec[t] = gini(p)
        E_per_sec[t] = entropy(p)

    res = pd.DataFrame({
        'time': times,
        'N': N_per_sec,
        'gini_kde': G_per_sec,
        'entropy_kde': E_per_sec
    })

    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f'{date}.csv')
    res.to_csv(path, index=False)
    
    print(f'process_date {date} {time.time() - t0:.2f} seconds')
    
    return path

def _init_worker(eval_coords_, cell_area_, bandwidth_, mask_=None, grid_meta_=None, cell_size_=None):
    # keep backward compatibility with your current initargs; we also allow passing mask/grid_meta
    global EVAL_COORDS, CELL_AREA, BANDWIDTH, MASK, GRID_META, SIGMA_PIX
    EVAL_COORDS = eval_coords_
    CELL_AREA   = cell_area_
    BANDWIDTH   = bandwidth_
    MASK        = mask_
    GRID_META   = grid_meta_
    if grid_meta_ is not None:
        y0, x0, cell_sz, H, W = GRID_META
        SIGMA_PIX = float(BANDWIDTH / cell_sz)
    elif cell_size_ is not None:
        # fall back if only cell_size was sent
        SIGMA_PIX = float(BANDWIDTH / cell_size_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell-size', type=float, default=0.5)
    ap.add_argument('--bandwidth', type=float, default=1.5)
    ap.add_argument('--cores', type=int, default=2)
    args = ap.parse_args()
    mask, xx, yy, cell_size = create_spatial_domain(args.cell_size)
    valid_idx = np.flatnonzero(mask.ravel())
    eval_coords = np.column_stack((xx.ravel()[valid_idx], yy.ravel()[valid_idx]))
    cell_area = float(cell_size * cell_size)

    # precompute grid meta for fast indexing
    # grid is regular: x = x0 + j*cell_size, y = y0 + i*cell_size
    x0 = xx[0,0]; y0 = yy[0,0]
    H, W = mask.shape
    grid_meta = (y0, x0, cell_size, H, W)

    dates = [f.replace('.csv', '') for f in os.listdir(linked_tb_path) if f.endswith('.csv')]

    with ProcessPoolExecutor(max_workers=args.cores,
                             initializer=_init_worker,
                             initargs=(eval_coords, cell_area, args.bandwidth, mask, grid_meta, cell_size)) as ex:
        for path in ex.map(process_date, dates):
            print(f"Saved: {path}")



if __name__ == '__main__':
    main()
