"""Create 2D spatial grid, floor-plan mask, and volume for the hospital waiting area.

Generates the building geometry files required by the diffusion model:
  - Grid cell size (m)
  - Binary mask of valid (indoor) grid cells
  - Coordinates of valid grid positions (for KD-tree lookups in preprocessing)
  - Room height (m)
  - Combined room volume (m3)

Must be run before run_model.py or any preprocessing script that uses the
building grid. Outputs are saved to data-clean/building/.
"""

import numpy as np
from shapely.geometry import Point, Polygon


def polygon_area(coords):
    """Compute the area of a polygon using the shoelace formula."""
    x, y = zip(*coords)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


if __name__ == "__main__":
    # Spatial extent of the Xovis sensor footprint (meters), derived from
    # the floor plan image calibration.
    image_extent = (0, 51, -0.02, 14.214)
    x1, x2, y1, y2 = image_extent
    cell_size = 0.5  # meters
    np.save('data-clean/building/building-grid-cell-size.npy', cell_size)

    x_grid = np.arange(x1, x2, cell_size)
    y_grid = np.arange(y1, y2, cell_size)

    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_shape = xx.shape

    zero_matrix = np.zeros(grid_shape)

    # Polygon vertices (meters) for the three sections of the waiting area,
    # measured from the architectural floor plan:
    #   1: main seating area (south wing)
    #   2: central corridor connecting wings
    #   3: north wing seating area
    waiting_area_1 = [[8, 0.5], [47.5, 0.5], [47.5, 6.3], [8, 6.3]]
    waiting_area_2 = [[0, 6.3], [51, 6.3], [51, 8.9], [0, 8.9]]
    waiting_area_3 = [[0, 8.9], [8, 8.9], [8, 12.8], [0, 12.8]]
    polygons = [Polygon(area) for area in [waiting_area_1, waiting_area_2, waiting_area_3]]

    # Create binary mask: True where grid cell center falls inside a waiting area
    mask = np.zeros(grid_shape, dtype=bool)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            point = Point(xx[i, j], yy[i, j])
            mask[i, j] = any(poly.contains(point) for poly in polygons)

    np.save('data-clean/building/building-grid-mask.npy', mask)

    # Precompute valid grid coordinates where mask is True (for KD-tree in preprocessing)
    valid_indices = np.argwhere(mask)
    valid_positions = np.column_stack((xx[mask], yy[mask]))
    np.save('data-clean/building/building-grid-mask-valid-positions.npy', valid_positions)

    # Room volume
    height = 3  # floor-to-ceiling height (meters), from building documentation
    np.save('data-clean/building/building-height.npy', height)

    area1 = polygon_area(waiting_area_1)
    area2 = polygon_area(waiting_area_2)
    area3 = polygon_area(waiting_area_3)

    vol1 = area1 * height
    vol2 = area2 * height
    vol3 = area3 * height
    combined_vol = vol1 + vol2 + vol3

    np.save('data-clean/building/building-volume.npy', combined_vol)
