"""
Save all global variables here
"""
import numpy as np

# Read in cars and notcars
folder_car = 'data/vehicles'
folder_notcar = 'data/non-vehicles'

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = -1

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 4  # HOG pixels per cell
cell_per_block = 1  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = False  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, 700]  # Min and max in y to search in slide_window()
smallest = [420, 475]
horizon = np.mean(smallest)
# calculate the y height series for window size series, eg [28, 28] [64, 64]
ysize_series = (64 * np.logspace(-1, 3, num=3 + 1 + 1, base=1.5)).astype(int)
print("ysize_series: ", ysize_series)
num_of_series = len(ysize_series)
ratio = np.power(((y_start_stop - horizon) / (smallest - horizon)), 1 / float(num_of_series - 1))
y_step = np.ceil(np.vstack([(smallest - horizon) * np.power(ratio, i) for i in range(num_of_series)]))
y_bound_list = (horizon + y_step).astype(int)       # !!!!!!!!! need iterate
print("y_bound_list: ", y_bound_list)
# calculate the sliding window shape series
xy_winshape = np.asarray([[ysize, ysize] for ysize in ysize_series]).astype(int)   # !!!!!! need iterate
print("xy_winshape: ", xy_winshape)
# calculate the minimum number of windows such that the overlap > 40%
y_height = y_bound_list[:, 1, np.newaxis] - y_bound_list[:, 0, np.newaxis]
xy_dim = np.hstack((np.ones_like(y_height) * 1280, y_height))
# min_winnum_xy = np.ceil((0.4 * xy_winshape + xy_dim) / xy_winshape)
min_winnum_xy = np.ceil((xy_dim - 0.5 * xy_winshape) / 0.5 / xy_winshape)
# min_winnum_y = np.ceil((0.4 * xy_winshape[:, 0, np.newaxis] + y_height) / xy_winshape[:, 0, np.newaxis])
# min_winnum_x = np.ceil((0.4 * xy_winshape[:, 0, np.newaxis] + 1280) / xy_winshape[:, 0, np.newaxis])
# calculate the overlap # !!! need iterate
overlap_xy = (xy_winshape * min_winnum_xy - xy_dim) / (min_winnum_xy - 1) / xy_winshape
print("overlap_xy", overlap_xy)

