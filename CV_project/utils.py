import numpy as np
import numba


@numba.njit
def generate_hough_space_v1(
    image: np.ndarray,
    n_pixel_step: int = 1,
    lower_dist: int = 10,
    upper_dist: int = 100,
    n_lines: int = 5,
    list_theta: np.ndarray = None, # for part 3, we don't need to search through all angles
):
    """
    Descriptions & Comments:
        * This function returns a searching space (3D Hough space) from the image space.
        * There are 3 searching parameters: rho, theta, and distance (where distance is the distance from one line to next closest parallel line)
        * We assumed the distance between closest parallel lines are uniform.
        * The difference between v1 and v2 is that v2 is searching for "exact" pixel location for parallel lines while v1 does not.
    Arguments:
        image (np.ndarray): input image (2D binary edge image)
        n_pixel_step (int): number of step for searching (used in distance and rho)
        lower_dist (int): lower bound for 
        list_distance (list, np.ndarray): list of distances for each line to its closest line
        n_lines (int): number of parallel lines
        list_theta (np.ndarray): user defined angle range
    Returns:
        accumulator (np.ndarray): 3d array searching space (rho, theta, dist) 
            * rho: distance to the point in hough space
            * theta: angle to the point in hough space
            * dist: distance between each line (assuming uniform distance between each line)
        list_rho (np.ndarray): 1d array searching space for rho. This is used for referencing later to extract the value of rho
        list_theta (np.ndarray): 1d array searching space for theta. This is used for referencing later to extract the value of theta
        list_distance (np.ndarray): 1d array searching space for distance. This is used for referencing later to extract the value of distance
    """
    print('Running Hough transform...')
    # Get the longest distance rho parameter
    max_rho = int(np.sqrt(image.shape[0] * image.shape[0] + image.shape[1] * image.shape[1]))
    # (1) get searching space for rho
    list_rho = np.arange(-max_rho, max_rho, n_pixel_step)
    # (2) get searching space for theta (-90 to 90 degree angle)
    list_theta = np.linspace(-np.pi / 2, np.pi / 2, 360) if list_theta is None else list_theta # make a finer than 180 for more precise location
    # (3) get searching space for list_distance
    list_distance = np.arange(lower_dist, upper_dist, n_pixel_step)
    # 3d accumulator - for rho, angle, and distance
    accumulator = np.zeros(shape = (len(list_rho), len(list_theta), len(list_distance)))
    print('\t# of theta searching space:', len(list_theta))
    print('\t# of distance searching space:', len(list_theta))
    print('\t3D accumulator shape:', accumulator.shape)

    # accumulate accumulator for hough space
    ys, xs = np.where(image)
    for x,y in zip(xs, ys):
        for theta_idx, theta in enumerate(list_theta):
            rho = max_rho + int(x * np.cos(theta) + y * np.sin(theta))
            for dist_idx, d in enumerate(list_distance):
                num_parallel = 0
                for n in range(n_lines):
                    x_parallel, y_parallel = x + int((d * n) * np.cos(theta)), y + int((d * n) * np.sin(theta))
                    if image[y_parallel, x_parallel] > 0:
                        num_parallel += 1
                if num_parallel == n_lines:
                    accumulator[rho, theta_idx, dist_idx] += 1
    return accumulator, list_rho, list_theta, list_distance, n_lines

def get_ylines(
    accumulator: np.ndarray,
    list_rho: np.ndarray,
    list_theta: np.ndarray,
    list_distance: np.ndarray,
    n_lines: int = 5
):
    idx_rho, idx_theta, idx_dist = np.unravel_index(accumulator.argmax(), accumulator.shape)
    max_mag = accumulator[idx_rho, idx_theta, idx_dist]
    d = list_distance[idx_dist]
    list_y_search = []
    list_t_search = []
    list_r_search = []
    list_mag = []
    for idx_rho, rho in enumerate(list_rho):
        for idx_theta, theta in enumerate(list_theta):
            mag = accumulator[idx_rho, idx_theta, idx_dist]
            if mag > max_mag // 3:
                list_y_search.append(int((rho - 0 * np.cos(theta)) / np.sin(theta)))
                list_mag.append(mag)
                list_t_search.append(theta)
                list_r_search.append(rho)
    list_y_search = np.array(list_y_search)[np.argsort(list_mag)[::-1]]
    list_t_search = np.array(list_t_search)[np.argsort(list_mag)[::-1]]
    list_r_search = np.array(list_r_search)[np.argsort(list_mag)[::-1]]

    list_y = []
    list_t = []
    list_r = []
    d_thres = d*n_lines
    for y, t, r in zip(list_y_search, list_t_search, list_r_search):
        if all([i < y-d_thres or i > y+d_thres for i in list_y]):
            list_y.append(y)
            list_t.append(t)
            list_r.append(r)
    return list_y, list_r, list_t, d
