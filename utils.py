import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fig2data(cur_mat):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(cur_mat, annot=True, cbar=False, xticklabels=False, yticklabels=False,
                             annot_kws={'size': 20, 'weight': 'bold'})
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    img_array = Image.frombytes("RGBA", (w, h), buf.tobytes())
    img_array = np.asarray(img_array)

    return img_array

def get_count_inversions(array):
    count = 0
    array_n = array.shape[0]

    for i in range(array_n - 1):
        for j in range(i+1, array_n):
            if array[i] > array[j]:
                count += 1

    return count

def is_solvable(init_mat, org_mat):
    row_dis = np.where(init_mat == 0)[0] - 0

    initial_mat = np.reshape(init_mat, -1)
    target_mat = np.reshape(org_mat, -1)

    ''' get count inversions '''
    initial_mat_inver = get_count_inversions(initial_mat)
    target_mat_inver = get_count_inversions(target_mat)

    if initial_mat.shape[0] % 2 == 0:
        if ((initial_mat_inver+row_dis) % 2) == (target_mat_inver % 2):
            return True
        else:
            return False
    else:
        if (initial_mat_inver % 2) == (target_mat_inver % 2):
            return True
        else:
            return False