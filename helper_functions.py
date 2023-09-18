"""
    A list of helper functions to assist data generation, model training, plotting images etc.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def one_hot_encoding(size=10):
    """
    Helper function returning one-hot encoding for the provided max size
    params:
    -------
        * size : Max possible size, default = 10
    returns:
    --------
        * Dictionary containing one hot encoding for each digit
    """
    digit_encoding = {i : [0]*10 for i in range(0, size)}
    for k in digit_encoding.keys():
        digit_encoding[k][k]=1
    return digit_encoding

def plot_images(nrows, ncols, figsize, img_arr, labels, default_shape = (-1, 28, 28), cmap = None, vmax = 1, vmin = 0):
    """
    Helper function used to plot images in a grid
    params:
    -------
        * nrows : Number of rows
        * ncols : Number of cols
        * figsize : size of the figure
        * img_arr : List of images
        * labels : Label corresponding to entries in img_arr
        * default_shape : Shape of each image
    returns:
    --------
        * None
    """
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    plt.setp(axes.flat, xticks = [], yticks = [])
    for i, ax in enumerate(axes.flat):
        ax.imshow(img_arr[i], vmax=vmax, vmin=vmin, cmap = cmap)
        if len(labels)>=1:
            ax.set_title(labels[i])
    plt.tight_layout()
    plt.show()

def clipped_zoom(img, zoom_factor, **kwargs):
    """
        A helper function which creates random crops/ zoom of images but with the same resolution as the input image.
        params:
        -------
            * img : image to be cropped/ zoomed
            * zoom_factor : eg. 1.5, 0.75 etc.
        returns:
        --------
            * return the cropped/ zoomed image
    """

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out