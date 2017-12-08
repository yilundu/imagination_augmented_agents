import numpy as np


def combine_images(images, ncolumns):
    if len(images) % ncolumns != 0:
        raise NotImplementedError
    columns = [np.concatenate(images[i::ncolumns], 0) for i in range(ncolumns)]
    result = np.concatenate(columns, 1)
    return result
