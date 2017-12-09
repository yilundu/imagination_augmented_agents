import os

import scipy.misc

from ..image import combine_images
from ..shell import mkdir, rm, run


def load_video(video, clean = True, verbose = False):
    mkdir('%s.images/' % video, clean = True)
    run(('ffmpeg', ('-i', video), '%s.images/%%d.png' % video), verbose)
    images = [None] * len(os.listdir('%s.images/' % video))
    for name in os.listdir('%s.images/' % video):
        index = int(name.split('.')[0]) - 1
        images[index] = scipy.misc.imread(os.path.join('%s.images' % video, name))
    if clean:
        rm('%s.images/' % video)
    return images


def save_video(images, video, clean = True, verbose = False):
    mkdir('%s.images/' % video, clean = True)
    for i, image in enumerate(images):
        scipy.misc.imsave(os.path.join('%s.images/' % video, '%d.png' % i), image)
    rm(video)
    run(('ffmpeg', ('-r', 60), ('-f', 'image2'), ('-s', '1920x1080'), ('-i', '%s.images/%%d.png' % video),
         ('-vcodec', 'libx264'), ('-crf', 25), ('-pix_fmt', 'yuv420p'), video), verbose)
    if clean:
        rm('%s.images/' % video)


def combine_videos(videos, ncolumns):
    nvideos, nframes = len(videos), len(videos[0])
    result = []
    for i in range(nframes):
        images = [videos[k][i] for k in range(nvideos)]
        result.append(combine_images(images, ncolumns))
    return result
