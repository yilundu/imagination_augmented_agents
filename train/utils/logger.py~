# reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import print_function

import os
from collections import OrderedDict

import numpy as np
import scipy.misc
# from PIL import Image
import tensorflow as tf

from . import shell


class Logger:
    def __init__(self, path):
        self.path = path
        self.writer = tf.summary.FileWriter(self.path)
        self.images = {}

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value = [tf.Summary.Value(tag = tag, simple_value = value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        # set up the web path
        print("Starting summary")
        web_path = os.path.join(self.path, 'web')

        # set up the image path
        path = os.path.join('images', '%s-%d' % (tag, step))
        shell.mkdir(os.path.join(web_path, path), clean = True)

        # save images to image path
        if step not in self.images:
            self.images[step] = OrderedDict()

        for i, image in enumerate(images):
            name = '%s-%d-%d.png' % (tag, step, i)
            print(image.shape)
            image = scipy.misc.toimage(image)
            print(os.path.join(web_path, path, name))
            scipy.misc.imsave(os.path.join(web_path, path, name), image)

            if tag not in self.images[step]:
                self.images[step][tag] = []
            self.images[step][tag].append(os.path.join(path, name))

        # set up html for visualization
        with open(os.path.join(web_path, 'index.html'), 'w') as stream:
            print('<meta http-equiv="refresh" content="60">', file = stream)
            for step in sorted(self.images.keys(), reverse = True):
                print('<h3>step [%d]</h3>' % step, file = stream)
                print('<table border="1" style="table-layout: fixed;">', file = stream)
                for tag in self.images[step].keys():
                    print('<tr>', file = stream)
                    for image in self.images[step][tag]:
                        print('<td halign="center" style="word-wrap: break-word;" valign="top">', file = stream)
                        print('<p>', file = stream)
                        print('<img src="%s" style="width:128px;">' % image, file = stream)
                        print('<br>', file = stream)
                        print('<p>%s</p>' % tag, file = stream)
                        print('</p>', file = stream)
                        print('</td>', file = stream)
                    print('</tr>', file = stream)
                print('</table>', file = stream)

    def histo_summary(self, tag, values, step, bins = 1000):
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        counts, edges = np.histogram(values, bins = bins)
        for edge in edges[1:]:
            hist.bucket_limit.append(edge)
        for count in counts:
            hist.bucket.append(count)
        summary = tf.Summary(value = [tf.Summary.Value(tag = tag, histo = hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
