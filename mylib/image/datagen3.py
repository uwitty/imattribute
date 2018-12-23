import codecs
import cv2
import os
import random

import keras.utils
import numpy as np

from mylib.image import utils

class DataGen3(object):
    def __init__(self, items, tags, output_names,
                 size=None, image_dir='images', batch_size=32, shuffle=True):
        self.items = None
        self.tags = tags
        self.output_names = output_names

        self.size = size
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.classes = [e[0] for e in tags]
        self.num_classes = len(tags)
        self.org_items = items

        if len(tags) != len(output_names):
            raise RuntimeError('length of tags and output_names are not matched')

        self.index = 0

    def __len__(self):
        return len(self.org_items)

    def __next__(self):
        result = []
        for i in range(self.batch_size):
            if self.items is None or self.index >= len(self.org_items):
                if self.shuffle:
                    self.items = random.sample(self.org_items, k=len(self.org_items))
                else:
                    self.items = self.org_items
                self.index = 0

            result.append(self.items[self.index])
            self.index += 1

        #print(result)
        return read_image_from_items(result, self.output_names,
                                     size=self.size,
                                     image_dir=self.image_dir,
                                     augmentate=self.shuffle)


def read_image_from_items(items, output_names,
                          size=None, image_dir='images', augmentate=False):
    def fpath(fname):
        return os.path.join(image_dir, fname)

    converter = utils.imaug if augmentate else None
    x = [np.array(utils.read_image(fpath(item['image']), size=size, converter=converter)) / 255.0 for item in items]

    y = {}
    for index, output_name in enumerate(output_names):
        #y[output_name] = np.zeros(2, dtype=np.int)
        y_tmp = [1 if index in item['tags'] else 0
                 for item in items]
        y[output_name] = keras.utils.to_categorical(y_tmp, 2)

    #print(x, y)
    #print(x[0].shape)
    return (np.array(x), y)


def main():
    import json
    fname = 'output2.json'
    with codecs.open(fname, 'r', 'utf_8') as f:
        info = json.load(f)
    output_names = ['output{}'.format(i) for i in range(len(info['tags']))]

    gen = DataGen3(info['train'], info['tags'],
                   output_names=output_names,
                   batch_size=32,
                   image_dir='/path/to/images_dir')

    print(len(gen))
    print(gen.__next__())

if __name__ == '__main__':
    main()
