# These functions simulate noisy projection data from image slices. 
# The slices were extracted from high quality reconstructions provided 
# during 2016 AAPM Low Dose CT Grand Challenge 
# https://www.aapm.org/grandchallenge/lowdosect/

import numpy as np
import odl
import os

# path to image slices
MAYO_FOLDER = ""

mu_water = 0.0192

# generate projection data from images
def generate_transform_mayo(images, operator, photons_per_pixel=50000):
    data = []
    for image in images:
        phantom = operator.domain.element(image.squeeze())
        transformed = operator(phantom)
        transformed = np.exp(-mu_water * transformed)
        noisy = odl.phantom.poisson_noise(transformed * photons_per_pixel)
        noisy = np.maximum(noisy, 1) / photons_per_pixel
        log_noisy = - np.log(noisy) / mu_water
        data.append(log_noisy.asarray())
    return np.expand_dims(np.array(data), axis = 1)

def generate_mayo(operator, mode, batch_size, val_ratio=0, photons_per_pixel=50000):
    shape = (batch_size, 1, 512, 512)
    folder = MAYO_FOLDER
    test = 'L286'
    data = []
    if mode == 'test':
        for (dirpath, dirnames, filenames) in os.walk(folder):
            data.extend([os.path.join(folder, fi) for fi in filenames if fi.startswith(test)])
        data = np.sort(data)
    else:
        for (dirpath, dirnames, filenames) in os.walk(folder):
            data.extend([os.path.join(folder, fi) for fi in filenames if not fi.startswith(test)])
        n_val = int(val_ratio * len(data))
        # Extract images for validation uniformly from the training dataset.
        # The same images are extracted, given the same val_ratio.
        data = np.sort(data)
        step = (len(data) + n_val - 1) // n_val
        if mode == 'validate':
            data = [data[i] for i in range(len(data)) if i % step == 0]
        else:
            data = [data[i] for i in range(len(data)) if i % step != 0]
    n_images = len(data)
    n_batches = int(n_images / batch_size)
    while True:
        if mode == 'train':
            np.random.shuffle(data)
        for i in range(n_batches):
            filenames = data[i * batch_size : (i + 1) * batch_size]
            images = []
            for fn in filenames:
                image = np.load(fn)
                image = image / 1000.0
                images.append(image)
            x = np.stack(images, axis = 0).reshape(shape)
            y = generate_transform_mayo(images, operator, photons_per_pixel)
            yield((x, y))
        if mode != 'train':
            break