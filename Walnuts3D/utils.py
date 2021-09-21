import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pylab


# Summary functions to view logs in the tensorboard

def summary_image(writer, name, image, it):
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image) + 1e-5)
    writer.add_image(name, image, it, dataformats='HW')

def summary_volume(writer, name, tensor, it):
    b_size, c_size, x_size, y_size, z_size = tensor.shape
    tensor_x = tensor[0, 0, x_size // 2, :, :].transpose(1, 0)
    summary_image(writer, name + '/x', tensor_x, it)
    tensor_y = tensor[0, 0, :, y_size // 2, :].transpose(1, 0)
    summary_image(writer, name + '/y', tensor_y, it)
    tensor_z = tensor[0, 0, :, :, z_size // 2]
    summary_image(writer, name + '/z', tensor_z, it)
    
def summaries(writer, result, true, loss, it, do_print=False):
    residual = result - true
    squared_error = residual ** 2
    mse = torch.mean(squared_error)
    psnr = 20 * torch.log10(torch.max(true)) - 10 * torch.log10(mse)
   
    if do_print:
        print(it, mse.item(), psnr.item())

    writer.add_scalar('loss', loss, it)
    writer.add_scalar('psnr', psnr, it)

    summary_volume(writer, 'result', result, it)
    #summary_volume(writer, 'true', true, it)

    
# Functions for showing and saving images
    
def show(rec):
        s = rec.shape[0] 
        rec_slices = [rec[s // 2, :, :], rec[:, s // 2, :], rec[:, :, s // 2]]
        fig, axes = pylab.subplots(1, 3, figsize=[20, 6])
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        for i in range(3):
            axes[i].imshow(rec_slices[i], clim=[0, 0.0814418], cmap='gray')
            axes[i].xaxis.set_visible(False)
            axes[i].yaxis.set_visible(False)
        fig.show()
        
def savefig(name):
    if not os.path.exists('./figures'):
        os.makedirs('./figures')    
    plt.savefig('figures/{}.pdf'.format(name));

def plot(im, name, clim = True):
    DPI = 100.0
    fig = plt.figure(figsize=np.array(im.shape) / DPI, dpi=DPI)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if clim:
        im = ax.imshow(im, clim=[0, 0.0814418], cmap = 'gray');
    else:
        im = ax.imshow(im, cmap = 'gray');
    savefig(name);
    plt.close(fig)

def plot3D(rec, name):
    s = rec.shape[0] 
    rec_slices = [rec[s // 2, :, :], rec[:, s // 2, :], rec[:, :, s // 2]]
    for i in range(3):
        plot(rec_slices[i], name+"_slice_{}".format(i), True)