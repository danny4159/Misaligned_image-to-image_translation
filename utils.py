import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import nibabel as nib
from monai.visualize import matshow3d, blend_images
import math
from PIL import Image
import random
import torch
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict


blend_and_transpose = lambda x, y, alpha=0.3: np.transpose(blend_images(x[None], y[None], alpha,cmap='hot'), (1, 2, 0))
"""
This lambda function blends two images and transposes the resulting image.

Parameters:
-----------
x : ndarray
    First image to blend. Should be a 2D ndarray.
y : ndarray
    Second image to blend. Should be a 2D ndarray.
alpha : float, optional
    The weight for blending the images. The higher the alpha, the more weight for the second image. Default is 0.3.

Returns:
--------
ndarray
    The blended and transposed image. Should be a 2D ndarray.

Examples:
---------
>>> img1 = np.random.rand(10, 10)
>>> img2 = np.random.rand(10, 10)
>>> blended_img = blend_and_transpose(img1, img2)
"""

# def plot_images(images, labels, siz=4, cmap=None):
#     """
#     This function plots a list of images with corresponding labels.
    
#     Parameters:
#     -----------
#     images : list of ndarray
#         List of images. Each image should be a 2D or 3D ndarray.
#     labels : list of str
#         List of labels. Each label corresponds to an image.
#     siz : int, optional
#         Size of each image when plotted. Default is 4.
#     cmap : str, optional
#         Colormap to use for displaying images. If 'gray', the image will be displayed in grayscale. 
#         Default is None, in which case the default colormap is used.
        
#     Raises:
#     -------
#     AssertionError
#         If the number of images does not match the number of labels.
    
#     Examples:
#     ---------
#     >>> img1 = np.random.rand(10, 10)
#     >>> img2 = np.random.rand(10, 10)
#     >>> plot_images([img1, img2], ['Image 1', 'Image 2'])
    
#     >>> img1 = np.random.rand(10, 10)
#     >>> img2 = np.random.rand(10, 10)
#     >>> plot_images([img1, img2], ['Image 1', 'Image 2'], cmap='gray')
#     """
#     assert len(images) == len(labels), "Mismatch in number of images and labels"
#     n = len(images)
    
#     plt.figure(figsize=(siz*n, siz))  # Adjust figure size based on number of images
#     for i in range(n):
#         plt.subplot(1, n, i+1)
#         plt.imshow(images[i])
#         if cmap == 'gray':
#             plt.gray()
#         plt.title(labels[i])
#     plt.show()

def plot_images(images, image_names):
    """
    Plot images using matplotlib.

    Args:
        images (list): A list of images. Each image should be a 2D numpy array.
        image_names (list): A list of names for the images. The names are used as titles for the subplots.

    """
    assert len(images) == len(image_names)
    plt.figure(figsize=(5, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(image_names[i])
        plt.imshow(images[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    plt.show()


def slice_array(array, block_size):
    """
    Slice a 3D NumPy array into smaller blocks along the third dimension.

    Args:
        array (numpy.ndarray): Input array to be sliced. It should have 3 dimensions.
        block_size (int): Size of each block along the third dimension.

    Returns:
        numpy.ndarray: Sliced array with shape (array.shape[0], array.shape[1], block_size, num_slices),
            where num_slices is array.shape[2] divided by block_size.

    Raises:
        AssertionError: If the input array does not have 3 dimensions or the block size does not evenly divide
            the shape of the input array along the third dimension.

    """
    assert len(array.shape) == 3, "Input array should have 3 dimensions."
    assert array.shape[2] % block_size == 0, "Block size should evenly divide the array shape."

    num_slices = array.shape[2] // block_size
    sliced_array = np.zeros((array.shape[0], array.shape[1], block_size, num_slices))

    for i in range(num_slices):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        sliced_array[:, :, :, i] = array[:, :, start_idx:end_idx]

    return sliced_array

def save_slices_to_nii(sliced_array, output_prefix):
    """
    Save the individual slices of a 4D NumPy array as NIfTI files.

    Args:
        sliced_array (numpy.ndarray): Input array containing the slices to be saved. It should have 4 dimensions.
        output_prefix (str): Prefix to be used for the output file names.

    Raises:
        AssertionError: If the input array does not have 4 dimensions.

    """
    assert len(sliced_array.shape) == 4, "Input array should have 4 dimensions."

    for i in range(sliced_array.shape[3]):
        data = sliced_array[:, :, :, i]
        nifti_img = nib.Nifti1Image(data, affine=np.eye(4))
        output_filename = f"{output_prefix}_{i+1}.nii.gz"
        nib.save(nifti_img, output_filename)


def calculate_mutual_info(image1, image2):
    """
    This function calculates the mutual information between two images.

    Parameters:
    image1 (np.array): The first image
    image2 (np.array): The second image

    Returns:
    float: The mutual information score
    """
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=20)
    return mutual_info_score(None, None, contingency=hist_2d)

def plot_blended_images(t1, t2, ncols=None):
        
    # Assuming blend_images is a function you've already defined
    # Blend images
    blended_images = np.zeros((t1.shape[0], t1.shape[2], t1.shape[3], 3))

    for i in range(t1.shape[0]):
        blended_images[i] = np.transpose(blend_images(t1[i], t2[i], alpha=0.15),(1,2,0))

    # Convert the 4D array into list of 3D arrays
    list_of_arrays = [blended_images[i] for i in range(blended_images.shape[0])]

    if ncols is None:
        ncols = math.ceil(np.sqrt(len(list_of_arrays)))
    
    # Calculate nrows and ncols
    nrows = math.ceil(len(list_of_arrays) / ncols)

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))

    # In case there are less images than slots in the grid, remove empty slots
    if nrows * ncols > len(list_of_arrays):
        for idx in range(len(list_of_arrays), nrows * ncols):
            fig.delaxes(axes.flatten()[idx])

    for idx, image in enumerate(list_of_arrays):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].imshow(image)
        axes[row, col].axis('off')  # to remove the axis

    plt.tight_layout()
    plt.show()
    
    return fig

def save_images(images, image_names, save_dir, epoch):
    """
    Save a list of images as PNG files.

    Args:
        images (list): A list of NumPy ndarrays representing the images to be saved.
        image_names (list): A list of strings representing the names to be used when saving the images.
        save_dir (str): The path to the directory where the images will be saved.
        epoch (int): The current epoch number to be included in the file name.

    """
    for image, name in zip(images, image_names):
        # Save the image as a PNG file
        plt.imsave(f"{save_dir}/epoch_{epoch}_{name}.png", image)

def tensor2im_minmax(image_tensor, imtype=np.uint8):
    """
    Convert a PyTorch tensor to an image (numpy array) with pixel values in the range 0~255.

    Args:
        image_tensor (torch.Tensor): A PyTorch tensor representing an image, with shape [C, H, W]
            where C is the number of channels, H is the height, and W is the width. The first dimension 
            should be the channels dimension, and it should have at least 1 channel. If it only has 1 
            channel, that channel is duplicated to form a 3-channel image.
        imtype (type, optional): The desired type for the pixels of the output image. Default is np.uint8,
            but it can be any type that is compatible with numpy arrays.

    Returns:
        np.ndarray: A 2D numpy array representing the image, with shape [H, W, C] and type `imtype`.
            The pixel values are scaled to be in the range 0~255.

    Note:
        The function first finds the minimum and maximum values in the input tensor, then scales the pixel 
        values to be in the range 0~1, and finally scales them to be in the range 0~255. As a result, 
        the output image has the full range of possible brightness levels, regardless of the range of the 
        pixel values in the input tensor.
    """
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        
    # Find min and max values
    min_val = np.min(image_numpy)
    max_val = np.max(image_numpy)

    # Scale to 0~1 range
    image_numpy = (image_numpy - min_val) / (max_val - min_val)

    # Scale to 0~255
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0 
    return image_numpy.astype(imtype)

# def get_current_visuals_for_pgan(real_A,fake_B,real_B):
#     """
#     This function prepares the visuals for a PGAN model.

#     It converts the real and fake images from tensors to images and returns them in an OrderedDict. 

#     Args:
#     real_A (torch.Tensor): Real image from the domain A.
#     fake_B (torch.Tensor): Generated fake image transformed from the domain A to B.
#     real_B (torch.Tensor): Real image from the domain B.

#     Returns:
#     OrderedDict: A dictionary containing the images. The keys are 'real_A', 'fake_B' and 'real_B' and the values are the corresponding images.
#     """
#     real_A = tensor2im_minmax(real_A.data)
#     fake_B = tensor2im_minmax(fake_B.data)
#     real_B = tensor2im_minmax(real_B.data)
#     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

def get_current_visuals_for_pgan(real_A,fake_B,real_B):
    """
    This function prepares the visuals for a PGAN model.

    It converts the real and fake images from tensors to images and returns them in an OrderedDict. 

    Args:
    real_A (torch.Tensor): Real image from the domain A.
    fake_B (torch.Tensor): Generated fake image transformed from the domain A to B.
    real_B (torch.Tensor): Real image from the domain B.

    Returns:
    OrderedDict: A dictionary containing the images. The keys are 'real_A', 'fake_B' and 'real_B' and the values are the corresponding images.
    """
    real_A, fake_B, real_B = real_A.data.cpu().numpy()[0], fake_B.data.cpu().numpy()[0], real_B.data.cpu().numpy()[0]

    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])



def get_current_visuals_for_cgan(real_A,fake_A,real_B,fake_B):
    """
    This function prepares the visuals for a CGAN model.

    It converts the real and fake images from tensors to images and returns them in an OrderedDict. 

    Args:
    real_A (torch.Tensor): Real image from the domain A.
    fake_A (torch.Tensor): Generated fake image similar to real image in domain A.
    real_B (torch.Tensor): Real image from the domain B.
    fake_B (torch.Tensor): Generated fake image similar to real image in domain B.

    Returns:
    OrderedDict: A dictionary containing the images. The keys are 'real_A', 'fake_A', 'real_B' and 'fake_B' and the values are the corresponding images.
    """
    # real_A = tensor2im_minmax(real_A.data)
    # fake_A = tensor2im_minmax(fake_A.data)
    # real_B = tensor2im_minmax(real_B.data)
    # fake_B = tensor2im_minmax(fake_B.data)
    real_A, fake_A, fake_B, real_B = real_A.data.cpu().numpy()[0], fake_A.data.cpu().numpy()[0], fake_B.data.cpu().numpy()[0], real_B.data.cpu().numpy()[0]


    return OrderedDict([('real_A', real_A), ('fake_A', fake_A), ('real_B', real_B), ('fake_B', fake_B)])

def plot_2d_slice(images, image_names, slice_idx):
    """
    Visualizes the 2D slices from the given 3D images at the specified slice index.

    Args:
        images (list[torch.Tensor]): List of 3D image tensors (batch_size, channel, width, height, depth) to be displayed.
        image_names (list[str]): List of names for the images for displaying as titles.
        slice_idx (int): The slice index along the depth axis to visualize the 2D slice from the 3D images.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))

    for idx, (image, title) in enumerate(zip(images, image_names)):
        slice_image = image[0, 0, :, :, slice_idx].cpu().numpy()
        axes[idx].imshow(slice_image, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(title)

    plt.show()

 # TODO: 여기에 필요한 함수를 추가 (docstring)