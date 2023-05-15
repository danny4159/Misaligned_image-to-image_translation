import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import nibabel as nib
from monai.visualize import matshow3d, blend_images
import math

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

def plot_images(images, labels, siz=4, cmap=None):
    """
    This function plots a list of images with corresponding labels.
    
    Parameters:
    -----------
    images : list of ndarray
        List of images. Each image should be a 2D or 3D ndarray.
    labels : list of str
        List of labels. Each label corresponds to an image.
    siz : int, optional
        Size of each image when plotted. Default is 4.
    cmap : str, optional
        Colormap to use for displaying images. If 'gray', the image will be displayed in grayscale. 
        Default is None, in which case the default colormap is used.
        
    Raises:
    -------
    AssertionError
        If the number of images does not match the number of labels.
    
    Examples:
    ---------
    >>> img1 = np.random.rand(10, 10)
    >>> img2 = np.random.rand(10, 10)
    >>> plot_images([img1, img2], ['Image 1', 'Image 2'])
    
    >>> img1 = np.random.rand(10, 10)
    >>> img2 = np.random.rand(10, 10)
    >>> plot_images([img1, img2], ['Image 1', 'Image 2'], cmap='gray')
    """
    assert len(images) == len(labels), "Mismatch in number of images and labels"
    n = len(images)
    
    plt.figure(figsize=(siz*n, siz))  # Adjust figure size based on number of images
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        if cmap == 'gray':
            plt.gray()
        plt.title(labels[i])
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
 
 # TODO: 여기에 필요한 함수를 추가 (docstring)