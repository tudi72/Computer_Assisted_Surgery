import numpy as np
from scipy import ndimage
import queue
from collections import deque

def check_voxel(image,nx,ny,nz):

    NZ,NY,NX = image.shape

    # check that they are bigger than zero and smaller than 256
    if 0 <= nz < NZ and 0 <= ny < NY and 0 <= nx < NX:
        return True
    else:
        return False


def region_grow(image, seed_point):
    """
    Performs a region growing on the image starting from 'seed_point'
    :param image: A 3D grayscale input image
    :param seed_point: The seed point for the algorithm
    :return: A 3D binary segmentation mask with the same dimensions as 'image'
    """
    segmentation_mask = np.zeros(image.shape, np.bool_)
    z, y, x = seed_point
    intensity = image[z, y, x]
    print(f'Image data at position ({x}, {y}, {z}) has value {intensity}')
    print('Computing region growing...', end='', flush=True)

    ## TODO: choose a lower and upper threshold
    threshold_lower = -900
    threshold_upper = 900
    _segmentation_mask = (np.greater(image, threshold_lower)
                          & np.less(image, threshold_upper)).astype(np.bool_)

    ## TODO: pre-process the segmented image with a morphological filter
    structure = np.ones((3, 3, 3))
    segmentation_mask = ndimage.binary_closing(segmentation_mask, structure=structure).astype(np.bool_)

    to_check = deque()
    to_check.append((z, y, x))

    while to_check:
        z, y, x = to_check.popleft()

        if _segmentation_mask[z, y, x]:
            # Mark the current point as visited
            _segmentation_mask[z, y, x] = False
            segmentation_mask[z, y, x] = True

            # These for loops will visit all the neighbors of a voxel and see if
            # they belong to the region
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dz == 0 and dy == 0 and dx == 0:
                            continue    # Skip the center point
                        nz, ny, nx = z + dz, y + dy, x + dx

                        ## TODO: implement the code which checks whether the current
                        ## voxel (nz, ny, nx) belongs to the region or not
                        
                        threshold_reg = 200
                        # check whether voxel is inside boundaries and inside segmentation mask
                        if check_voxel(image,nx,ny,nz) and _segmentation_mask[nz, ny, nx]:
                            
                            if np.abs(image[nz, ny, nx] - intensity) < threshold_reg:   
                                to_check.append((nz, ny, nx))

                        ## OPTIONAL TODO: implement a stop criteria such that the algorithm
                        ## doesn't check voxels which are too far away

    # Post-process the image with a morphological filter
    structure = np.ones((3, 3, 3))
    segmentation_mask = ndimage.binary_closing(segmentation_mask, structure=structure).astype(np.bool_)
    
    print('\rComputing region growing... [DONE]', flush=True)

    return segmentation_mask