import numpy as np


def pad2d(matrix, p):
    """Zero pads a 2D matrix with p padding size.

    Args:
        matrix: original 2D numpy matrix
        p: size of padding around border

    Returns:
        padded matrix
    """
    if matrix is None or p is None:
        return None

    if p == 0:
        return matrix
    
    c, h, w = matrix.shape[-3:]
    new_shape = (c, h + 2 * p, w + 2 * p)
    transformed = np.zeros(new_shape, dtype=matrix.dtype)
    transformed[:, p: p + h, p: w + p] = matrix
    return transformed


def conv2d_naive(matrix, kernel, b=None, stride=1, padding=0):
    """Performs naive 2d convolution on matrix.
    
    Args:
        matrix: original matrix (C, H, W)
        kernel: mask
        b: bias
        stride: kernel stride
        padding: padding amount

    Returns:
        feature map
    """
    if matrix is None or kernel is None:
        return None
    
    matrix = pad2d(matrix, padding)
    c, h, w = matrix.shape[-3:]
    kernel_c, kernel_h, kernel_w = kernel.shape[-3:]
    
    h_out = np.floor((h - kernel_h + 2 * padding) / stride) + 1
    w_out = np.floor((w - kernel_w + 2 * padding) / stride) + 1
    new_shape = (h_out.astype(np.int64), w_out.astype(np.int64))

    feature_map = np.zeros(new_shape, dtype=matrix.dtype)

    output_i = 0
    for i in range(0, h - kernel_h + 1, stride):
        output_j = 0
        for j in range(0, w - kernel_w + 1, stride):
            patch = matrix[:, i: i + kernel_h, j: j + kernel_w]
            feature_map[output_i][output_j] = np.sum(patch * kernel)
            output_j += 1    
        output_i += 1

    return feature_map


def create_synth_img():
    """Creates synthetic image with an edge."""
    shape = (1, 16, 16)
    img = np.zeros(shape)
    img[:, :, 8:] = 1.0
    return img


# ========== TESTING ========== #

sobel_x = np.array([
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
], dtype=np.float32)

matrix = create_synth_img()
y = conv2d_naive(matrix, sobel_x)
print(y)