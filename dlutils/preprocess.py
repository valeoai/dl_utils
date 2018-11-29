import numpy as np

def unpreprocess(x, data_format,mode):
    """unpreprocesses a Numpy array encoding a batch of images.
    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        unreprocessed Numpy array.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if mode == 'tf':
        x += 1.
        x *= 127.5
        np.clip(x / 225., 0, 1)
        x = x.astype(int)
        return x

    if mode == 'torch':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            if std is not None:
                x[0, :, :] *= std[0]
                x[1, :, :] *= std[1]
                x[2, :, :] *= std[2]
                
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]

        else:
            if std is not None:
                x[:, 0, :, :] *= std[0]
                x[:, 1, :, :] *= std[1]
                x[:, 2, :, :] *= std[2]
                
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]

    else:
        if std is not None:
            x[..., 0] *= std[0]
            x[..., 1] *= std[1]
            x[..., 2] *= std[2]        
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if mode == 'torch':
        x *= 255.
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            
    return x
