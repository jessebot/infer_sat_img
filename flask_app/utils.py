"""
This module was provided by Overstory,
but then cleaned up for style and comments by Jesse Hitch

# UNET MODEL
# https://github.com/jaxony/unet-pytorch/blob/master/model.py
"""
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from os import path
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


MODEL_PATH = './live_model.pickle'
plt.rcParams["figure.figsize"] = (10, 10)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)

        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """
    `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64,
                 up_mode='transpose', merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError(f'"{up_mode}" is not a valid mode for upsampling.'
                             'Only "transpose" and "upsample" are allowed.')

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError(f'"{merge_mode}" is not a valid mode for merging '
                             'up and down paths. Only "concat" and "add" are '
                             'allowed.')

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError('up_mode "upsample" is incompatible with '
                             'merge_mode "add" at the moment, because it does'
                             'not make sense to use nearest neighbour to '
                             'reduce depth channels (by half).')

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # no activation on the output
        # x = self.conv_final(x)
        # maybe add the sigmoid here instead of postprocess
        x = F.relu(self.conv_final(x))

        return x


# UTILS
def _ensure_opened(ds):
    """
    Ensure that `ds` is an opened Rasterio dataset & not a str/pathlike object
    """
    return ds if type(ds) == rasterio.io.DatasetReader else rasterio.open(str(ds), "r")


def read_crop(ds, crop, bands=None, pad=False):
    """
    Read rasterio `crop` for the given `bands`..
    Args:
        ds: Rasterio dataset.
        crop: Tuple or list containing the area to be cropped (px, py, w, h).
        bands: List of `bands` to read from the dataset.
    Returns:
        A numpy array containing the read image `crop` (bands * h * w).
    """
    ds = _ensure_opened(ds)
    if pad:
        raise ValueError('padding not implemented yet.')

    if bands is None:
        bands = [i for i in range(1, ds.count+1)]

    # assert len(bands) <= ds.count, "`bands` cannot contain more bands than the number of bands in the dataset."
    # assert max(bands) <= ds.count, "The maximum value in `bands` should be smaller or equal to the band count."

    window = None

    if crop is not None:
        assert len(crop) == 4, "`crop` should be a tuple or list of shape (px, py, w, h)."
        px, py, w, h = crop
        log.info(f"px:{px} py:{py} w:{w} h:{h}")
        w = ds.width - px if (px + w) > ds.width else w
        h = ds.height - py if (py + h) > ds.height else h
        assert (px + w) <= ds.width, "The crop (px + w) is larger than the dataset width."
        assert (py + h) <= ds.height, "The crop (py + h) is larger than the dataset height."
        window = rasterio.windows.Window(px, py, w, h)

    meta = ds.meta
    meta.update(count=len(bands))
    if crop is not None:
        meta.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, ds.transform)})
    return ds.read(bands, window=window), meta


def plot_rgb(img, clip_percentile=(2, 98), clip_values=None, bands=[3, 2, 1],
             figsize=(20, 20), nodata=None, figtitle=None, crop=None, ax=None):
    """
    Plot clipped (and optionally cropped) RGB image.
    Args:
        img: Path to image, rasterio dataset or numpy array of shape (bands, height, width).
        clip_percentile: (min percentile, max percentile) to use for clippping.
        clip_values: (min value, max value) to use for clipping (if set clip_percentile is ignored).
        bands: Bands to use as RGB values (starting at 1).
        figsize: Size of the matplotlib figure.
        figtitle: Title to use for the figure (if None and img is a path we will use the image filename).
        crop: Window to use to crop the image (px, py, w, h).
        ax: If not None, use this Matplotlib axis for plotting.
    Returns:
        A matplotlib figure.
    """
    meta = None

    if isinstance(img, str):
        assert path.exists(img), "{} does not exist!".format(img)
        figtitle = path.basename(img) if figtitle is None else figtitle
        img = rasterio.open(img)
        img, meta = read_crop(img, crop, bands)

    elif isinstance(img, rasterio.io.DatasetReader):
        img, meta = read_crop(img, crop, bands)

    elif isinstance(img, np.ndarray):
        assert len(img.shape) <= 3, "Array should have no more than 3 dimensions."
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        elif img.shape[0] > 3:
            img = img[np.array(bands) - 1, :, :]
        if crop is not None:
            img = img[:, py:py+h, px:px+w]

    else:
        raise ValueError(f"img should be str, rasterio dataset or numpy array. (got {type(img)})")

    img = img.astype(float)
    nodata = nodata if nodata is not None else (meta['nodata'] if meta is not None else None)
    if nodata is not None:
        img[img == nodata] = np.nan

    if clip_values is not None:
        assert len(clip_values) == 2, "Clip values should have the shape (min value, max value)"
        assert clip_values[0] < clip_values[1], "clip_values[0] should be smaller than clip_values[1]"

    elif clip_percentile is not None:
        assert len(clip_percentile) == 2, "Clip_percentile should have the shape (min percentile, max percentile)"
        assert clip_percentile[0] < clip_percentile[1], "clip_percentile[0] should be smaller than clip_percentile[1]"
        clip_values = None if clip_percentile == (0, 100) else [np.nanpercentile(img, clip_percentile[i]) for i in range(2)]

    if clip_values is not None:
        img[~np.isnan(img)] = np.clip(img[~np.isnan(img)], *clip_values)
    clip_values = (np.nanmin(img), np.nanmax(img)) if clip_values is None else clip_values
    img[~np.isnan(img)] = (img[~np.isnan(img)] - clip_values[0])/(clip_values[1] - clip_values[0])

    if img.shape[0] <= 3:
        img = np.transpose(img, (1, 2, 0))
    alpha = np.all(~np.isnan(img), axis=2)[:,:,np.newaxis].astype(float)
    img = np.concatenate((img, alpha), axis=2)

    if not ax:
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(figtitle) if figtitle is not None else None
        ax.imshow(img)
        plt.close()
        return figure
    else:
        ax.imshow(img)


def tif_to_image(path, crop, bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    this was unused, so I removed it:
    tile_size = 512  # size model is trained on
    """
    ds = _ensure_opened(path)
    log.info(f"Shape of image we recieved is: {ds.read().shape}")
    image, meta = read_crop(ds, crop, bands=bands)
    return image, meta


def infer_image(file_path, plot=False, use_gpu=False):
    """
    same but dont show, only load into numpy array so we can predict on it
    """
    ds = _ensure_opened(file_path)
    image = ds.read()
    log.info(f"image shape is: {str(image.shape)}")

    # data normalization
    inputs = image[:10, :, :].astype(float)

    # ugly rescaling
    for band in range(inputs.shape[0]):
        # compute 90th percentile
        perc = np.percentile(inputs[band, :, :], 90)
        if perc > 0:
            inputs[band, :, :][inputs[band, :, :] > perc] = perc
            inputs[band, :, :] = inputs[band, :, :] / perc

    model = UNet(num_classes=1, in_channels=10, depth=5,
                 start_filts=16, up_mode='transpose',
                 merge_mode='concat')

    if use_gpu:
        log.info(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        log.info(f"torch.cuda.get_device_name: {torch.cuda.get_device_name()}")
        device = 'cuda'
    else:
        device = 'cpu'

    log.info(f"Torch device should be: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    log.info("Beginning tensor work...")
    tensor = torch.tensor(np.expand_dims(inputs, 0)).float()

    log.info(tensor.device)
    res = model.forward(tensor)

    log.info("Starting numpy reshape...")
    res = res.detach().numpy().reshape(image.shape[1], image.shape[2])
    res[res > 0.5] = 1
    res[res <= 0.5] = 0

    if plot:
        fig, axs = plt.subplots(2)
        axs[0].imshow(np.transpose(inputs[:3, :, :], (1, 2, 0)))
        axs[1].imshow(res, alpha=0.5, cmap='gray', vmin=0, vmax=1)

    return res
