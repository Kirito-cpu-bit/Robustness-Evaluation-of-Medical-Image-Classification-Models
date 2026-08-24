"""Replace grouped Conv2D (depthwise) layers for DirectML GPU compatibility."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D


def _is_depthwise_conv(layer: Conv2D) -> bool:
    groups = getattr(layer, "groups", 1)
    return groups > 1 and groups == layer.filters


def _convert_depthwise_weights(conv2d_layer: Conv2D, depthwise_layer: DepthwiseConv2D):
    weights = conv2d_layer.get_weights()
    if not weights:
        return weights
    kernel, *rest = weights
    kernel = np.transpose(kernel, (0, 1, 3, 2))
    return [kernel, *rest]


def clone_depthwise_conv(layer):
    if isinstance(layer, Conv2D) and _is_depthwise_conv(layer):
        return DepthwiseConv2D(
            kernel_size=layer.kernel_size[0],
            strides=layer.strides,
            padding=layer.padding,
            use_bias=layer.use_bias,
            depth_multiplier=1,
            name=layer.name,
        )
    return layer.__class__.from_config(layer.get_config())


def patch_convnext_for_gpu(model: tf.keras.Model) -> tf.keras.Model:
    """Return a GPU-safe clone of ConvNeXt with DepthwiseConv2D layers."""
    patched = tf.keras.models.clone_model(model, clone_function=clone_depthwise_conv)
    for src, dst in zip(model.layers, patched.layers):
        if isinstance(src, Conv2D) and isinstance(dst, DepthwiseConv2D) and _is_depthwise_conv(src):
            dst.set_weights(_convert_depthwise_weights(src, dst))
        elif hasattr(src, "get_weights") and src.get_weights():
            try:
                dst.set_weights(src.get_weights())
            except ValueError:
                pass
    return patched
