"""ResNet18 backbone compatible with qubvel/classification_models ImageNet weights."""
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    ZeroPadding2D,
)
from tensorflow.keras.utils import get_file

from train_resnet50 import INPUT_SHAPE

# qubvel/classification_models release 0.0.1 (1.0.0 URLs are broken).
WEIGHTS_URL = (
    "https://github.com/qubvel/classification_models/releases/download/0.0.1/"
    "resnet18_imagenet_1000_no_top.h5"
)
WEIGHTS_MD5 = "318e3ac0cd98d51e917526c9f62f0b50"
WEIGHTS_FILENAME = "resnet18_imagenet_1000_no_top.h5"


def _conv_params():
    return {
        "kernel_initializer": "he_uniform",
        "use_bias": False,
        "padding": "valid",
    }


def _bn_params():
    return {
        "axis": 3,
        "momentum": 0.99,
        "epsilon": 2e-5,
        "center": True,
        "scale": True,
    }


def _block_names(stage, block):
    name_base = f"stage{stage + 1}_unit{block + 1}_"
    return (
        name_base + "conv",
        name_base + "bn",
        name_base + "relu",
        name_base + "sc",
    )


def _residual_conv_block(filters, stage, block, strides=(1, 1), cut="pre"):
    conv_name, bn_name, relu_name, sc_name = _block_names(stage, block)
    conv_params = _conv_params()
    bn_params = _bn_params()

    def block_fn(input_tensor):
        x = BatchNormalization(name=bn_name + "1", **bn_params)(input_tensor)
        x = Activation("relu", name=relu_name + "1")(x)

        if cut == "pre":
            shortcut = input_tensor
        elif cut == "post":
            shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError(f'Unsupported cut type: {cut}')

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + "1", **conv_params)(x)
        x = BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = Activation("relu", name=relu_name + "2")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + "2", **conv_params)(x)
        return Add()([x, shortcut])

    return block_fn


def _build_resnet18(input_shape, include_top=False, classes=1000):
    """Match qubvel ResNet18 layer names for weight loading."""
    img_input = Input(shape=input_shape, name="data")
    no_scale_bn = {**_bn_params(), "scale": False}

    x = BatchNormalization(name="bn_data", **no_scale_bn)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), name="conv0", **_conv_params())(x)
    x = BatchNormalization(name="bn0", **_bn_params())(x)
    x = Activation("relu", name="relu0")(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="pooling0")(x)

    repetitions = (2, 2, 2, 2)
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            filters = 64 * (2**stage)
            if block == 0 and stage == 0:
                x = _residual_conv_block(filters, stage, block, strides=(1, 1), cut="post")(x)
            elif block == 0:
                x = _residual_conv_block(filters, stage, block, strides=(2, 2), cut="post")(x)
            else:
                x = _residual_conv_block(filters, stage, block, strides=(1, 1), cut="pre")(x)

    x = BatchNormalization(name="bn1", **_bn_params())(x)
    x = Activation("relu", name="relu1")(x)

    if include_top:
        x = GlobalAveragePooling2D(name="pool1")(x)
        x = Dense(classes, name="fc1")(x)
        x = Activation("softmax", name="softmax")(x)

    return Model(img_input, x, name="resnet18")


def _download_weights():
    return get_file(
        WEIGHTS_FILENAME,
        WEIGHTS_URL,
        cache_subdir="models",
        md5_hash=WEIGHTS_MD5,
    )


def _load_imagenet_weights(model):
    weights_path = _download_weights()
    model.load_weights(weights_path)
    print(f"Loaded ResNet18 ImageNet weights from: {WEIGHTS_URL}", flush=True)


def ResNet18(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE):
    """ResNet18 feature extractor with qubvel-compatible ImageNet weights."""
    model = _build_resnet18(input_shape, include_top=include_top)
    if weights == "imagenet":
        _load_imagenet_weights(model)
    elif weights is not None:
        raise ValueError(f"Unsupported weights: {weights}")
    return model
