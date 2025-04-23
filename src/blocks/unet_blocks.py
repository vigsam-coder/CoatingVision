import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from .common_blocks import Conv3x3BnReLU
from backbones.keras_encoders import Backbones


# Decoder block using UpSampling2D
def upsampling_decoder_block(filters, stage, use_batchnorm=False):
    def block(x, skip=None):
        x = layers.UpSampling2D(size=(2, 2), name=f"upsample_{stage}")(x)
        if skip is not None:
            x = layers.Concatenate(name=f"concat_{stage}")([x, skip])
        x = Conv3x3BnReLU(filters, use_batchnorm, name=f"conv_{stage}_1")(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=f"conv_{stage}_2")(x)
        return x
    return block

# Decoder block using Conv2DTranspose
def transpose_decoder_block(filters, stage, use_batchnorm=False):
    def block(x, skip=None):
        x = layers.Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding="same",
                                   use_bias=not use_batchnorm, name=f"transpose_{stage}")(x)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"bn_{stage}")(x)
        x = layers.Activation("relu", name=f"relu_{stage}")(x)
        if skip is not None:
            x = layers.Concatenate(name=f"concat_{stage}")([x, skip])
        x = Conv3x3BnReLU(filters, use_batchnorm, name=f"conv_{stage}")(x)
        return x
    return block

# Function to build the full U-Net model
def build_unet_model(
    backbone,
    decoder_block,
    skip_layers,
    decoder_filters=(256, 128, 64, 32, 16),
    num_upsample=5,
    classes=1,
    activation="sigmoid",
    use_batchnorm=True
):
    inputs = backbone.input
    x = backbone.output

    skips = [backbone.get_layer(name=s).output if isinstance(s, str) else backbone.layers[s].output
             for s in skip_layers]

    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center2')(x)

    for i in range(num_upsample):
        skip = skips[i] if i < len(skips) else None
        x = decoder_block(decoder_filters[i], i, use_batchnorm)(x, skip)

    x = layers.Conv2D(classes, (3, 3), padding="same", name="final_conv")(x)
    x = layers.Activation(activation, name="final_activation")(x)

    return models.Model(inputs, x)

