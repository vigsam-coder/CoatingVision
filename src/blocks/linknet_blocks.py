import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from .common_blocks import Conv3x3BnReLU, Conv1x1BnReLU

def UpsampleBlock(filters, stage, use_batchnorm):
    def block(x, skip=None):
        x = Conv1x1BnReLU(x.shape[-1] // 4, use_batchnorm, name=f'stage{stage}_conv1')(x)
        x = layers.UpSampling2D((2, 2), name=f'stage{stage}_upsample')(x)
        x = Conv3x3BnReLU(x.shape[-1], use_batchnorm, name=f'stage{stage}_conv2')(x)
        x = Conv1x1BnReLU(filters if skip is None else skip.shape[-1], use_batchnorm, name=f'stage{stage}_conv3')(x)

        if skip is not None:
            x = layers.Add(name=f'stage{stage}_add')([x, skip])
        return x

    return block

def TransposeBlock(filters, stage, use_batchnorm):
    def block(x, skip=None):
        x = Conv1x1BnReLU(x.shape[-1] // 4, use_batchnorm, name=f'stage{stage}_conv1')(x)
        x = layers.Conv2DTranspose(
            filters=x.shape[-1],
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=not use_batchnorm,
            name=f'stage{stage}_deconv'
        )(x)
        if use_batchnorm:
            x = layers.BatchNormalization(axis=3, name=f'stage{stage}_bn')(x)
        x = layers.Activation('relu', name=f'stage{stage}_relu')(x)
        x = Conv1x1BnReLU(filters if skip is None else skip.shape[-1], use_batchnorm, name=f'stage{stage}_conv2')(x)

        if skip is not None:
            x = layers.Add(name=f'stage{stage}_add')([x, skip])
        return x

    return block

def build_linknet_model(backbone, decoder_block, skip_layers, decoder_filters, num_classes, activation, use_batchnorm):
    inputs = backbone.input
    x = backbone.output

    skips = [backbone.get_layer(name).output if isinstance(name, str) else backbone.layers[name].output
             for name in skip_layers]

    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center2')(x)

    for i, filters in enumerate(decoder_filters):
        skip = skips[i] if i < len(skips) else None
        x = decoder_block(filters, stage=i, use_batchnorm=use_batchnorm)(x, skip)

    x = layers.Conv2D(num_classes, kernel_size=3, padding='same', name='final_conv')(x)
    x = layers.Activation(activation, name=activation)(x)

    return models.Model(inputs, x)
