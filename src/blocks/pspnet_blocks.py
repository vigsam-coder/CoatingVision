import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from .common_blocks import Conv1x1BnReLU

def SpatialContextBlock(level, conv_filters=512, pooling_type='avg', use_batchnorm=True):
    if pooling_type not in ('max', 'avg'):
        raise ValueError(f"Unsupported pooling type - `{pooling_type}`. Use `avg` or `max`.")

    Pooling2D = layers.MaxPool2D if pooling_type == 'max' else layers.AveragePooling2D
    pool_size = up_size = [level, level]

    def blocks(input_tensor):
        # Pooling and upsampling for spatial pyramid levels
        x = Pooling2D(pool_size, strides=pool_size, padding='same')(input_tensor)
        x = Conv1x1BnReLU(conv_filters, use_batchnorm)(x)
        x = layers.UpSampling2D(up_size, interpolation='bilinear')(x)
        return x

    return blocks


def build_psp(backbone, psp_layer_idx, pooling_type='avg', conv_filters=512, use_batchnorm=True,
              final_upsampling_factor=8, classes=21, activation='softmax', dropout=None):
    input_ = backbone.input
    x = backbone.get_layer(psp_layer_idx).output

    # Build spatial pyramid context blocks
    x1 = SpatialContextBlock(1, conv_filters, pooling_type, use_batchnorm)(x)
    x2 = SpatialContextBlock(2, conv_filters, pooling_type, use_batchnorm)(x)
    x3 = SpatialContextBlock(3, conv_filters, pooling_type, use_batchnorm)(x)
    x6 = SpatialContextBlock(6, conv_filters, pooling_type, use_batchnorm)(x)

    # Concatenate spatial pyramid outputs and apply aggregation
    x = layers.Concatenate(axis=3)([x, x1, x2, x3, x6])
    x = Conv1x1BnReLU(conv_filters, use_batchnorm)(x)

    # Apply dropout if specified
    if dropout is not None:
        x = layers.SpatialDropout2D(dropout)(x)

    # Final convolution and upsampling to output segmentation map
    x = layers.Conv2D(classes, (3, 3), padding='same')(x)
    x = layers.UpSampling2D(final_upsampling_factor, interpolation='bilinear')(x)
    x = layers.Activation(activation)(x)

    model = models.Model(input_, x)
    return model


