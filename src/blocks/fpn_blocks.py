import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from .common_blocks import Conv3x3BnReLU
def double_conv_block(x, filters, use_batchnorm, name_prefix):
    x = Conv3x3BnReLU(filters, use_batchnorm, name=name_prefix + "_1")(x)
    x = Conv3x3BnReLU(filters, use_batchnorm, name=name_prefix + "_2")(x)
    return x

def fpn_block(x, skip, filters, stage):
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, padding='same', name=f'fpn_p{stage}_input_conv')(x)

    skip = layers.Conv2D(filters, 1, padding='same', name=f'fpn_p{stage}_skip_conv')(skip)

    x = layers.UpSampling2D(size=(2, 2), name=f'fpn_p{stage}_upsample')(x)
    x = layers.Add(name=f'fpn_p{stage}_add')([x, skip])
    return x

def build_fpn_model(backbone, skip_layers, pyramid_filters, segmentation_filters,
                    num_classes, activation, use_batchnorm, aggregation, dropout_rate):

    inputs = backbone.input
    x = backbone.output
    skips = [backbone.get_layer(name=layer).output if isinstance(layer, str) else
             backbone.get_layer(index=layer).output for layer in skip_layers]

    # Build FPN blocks
    p5 = fpn_block(x, skips[0], pyramid_filters, stage=5)
    p4 = fpn_block(p5, skips[1], pyramid_filters, stage=4)
    p3 = fpn_block(p4, skips[2], pyramid_filters, stage=3)
    p2 = fpn_block(p3, skips[3], pyramid_filters, stage=2)

    # Apply segmentation head to each
    s5 = double_conv_block(p5, segmentation_filters, use_batchnorm, "segm5")
    s4 = double_conv_block(p4, segmentation_filters, use_batchnorm, "segm4")
    s3 = double_conv_block(p3, segmentation_filters, use_batchnorm, "segm3")
    s2 = double_conv_block(p2, segmentation_filters, use_batchnorm, "segm2")

    # Upsample to same size
    s5 = layers.UpSampling2D(size=8, name='up_s5')(s5)
    s4 = layers.UpSampling2D(size=4, name='up_s4')(s4)
    s3 = layers.UpSampling2D(size=2, name='up_s3')(s3)

    # Merge features
    if aggregation == 'sum':
        x = layers.Add(name='merge_sum')([s2, s3, s4, s5])
    elif aggregation == 'concat':
        x = layers.Concatenate(axis=-1, name='merge_concat')([s2, s3, s4, s5])
    else:
        raise ValueError("aggregation must be 'sum' or 'concat'")

    if dropout_rate:
        x = layers.SpatialDropout2D(dropout_rate, name='dropout')(x)

    # Final layers
    x = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='final_conv')(x)
    x = layers.UpSampling2D(size=2, interpolation='bilinear', name='final_upsample')(x)

    x = layers.Conv2D(num_classes, kernel_size=3, padding='same', name='output_conv')(x)
    x = layers.Activation(activation, name='output_activation')(x)

    return models.Model(inputs, x)