
from blocks.pspnet_blocks import build_psp
from backbones.keras_encoders import Backbones

def PSPNet(backbone_name='resnet50',
           input_shape=(384, 384, 3),
           classes=21, activation='softmax',
           encoder_weights='imagenet',
           downsample_factor=8,
           psp_conv_filters=512,
           psp_pooling_type='avg',
           psp_use_batchnorm=True,
           psp_dropout=None, **kwargs):
    """PSPNet for image semantic segmentation using a backbone model."""

    # Load backbone model
    backbone = Backbones.get_backbone(backbone_name, input_shape=input_shape, weights=encoder_weights,
                                      include_top=False, **kwargs)
    feature_layers = Backbones.get_feature_layers(backbone_name)

    # Determine the PSP layer based on downsample factor
    if downsample_factor == 16:
        psp_layer_idx = feature_layers[0]
    elif downsample_factor == 8:
        psp_layer_idx = feature_layers[1]
    elif downsample_factor == 4:
        psp_layer_idx = feature_layers[2]
    else:
        raise ValueError(f"Unsupported downsample factor: {downsample_factor}. Use 4, 8, or 16.")

    # Build PSP model
    model = build_psp(
        backbone,
        psp_layer_idx,
        pooling_type=psp_pooling_type,
        conv_filters=psp_conv_filters,
        use_batchnorm=psp_use_batchnorm,
        final_upsampling_factor=downsample_factor,
        classes=classes,
        activation=activation,
        dropout=psp_dropout
    )
    return model
