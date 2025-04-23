from blocks.linknet_blocks import build_linknet_model, UpsampleBlock, TransposeBlock
from backbones.keras_encoders import Backbones

def LinkNet(
        backbone_name='resnet50',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        encoder_weights='imagenet',
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(None, None, None, None, 16),
        decoder_use_batchnorm=True,
        **kwargs
):
    if decoder_block_type == 'upsampling':
        decoder_block = UpsampleBlock
    elif decoder_block_type == 'transpose':
        decoder_block = TransposeBlock
    else:
        raise ValueError(f"Invalid decoder_block_type: {decoder_block_type}")

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name)

    model = build_linknet_model(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_layers=encoder_features,
        decoder_filters=decoder_filters,
        num_classes=classes,
        activation=activation,
        use_batchnorm=decoder_use_batchnorm,
    )

    return model

