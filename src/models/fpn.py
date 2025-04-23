from backbones.keras_encoders import Backbones
from blocks.fpn_blocks import build_fpn_model

def FPN(backbone_name='resnet50',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        encoder_weights='imagenet',
        encoder_features='default',
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_aggregation='sum',
        pyramid_dropout=None,
        **kwargs):

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name)

    model = build_fpn_model(
        backbone=backbone,
        skip_layers=encoder_features,
        pyramid_filters=pyramid_block_filters,
        segmentation_filters=pyramid_block_filters // 2,
        num_classes=classes,
        activation=activation,
        use_batchnorm=pyramid_use_batchnorm,
        aggregation=pyramid_aggregation,
        dropout_rate=pyramid_dropout
    )

    return model

