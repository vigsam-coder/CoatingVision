from backbones.keras_encoders import Backbones
from blocks.unet_blocks import build_unet_model,upsampling_decoder_block,transpose_decoder_block

def Unet(
    backbone_name="resnet50",
    input_shape=(None, None, 3),
    classes=1,
    activation="sigmoid",
    encoder_weights="imagenet",
    encoder_features="default",
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
    decoder_use_batchnorm=True,
    **kwargs
):
    if decoder_block_type == "upsampling":
        decoder_block = upsampling_decoder_block
    elif decoder_block_type == "transpose":
        decoder_block = transpose_decoder_block
    else:
        raise ValueError(f"Unsupported decoder block: {decoder_block_type}")

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs
    )

    if encoder_features == "default":
        encoder_features = Backbones.get_feature_layers(backbone_name)

    model = build_unet_model(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        num_upsample=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm
    )
    return model