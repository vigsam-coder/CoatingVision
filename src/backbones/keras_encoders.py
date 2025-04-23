import tensorflow as tf

class Backbones:
    BACKBONE_DICT = {
        "resnet50": tf.keras.applications.ResNet50,
        "resnet101": tf.keras.applications.ResNet101,
        "resnet152": tf.keras.applications.ResNet152,
        "convnextbase": tf.keras.applications.ConvNeXtBase,
        "convnextlarge": tf.keras.applications.ConvNeXtLarge,
        "convnextsmall": tf.keras.applications.ConvNeXtSmall,
        "convnexttiny": tf.keras.applications.ConvNeXtTiny,
        "convnextxlarge": tf.keras.applications.ConvNeXtXLarge,
        "efficientnetv2b0": tf.keras.applications.EfficientNetV2B0,
        "efficientnetv2b1": tf.keras.applications.EfficientNetV2B1,
        "efficientnetv2b2": tf.keras.applications.EfficientNetV2B2,
        "efficientnetv2b3": tf.keras.applications.EfficientNetV2B3,
        "efficientnetv2l": tf.keras.applications.EfficientNetV2L,
        "efficientnetv2m": tf.keras.applications.EfficientNetV2M,
        "efficientnetv2s": tf.keras.applications.EfficientNetV2S,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
        "efficientnetb1": tf.keras.applications.EfficientNetB1,
        "efficientnetb2": tf.keras.applications.EfficientNetB2,
        "efficientnetb3": tf.keras.applications.EfficientNetB3,
        "efficientnetb4": tf.keras.applications.EfficientNetB4,
        "efficientnetb5": tf.keras.applications.EfficientNetB5,
        "efficientnetb6": tf.keras.applications.EfficientNetB6,
        "efficientnetb7": tf.keras.applications.EfficientNetB7,
    }

    FEATURE_LAYERS = {
        "resnet50": ("conv4_block6_out", "conv3_block4_out", "conv2_block3_out", "conv1_relu"),
        "resnet101": ("conv4_block23_out", "conv3_block4_out", "conv2_block3_out", "conv1_relu"),
        "resnet152": ("conv4_block36_out", "conv3_block8_out", "conv2_block3_out", "conv1_relu"),
        "convnextbase": ("convnext_base_stage_2_block_0_gelu","convnext_base_stage_1_block_0_gelu","convnext_base_stage_0_block_0_gelu"),#,"convnext_base_stem"),
        "convnextlarge": ("convnext_large_stage_2_block_0_gelu","convnext_large_stage_1_block_0_gelu","convnext_large_stage_0_block_0_gelu"),
        "convnextsmall": ("convnext_small_stage_2_block_0_gelu","convnext_small_stage_1_block_0_gelu","convnext_small_stage_0_block_0_gelu"),
        "convnexttiny": ("convnext_tiny_stage_2_block_0_gelu","convnext_tiny_stage_1_block_0_gelu","convnext_tiny_stage_0_block_0_gelu"),
        "convnextxlarge": ("convnext_xlarge_stage_2_block_0_gelu","convnext_xlarge_stage_1_block_0_gelu","convnext_xlarge_stage_0_block_0_gelu"),
        "efficientnetv2b0": ('block6a_expand_activation', 'block4a_expand_activation',
                           'block2b_expand_activation', 'stem_activation'),
        "efficientnetv2b1": ('block6a_expand_activation', 'block4a_expand_activation',
                           'block2b_expand_activation', 'stem_activation'),
        "efficientnetv2b2": ('block6a_expand_activation', 'block4a_expand_activation',
                           'block2b_expand_activation', 'stem_activation'),
        "efficientnetv2b3": ('block6a_expand_activation', 'block4a_expand_activation',
                           'block2b_expand_activation', 'stem_activation'),
        "efficientnetv2l": ('block6a_expand_activation', 'block4a_expand_activation',
                           'block2b_expand_activation', 'stem_activation'),
        "efficientnetv2m": ('block6a_expand_activation', 'block4a_expand_activation',
                           'block2b_expand_activation', 'stem_activation'),
        "efficientnetv2s": ('block6a_expand_activation', 'block4a_expand_activation',
                           'block2b_expand_activation', 'stem_activation'),
        "efficientnetb0": ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb1': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb2': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb3': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb4': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb5': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb6': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb7': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
    }

    @staticmethod
    def get_backbone(backbone_name, input_shape=None, weights="imagenet", include_top=False, **kwargs):
        backbone_name = backbone_name.lower()
        if backbone_name not in Backbones.BACKBONE_DICT:
            raise ValueError(
                f"Backbone '{backbone_name}' not supported. Choose from {list(Backbones.BACKBONE_DICT.keys())}.")

        return Backbones.BACKBONE_DICT[backbone_name](
            include_top=include_top,
            weights=weights,
            input_shape=input_shape,
            **kwargs
        )


    @staticmethod
    def get_feature_layers(backbone_name):
        backbone_name = backbone_name.lower()
        if backbone_name not in Backbones.FEATURE_LAYERS:
            raise ValueError(
                f"Feature layers for '{backbone_name}' not defined. Choose from {list(Backbones.FEATURE_LAYERS.keys())}.")

        return Backbones.FEATURE_LAYERS[backbone_name]
