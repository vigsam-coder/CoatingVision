import tensorflow.keras.layers as layers


def Conv2dBn(filters, kernel_size, activation=None, use_batchnorm=False, **kwargs):
    """Conv2D layer with optional batch normalization"""

    def wrapper(input_tensor):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None,
            use_bias=not use_batchnorm,
            padding='same',
            **kwargs
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization()(x)

        if activation:
            x = layers.Activation(activation)(x)

        return x

    return wrapper


def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    """Convenience function for 3x3 Conv2D with BatchNorm and ReLU"""
    return Conv2dBn(filters, 3, activation='relu', use_batchnorm=use_batchnorm, name=name)


def Conv1x1BnReLU(filters, use_batchnorm, name=None):
    """Convenience function for 1x1 Conv2D with BatchNorm and ReLU"""
    return Conv2dBn(filters, 1, activation='relu', use_batchnorm=use_batchnorm, name=name)
