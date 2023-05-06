import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

print(tf.version.VERSION)

model = tf.keras.models.Sequential([
    tf.keras.applications.MobileNetV3Small(
        input_shape=None,
        alpha=1.0,
        minimalistic=False,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        classes=1000,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation='softmax',
        include_preprocessing=True
    )
])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
# )

# model.fit(
#     ds_train,
#     epochs=6,
#     validation_data=ds_test,
# )