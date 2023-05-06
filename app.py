import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

print(tf.version.VERSION)
IMAGENET_TEST_IMAGE = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
UNZIP_PATH = tf.keras.utils.get_file("cats_and_dogs.zip", origin=IMAGENET_TEST_IMAGE, extract=True)
PATH = os.path.join(os.path.dirname(UNZIP_PATH), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

BATCH_SIZE = 32
IMAGE_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, 
    shuffle=True, 
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


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