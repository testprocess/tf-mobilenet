import matplotlib.pyplot as plt
import os
import tensorflow as tf
import dataset as tfdataset

print(tf.version.VERSION)
tf.config.experimental.list_physical_devices('GPU')


class Train():
    def __init__(self):
        data = tfdataset.Dataset()
        (self.train_dataset, self.validation_dataset) = data.download_data()

        self.epochs = 10

        self.class_names = self.train_dataset.class_names
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])

        self.run()


    def run(self):
        self.model()
        self.fit()
        self.show_accuracy()

    def model(self):
        val_batches = tf.data.experimental.cardinality(self.validation_dataset)
        test_dataset = self.validation_dataset.take(val_batches // 5)
        self.validation_dataset = self.validation_dataset.skip(val_batches // 5)

        self.model = tf.keras.models.Sequential([
            tf.keras.applications.MobileNetV3Small(
                input_shape=None,
                alpha=1.0,
                minimalistic=False,
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                classes=1000,
                pooling=None,
                dropout_rate=0.2,
                classifier_activation='softmax',
                include_preprocessing=True
            )
        ])

        self.model.training = False

        image_batch, label_batch = next(iter(self.train_dataset))
        feature_batch = self.model(image_batch)
        print(feature_batch.shape)

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print(feature_batch_average.shape)

        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)
        print(prediction_batch.shape)

        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = self.data_augmentation(inputs)
        x = self.preprocess_input(x)
        x = self.model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)


    def fit(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.validation_dataset,
        )


    def show_accuracy(self):
        loss0, accuracy0 = self.model.evaluate(self.validation_dataset)
        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()


train = Train()
