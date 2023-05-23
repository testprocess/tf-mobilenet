
import matplotlib.pyplot as plt
import os
import tensorflow as tf


class Dataset():
    def __init__(self):
        self.imagenet_test = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        self.unzip_path = tf.keras.utils.get_file("cats_and_dogs.zip", origin=self.imagenet_test, extract=True)
        self.path = os.path.join(os.path.dirname(self.unzip_path), 'cats_and_dogs_filtered')

        self.train_dir = os.path.join(self.path, "train")
        self.validation_dir = os.path.join(self.path, "validation")

        self.batch_size = 32
        self.image_size = (160, 160)


    def download_data(self):
        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.train_dir, 
            shuffle=True, 
            batch_size=self.batch_size,
            image_size=self.image_size
        )

        self.validation_dataset = tf.keras.utils.image_dataset_from_directory(
            self.validation_dir,
            shuffle=True,
            batch_size=self.batch_size,
            image_size=self.image_size
        )

        self.class_names = self.train_dataset.class_names


        return (self.train_dataset, self.validation_dataset)

    def show_plot(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i+1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
        #plt.show()