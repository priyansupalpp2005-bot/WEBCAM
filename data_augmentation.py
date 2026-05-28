import tensorflow as tf

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = datagen.flow_from_directory(
    "dataset/train",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = datagen.flow_from_directory(
    "dataset/test",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical"
)
