import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial

globalSeed = 42


dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info = True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes 

test_set_raw, valid_set_raw, train_set_raw = tfds.load("tf_flowers",
                                                       split=["train[:10%]","train[10%:25%]","train[25%:]"],
                                                       as_supervised = True)

batch_size = 32
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224,width=224,crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
train_set = train_set_raw.map(lambda x,y: (preprocess(x),y))
train_set = train_set.shuffle(1000,seed=globalSeed).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda x,y: (preprocess(x),y)).batch(batch_size)
test_set = test_set_raw.map(lambda x,y: (preprocess(x),y)).batch(batch_size)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal",seed=globalSeed),
    tf.keras.layers.RandomRotation(factor=0.05, seed=globalSeed),
    tf.keras.layers.RandomContrast(factor=0.2, seed=globalSeed)
])

base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes,activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input,outputs=output)

for layer in base_model.layers:
    layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=10)
