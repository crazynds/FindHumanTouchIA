import tensorflow as tf
import tensorflow_datasets as tfds
import googlemaps_images
import matplotlib.pyplot as plt
from functools import partial
import random
import numpy as np
import os

globalSeed = 42

_, info = tfds.load("googlemaps_images", as_supervised=True, with_info = True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes 

valid_set_raw, train_set_raw = tfds.load("googlemaps_images",
                                                       split=["train[:15%]","train[15%:]"],
                                                       as_supervised = True)

batch_size = 8
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=256,width=256,crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])
train_set = train_set_raw.map(lambda x,y: (preprocess(x),y))
train_set = train_set.shuffle(1000,seed=globalSeed).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda x,y: (preprocess(x),y)).batch(batch_size)


base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes,activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input,outputs=output)

for layer in base_model.layers:
    layer.trainable = False


print(dataset_size)
print(class_names)
print(n_classes)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam', metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=10)



l = list(valid_set_raw)
for _ in range(4):
    i = random.randint(0,len(l)-1)
    data = l[i][0]
    plt.imshow(data)
    plt.show()

    out = model.predict(np.array([data,]))[0].tolist()
    idx = out.index(max(out))
    print(class_names[idx])


