import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

TRAIN_DIR = 'chest_xray/train'
TEST_DIR = 'chest_xray/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10,
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(TRAIN_DIR,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
test_data = test_gen.flow_from_directory(TEST_DIR,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

base_model = VGG16(weights='imagenet', include_top=False,
    input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

model.save('model/xray_model.h5')
print("\nModel saved successfully!")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.legend()
plt.savefig('model/accuracy.png')
print("Accuracy graph saved!")