import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# All 15 labels
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
    'Hernia', 'No Finding'
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load CSV
print("Loading dataset...")
df = pd.read_csv('Data_Entry_2017.csv')
df = df[['Image Index', 'Finding Labels']]

# Create multi-label columns
for label in LABELS:
    df[label] = df['Finding Labels'].apply(
        lambda x: 1 if label in x.split('|') else 0)

# Find all images
image_folders = [f'images_{str(i).zfill(3)}' for i in range(1, 13)]

def find_image(filename):
    for folder in image_folders:
        path = os.path.join(folder, 'images', filename)
        if os.path.exists(path):
            return path
    return None

print("Finding image paths...")
df['path'] = df['Image Index'].apply(find_image)
df = df.dropna(subset=['path'])
print(f"Found {len(df)} images")

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Data generator
def load_batch(batch_df):
    images = []
    labels = []
    for _, row in batch_df.iterrows():
        try:
            img = Image.open(row['path']).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append([row[l] for l in LABELS])
        except:
            continue
    return np.array(images), np.array(labels)

def data_generator(dataframe, batch_size):
    while True:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(dataframe), batch_size):
            batch = dataframe.iloc[i:i+batch_size]
            X, y = load_batch(batch)
            if len(X) > 0:
                yield X, y

# Build model
print("Building model...")
base_model = DenseNet121(weights='imagenet', include_top=False,
    input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(LABELS), activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.summary()

# Train
steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(test_df) // BATCH_SIZE

print("\nStarting training...")
history = model.fit(
    data_generator(train_df, BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=data_generator(test_df, BATCH_SIZE),
    validation_steps=validation_steps
)

# Save
os.makedirs('model', exist_ok=True)
model.save('model/xray_nih_model.h5')
np.save('model/labels.npy', np.array(LABELS))
print("\nModel saved to model/xray_nih_model.h5")

# Plot accuracy
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()
plt.savefig('model/nih_accuracy.png')
print("Accuracy graph saved!")