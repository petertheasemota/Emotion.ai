# ===========================
# Step 1: Import Dependencies
# ===========================
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from google.colab import files

# ===========================
# Step 2: Load Pretrained Model
# ===========================
# Use MobileNetV2 as a feature extractor
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers (no training)
for layer in base_model.layers:
    layer.trainable = False

# ===========================
# Step 3: Add Custom Layers
# ===========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
# For demonstration, 7 emotion classes (can be changed later)
preds = Dense(7, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=preds)

# ===========================
# Step 4: Compile Model
# ===========================
model.compile(optimizer=Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===========================
# Step 5: Save Model
# ===========================
model.save('emotion_model.h5')
print("✅ Model exported successfully as emotion_model.h5")

# ===========================
# Step 6: Download to Local Machine
# ===========================
files.download('emotion_model.h5')
