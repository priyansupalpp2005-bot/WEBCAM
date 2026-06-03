from tensorflow.keras.preprocessing.image import ImageDataGenerator

script_dir = Path(__file__).resolve().parent
train_dir = script_dir / 'dataset' / 'train' 
if not train_dir.exists():
    raise FileNotFoundError(f"Train dataset folder not found: {train_dir}")

print(f"Loading images from: {train_dir}")

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)
print(train_generator.class_indices)