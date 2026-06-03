import joblib

model = joblib.load("gesture_model.pkl")
encoder = joblib.load("label_encoder.pkl")

print("Classes:")
print(encoder.classes_)

print("\nNumber of classes:")
print(len(encoder.classes_))