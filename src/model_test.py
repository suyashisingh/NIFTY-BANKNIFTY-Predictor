from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

try:
    model = load_model('models/lstm_next_day.h5', custom_objects={'mse': MeanSquaredError()})
    print("✅ Model loaded successfully!")
    print(f"Model summary: {model.summary()}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
