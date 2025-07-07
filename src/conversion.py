import onnx
import tf2onnx
from tensorflow.keras.models import load_model

# Load your Keras model
model = load_model('models/lstm_next_day.h5', compile=False)

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, output_path='models/lstm_next_day.onnx')
print("Model converted to ONNX format")
