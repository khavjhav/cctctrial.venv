import numpy as np

# Create dummy input data
input_data = np.random.randn(1, 3, 416, 416).astype(np.float32)

# Run inference
output = sparse_model.run([input_data])

# Process the output
# ... (your code to process the output)
