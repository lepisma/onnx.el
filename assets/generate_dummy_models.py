"""Script to generate dummy ONNX model files for testing.

Use something like `uv run --with torch --with onnx <script-path>`
"""

import torch
import torch.nn as nn
import torch.onnx

class MeanPoolModel(nn.Module):
    def __init__(self):
        super(MeanPoolModel, self).__init__()

    def forward(self, x):
        # Compute the mean across the second dimension (length)
        x = x.mean(dim=1)  # Mean over length dimension (shape: 20x100 -> 20)
        return x

model = MeanPoolModel()
model.eval()

dummy_input = torch.randn(20, 100)

onnx_file_path = "mean_20x100_to_20.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,
    opset_version=11
)

print(f"ONNX model has been saved to {onnx_file_path}")
