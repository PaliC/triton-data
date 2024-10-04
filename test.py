import torch
from PIL import Image
import requests
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification


# def my_function(x, y):
#     return torch.sin(torch.cos(x + y))

# Or use a PyTorch model
# model = YourPyTorchModel()
# optimized_function = torch.compile(my_function)
# Or for a model:
# optimized_model = torch.compile(model)
# x = torch.randn(1000, device="cuda")
# y = torch.randn(1000, device="cuda")
# result = optimized_function(x, y)
# model = timm.create_model('gluon_inception_v3', pretrained=True)
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to("cuda")
model = torch.compile(model)

processed_input = processor(image, return_tensors='pt').to(device="cuda")

with torch.no_grad():
    _ = model(**processed_input)
