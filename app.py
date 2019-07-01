#!/usr/bin/python

import caffe2.python.onnx.backend as backend
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
import onnx
import sys
import time
import torch
from torchvision import datasets, transforms


PORT_NUMBER = 8080
start = time.time()

# Check for CUDA resource
if not torch.cuda.is_available():
    sys.exit('Service requires a GPU')

# Load the ONNX model
model = onnx.load("model.onnx")

print("Checking model", flush=True)
onnx.checker.check_model(model)

print("Preparing backend", flush=True)
rep = backend.prepare(model, device="CUDA:0")  # or "CPU"

print("Setup transform",  flush=True)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

print("Load MNIST dataset", flush=True)
testset = datasets.MNIST('./MNIST_data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
images, labels = next(iter(testloader))
images = images.view(images.shape[0], -1)
image_count = images.size()[0]
end = time.time()
print("Loading time: {0:f} secs)".format(end - start))


# API Handler for MNIST test images
class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path_components = self.path.split("/")
        if path_components[1] == 'mnist':
            try:
                index = int(path_components[2])
                self.send_header('Content-type', 'application/json')
                if index < image_count:
                    payload = {}
                    payload['label'] = int(labels[index])
                    img = images[index].view(1, 784)
                    outputs = rep.run(img.numpy())
                    predicted = int(np.argmax(outputs))
                    payload['predicted'] = predicted
                    body = json.dumps(payload)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(bytes(body, "utf8"))
            except (IndexError, ValueError):
                self.send_response(404)
                self.end_headers()
                body = "Not found. Invoke using the form /mnist/<index of test image>. For example, /mnist/24"
                self.wfile.write(bytes(body, "utf8"))

        else:
            self.send_response(400)
            self.end_headers()
            body = "This service verifies a model using the MNIST Test data set. Invoke using the form /mnist/<index of test image>. For example, /mnist/24"
            self.wfile.write(bytes(body, "utf8"))


try:
    server = HTTPServer(('', PORT_NUMBER), MyHandler)
    print('Started httpserver on port', PORT_NUMBER)
    server.serve_forever()

except KeyboardInterrupt:
    server.server_close()
    print('Stopping server')
