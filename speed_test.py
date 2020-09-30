import time
import torch
from tqdm import tqdm
import sys
from model_factory import get_model
from img_utils import get_samples, get_device

device = torch.device(sys.argv[1])
N = 10000
B = 100
model = get_model(key='mnist_noman', dataset='mnist', noise='determinisitc', device=device)
model.model = model.model.to(device)
images = get_samples(N)[0]
start = time.time()
for i in tqdm(range(N)):
    model.ask_model(images[i:i+1])
print(time.time() - start)

start = time.time()
for i in tqdm(range(0, N, B)):
    model.ask_model(images[i:i+B])
print(time.time() - start)