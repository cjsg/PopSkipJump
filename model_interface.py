import numpy as np
import torch
from torchvision import transforms


class ModelInterface:
    def __init__(self, model, bounds=(0, 1)):
        self.model = model
        self.bounds = bounds
        pass

    def ask_model(self, images):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                             [0.2023, 0.1994, 0.2010])])
        img_tr = [transform(i) for i in images]
        out = self.model(torch.stack(img_tr))
        return np.argmax(out.detach().numpy(), axis=1)

    def forward_one(self, image, a):
        batch = np.stack([image])
        labels = self.ask_model(batch)
        if labels[0] != a.true_label:
            distance = a.calculate_distance(image, self.bounds)
            if a.distance.value > distance.value:
                a.distance = distance
                a.perturbed = image
            return 1
        else:
            return 0

    def forward(self, batch, a):
        outs = list()
        for image in batch:
            outs.append(self.forward_one(image, a))
        return np.array(outs)
