import numpy as np


class ModelInterface:
    def __init__(self, model, bounds=(0, 1)):
        self.model = model
        self.bounds = bounds

    def forward_one(self, image, a):
        batch = np.stack([image])
        labels = self.model.ask_model(batch)
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
