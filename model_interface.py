import numpy as np
import random


class ModelInterface:
    def __init__(self, models, bounds=(0, 1)):
        self.models = models
        self.bounds = bounds
        self.model_calls = 0

    def forward_one(self, image, a):
        batch = np.stack([image])
        m_id = random.choice(list(range(len(self.models))))
        self.model_calls += 1
        labels = self.models[m_id].ask_model(batch)
        if labels[0] != a.true_label:
            distance = a.calculate_distance(image, self.bounds)
            if a.distance.value > distance.value:
                a.distance = distance
                a.perturbed = image
            return 1
        else:
            return 0

    def forward(self, images, a):
        batch = np.stack(images)
        m_id = random.choice(list(range(len(self.models))))
        self.model_calls += len(images)
        labels = self.models[m_id].ask_model(batch)
        for i in range(len(images)):
            if labels[i] != a.true_label:
                distance = a.calculate_distance(images[i], self.bounds)
                if a.distance.value > distance.value:
                    a.distance = distance
                    a.perturbed = images[i]
        outs = labels != a.true_label
        return outs * 1
