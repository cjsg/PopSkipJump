import numpy as np
import random


class ModelInterface:
    def __init__(self, models, bounds=(0, 1), sampling_freq=None):
        self.models = models
        self.bounds = bounds
        self.model_calls = 0
        self.sampling_freq = sampling_freq
        self.sampling_conf = 0.40

    def forward_one(self, image, a):
        m_id = random.choice(list(range(len(self.models))))
        if self.sampling_freq is not None:
            batch = np.stack([image] * self.sampling_freq)
            outs = self.models[m_id].ask_model(batch)
            label_freqs = np.bincount(outs)
            if np.max(label_freqs) >= self.sampling_conf * self.sampling_freq:
                label = np.argmax(label_freqs)
            else:
                label = -2
        else:
            batch = np.stack([image])
            self.model_calls += 1
            label = self.models[m_id].ask_model(batch)[0]
        if label != -2 and label != a.true_label:
            distance = a.calculate_distance(image, self.bounds)
            if a.distance.value > distance.value:
                a.distance = distance
                a.perturbed = image
            return 1
        elif label == -2:
            return -1
        else:
            return 0

    def forward(self, images, a):
        batch = np.stack(images)
        m_id = random.choice(list(range(len(self.models))))
        self.model_calls += len(images)
        if self.sampling_freq is None:
            labels = self.models[m_id].ask_model(batch)
        else:
            inp_batch = np.tile(batch, (self.sampling_freq, 1, 1))
            outs = self.models[m_id].ask_model(inp_batch).reshape(self.sampling_freq, len(images)).T
            N = outs.max() + 1
            id = outs + (N * np.arange(outs.shape[0]))[:, None]
            freqs = np.bincount(id.ravel(), minlength=N * outs.shape[0]).reshape(-1, N)
            labels = np.argmax(freqs, axis=1)
            label_freq = np.max(freqs, axis=1)
            labels[label_freq < self.sampling_conf * self.sampling_freq] = -2

        for i in range(len(images)):
            if labels[i] != -2 and labels[i] != a.true_label:
                distance = a.calculate_distance(images[i], self.bounds)
                if a.distance.value > distance.value:
                    a.distance = distance
                    a.perturbed = images[i]
        ans = (labels != a.true_label) * 1
        ans[labels == -2] = -1
        return ans
