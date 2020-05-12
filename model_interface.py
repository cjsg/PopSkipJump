import numpy as np
import random
from conf import SAMPLING_CONF, SAMPLING_FREQ, SLACK


class ModelInterface:
    def __init__(self, models, bounds=(0, 1), n_classes=None, slack=0.10, sampling_freq=10):
        self.models = models
        self.bounds = bounds
        self.n_classes = n_classes
        self.model_calls = 0
        self.sampling_freq = sampling_freq
        if self.sampling_freq is not None:
            # self.threshold = SAMPLING_FREQ * SAMPLING_CONF
            self.slack = slack * sampling_freq

    def forward_one(self, image, a):
        m_id = random.choice(list(range(len(self.models))))
        if self.sampling_freq is not None:
            batch = np.stack([image] * self.sampling_freq)
            outs = self.models[m_id].ask_model(batch)
            label_freqs = np.bincount(outs, minlength=self.n_classes)
            true_freq = label_freqs[a.true_label]
            adv_freq = np.max(label_freqs[np.arange(self.n_classes) != a.true_label])
            if true_freq + self.slack >= adv_freq:
                label = a.true_label
            else:
                label = -2
        else:
            batch = np.stack([image])
            self.model_calls += 1
            label = self.models[m_id].ask_model(batch)[0]
        if label != a.true_label:
            distance = a.calculate_distance(image, self.bounds)
            if a.distance > distance:
                a.distance = distance
                a.perturbed = image
            return 1
        else:
            return 0

    def forward(self, images, a):
        batch = np.stack(images)
        m_id = random.choice(list(range(len(self.models))))
        self.model_calls += len(images)
        if self.sampling_freq is None:
            labels = self.models[m_id].ask_model(batch)
            ans = (labels != a.true_label) * 1
        else:
            inp_batch = np.tile(batch, (self.sampling_freq, 1, 1))
            outs = self.models[m_id].ask_model(inp_batch).reshape(self.sampling_freq, len(images)).T
            N = self.n_classes
            id = outs + (N * np.arange(outs.shape[0]))[:, None]
            freqs = np.bincount(id.ravel(), minlength=N * outs.shape[0]).reshape(-1, N)
            true_freqs = freqs[:, a.true_label]
            r = list(range(self.n_classes))
            false_freqs = np.max(freqs[:, r[:a.true_label] + r[a.true_label+1:]], axis=1)
            ans = (false_freqs > true_freqs + self.slack) * 1

        for i in range(len(images)):
            if ans[i] == 1:
                distance = a.calculate_distance(images[i], self.bounds)
                if a.distance > distance:
                    a.distance = distance
                    a.perturbed = images[i]
        return ans
