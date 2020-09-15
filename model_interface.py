import numpy as np
import random


class ModelInterface:
    def __init__(self, models, bounds=(0, 1), n_classes=None, slack=0.10, noise='deterministic', new_adv_def=False):
        self.models = models
        self.bounds = bounds
        self.n_classes = n_classes
        self.model_calls = 0
        self.slack_prop = slack
        self.noise = noise
        self.new_adversarial_def = new_adv_def

    def forward_one(self, image, a, freq, is_original=False):
        slack = self.slack_prop * freq
        m_id = random.choice(list(range(len(self.models))))
        if self.noise != 'deterministic':
            new_def_threshold = 0.6 if is_original else 0.5
            batch = np.stack([image] * freq)
            outs = self.models[m_id].ask_model(batch)
            self.model_calls += freq
            label_freqs = np.bincount(outs, minlength=self.n_classes)
            true_freq = label_freqs[a.true_label]
            adv_freq = np.max(label_freqs[np.arange(self.n_classes) != a.true_label])
            if self.new_adversarial_def and true_freq >= new_def_threshold * freq:
                label = a.true_label
            elif not self.new_adversarial_def and true_freq + slack >= adv_freq:
                label = a.true_label
            else:
                label = -2
        else:
            batch = np.stack([image])
            label = self.models[m_id].ask_model(batch)[0]
            self.model_calls += 1
        if label != a.true_label:
            distance = a.calculate_distance(image, self.bounds)
            if a.distance > distance:
                a.distance = distance
                a.perturbed = image
            return 1
        else:
            return 0

    def get_probs(self, image):
        """
            WARNING
            This function should only be used for capturing statistics.
            It should not be a part of a decision based attack.
        """
        m_id = random.choice(list(range(len(self.models))))
        outs = self.models[m_id].get_probs(image[None])
        return outs

    def get_grads(self, images, true_label):
        """
            WARNING
            This function should only be used for capturing statistics.
            It should not be a part of a decision based attack.
        """
        m_id = random.choice(list(range(len(self.models))))
        outs = self.models[m_id].get_grads(images, true_label)
        return outs

    def forward(self, images, a, freq, average=False, remember=True):
        slack = self.slack_prop * freq
        batch = np.stack(images)
        m_id = random.choice(list(range(len(self.models))))
        if self.noise == 'deterministic':
            labels = self.models[m_id].ask_model(batch)
            ans = (labels != a.true_label) * 1
            self.model_calls += len(images)
        else:
            inp_batch = np.tile(batch, (freq, 1, 1))
            outs = self.models[m_id].ask_model(inp_batch).reshape(freq, len(images)).T
            self.model_calls += (len(images) * freq)
            N = self.n_classes
            id = outs + (N * np.arange(outs.shape[0]))[:, None]
            freqs = np.bincount(id.ravel(), minlength=N * outs.shape[0]).reshape(-1, N)
            true_freqs = freqs[:, a.true_label]
            r = list(range(self.n_classes))
            false_freqs = np.max(freqs[:, r[:a.true_label] + r[a.true_label + 1:]], axis=1)

            if self.new_adversarial_def:
                ans = (true_freqs < 0.5 * freq) * 1
            else:
                ans = (false_freqs > true_freqs + slack) * 1

        if remember:
            for i in range(len(images)):
                if ans[i] == 1:
                    distance = a.calculate_distance(images[i], self.bounds)
                    if a.distance > distance:
                        a.distance = distance
                        a.perturbed = images[i]
        if average and self.noise != 'deterministic':
            adv_prob = 1 - (true_freqs / freq)
            return adv_prob
        else:
            return ans
