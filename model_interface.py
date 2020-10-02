import numpy as np
import random
import torch


class ModelInterface:
    def __init__(self, models, bounds=(0, 1), n_classes=None, slack=0.10, noise='deterministic', new_adv_def=False,
                 device=None):
        self.models = models
        self.bounds = bounds
        self.n_classes = n_classes
        self.model_calls = 0
        self.slack_prop = slack
        self.noise = noise
        self.new_adversarial_def = new_adv_def
        self.device = device
        self.send_models_to_device()

    def send_models_to_device(self):
        for model in self.models:
            model.model = model.model.to(self.device)

    def forward_one(self, image, a, freq, is_original=False):
        slack = self.slack_prop * freq
        m_id = random.choice(list(range(len(self.models))))
        if self.noise != 'deterministic':
            new_def_threshold = 0.6 if is_original else 0.5
            batch = torch.stack([image] * freq)
            outs = self.models[m_id].ask_model(batch)
            self.model_calls += freq
            label_freqs = torch.bincount(outs, minlength=self.n_classes)
            true_freq = label_freqs[a.true_label]
            adv_freq = torch.max(label_freqs[np.arange(self.n_classes) != a.true_label])
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

    def sample_bernoulli(self, probs):
        self.model_calls += len(probs)
        return torch.bernoulli(probs)


    def get_probs_(self, images):
        """
            WARNING
            This function should only be used for capturing statistics.
            It should not be a part of a decision based attack.
        """
        m_id = random.choice(list(range(len(self.models))))
        outs = self.models[m_id].get_probs(images)
        return outs

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
        if type(images) != torch.Tensor:
            images = torch.tensor(images).to(self.device)
        slack = self.slack_prop * freq
        batch = torch.stack(tuple(images))
        m_id = random.choice(list(range(len(self.models))))
        if self.noise == 'deterministic':
            labels = self.models[m_id].ask_model(batch)
            ans = (labels != a.true_label) * 1
            self.model_calls += len(images)
        else:
            inp_batch = batch.repeat(freq, 1, 1)
            outs = self.models[m_id].ask_model(inp_batch).reshape(freq, len(images)).T
            self.model_calls += (len(images) * freq)
            N = self.n_classes
            id = outs + (N * torch.arange(outs.shape[0]).to(self.device))[:, None]
            freqs = torch.bincount(id.flatten(), minlength=N * outs.shape[0]).view(-1, N)
            true_freqs = freqs[:, a.true_label]
            r = list(range(self.n_classes))
            false_freqs = torch.max(freqs[:, r[:a.true_label] + r[a.true_label + 1:]], dim=1)[0]

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
