import random
import torch


class ModelInterface:
    """
        All queries to classifiers/models to should happend via this class.
        It is a wrapper over a set of models that:
            - tracks model calls
            - implements the logic to pick a model
            - implements the definition of an adversarial example
    """
    def __init__(self, models, bounds=(0, 1), n_classes=None, slack=0.10, noise='deterministic',
                 new_adv_def=False, device=None):
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

    def sample_bernoulli(self, probs):
        self.model_calls += probs.numel()
        return torch.bernoulli(probs)

    def decision(self, batch, true_label, num_queries=1):
        """
        :param true_label: True labels of the original image being attacked
        :param num_queries: Number of times to query each image
        :param batch: A batch of images
        :return: decisions of shape = (len(batch), num_queries)
        """
        probs = self.get_probs_(images=batch)
        self.model_calls += batch.shape[0] * num_queries
        if self.noise == 'deterministic':
            prediction = probs.argmax(dim=1).view(-1, 1).repeat(1, num_queries)
            return (prediction != true_label) * 1.0
        else:
            probs = probs[:, true_label]
            probs = probs.view(-1, 1).repeat(1, num_queries)
            decisions = self.sample_bernoulli(1 - probs)
            return decisions

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

    # TODO: Will be deprecated soon
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
