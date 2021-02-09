import random
import torch
import torch.nn.functional as F


class ModelInterface:
    """
        All queries to classifiers/models to should happend via this class.
        It is a wrapper over a set of models that:
            - tracks model calls
            - implements the logic to pick a model
            - implements the definition of an adversarial example
    """
    def __init__(self, models, bounds=(0, 1), n_classes=None, slack=0.10, noise='deterministic',
                 new_adv_def=False, device=None, flip_prob=0.0, smoothing_noise=0., crop_size=None):
        self.models = models
        self.bounds = bounds
        self.n_classes = n_classes
        self.model_calls = 0
        self.slack_prop = slack
        self.noise = noise
        self.new_adversarial_def = new_adv_def
        self.device = device
        self.send_models_to_device()
        self.flip_prob = flip_prob
        self.smoothing_noise = smoothing_noise
        self.crop_size = crop_size

    def send_models_to_device(self):
        for model in self.models:
            model.model = model.model.to(self.device)

    def sample_bernoulli(self, probs):
        self.model_calls += probs.numel()
        return torch.bernoulli(probs)

    def decision(self, batch, label, num_queries=1, targeted=False):
        N = batch.shape[0] * num_queries
        self.model_calls += batch.shape[0] * num_queries
        # if N <= 100*1000:
        if batch.ndim == 3:
            new_batch = batch.repeat(num_queries, 1, 1)
        else:
            new_batch = batch.repeat(num_queries, 1, 1, 1)
        decisions = self._decision(new_batch, label, targeted)
        decisions = decisions.view(-1, len(batch)).transpose(0, 1)
        # elif num_queries <= 100*1000:
        #     decisions = torch.zeros(len(batch), num_queries, device=batch.device)
        #     for b in range(len(batch)):
        #         if batch.ndim == 3:
        #             new_batch = batch[b].view(-1, 1, 1).repeat(num_queries, 1, 1)
        #         else:
        #             new_batch = batch[b].view(-1, 1, 1, 1).repeat(num_queries, 1, 1, 1)
        #         decisions[b] = self._decision(new_batch, label, targeted)
        # else:
        #     decisions = torch.zeros(len(batch), num_queries, device=batch.device)
        #     for q in range(num_queries):
        #         decisions[:, q] = self._decision(batch, label, targeted)
        return decisions

    def _decision(self, batch, label, targeted=False):
        """
        :param label: True/Targeted labels of the original image being attacked
        :param num_queries: Number of times to query each image
        :param batch: A batch of images
        :param targeted: if targeted is true, label=targeted_label else label=true_label
        :return: decisions of shape = (len(batch), num_queries)
        """
        if self.noise == 'deterministic':
            probs = self.get_probs_(images=batch)
            prediction = probs.argmax(dim=1)
            if targeted:
                return (prediction == label) * 1.0
            else:
                return (prediction != label) * 1.0
        elif self.noise == 'dropout':
            probs = self.get_probs_(images=batch)
            prediction = probs.argmax(dim=1)
            if targeted:
                return (prediction == label) * 1.0
            else:
                return (prediction != label) * 1.0
        elif self.noise == 'smoothing':
            rv = torch.randn(size=batch.shape, device=self.device)
            batch_ = batch + self.smoothing_noise * rv
            batch_ = torch.clamp(batch_, self.bounds[0], self.bounds[1])
            probs = self.get_probs_(images=batch_)
            prediction = probs.argmax(dim=1)
            if targeted:
                return (prediction == label) * 1.0
            else:
                return (prediction != label) * 1.0
        elif self.noise == 'cropping':
            size = batch.shape[1]
            x_start = torch.randint(low=0, high=size+1-self.crop_size, size=(1, len(batch)))[0]
            x_end = x_start + self.crop_size
            y_start = torch.randint(low=0, high=size+1-self.crop_size, size=(1, len(batch)))[0]
            y_end = y_start + self.crop_size
            cropped = [b[x_start[i]:x_end[i], y_start[i]:y_end[i]] for i, b in enumerate(batch)]
            cropped_batch = torch.stack(cropped)
            if cropped_batch.ndim == 4:
                resized = F.interpolate(cropped_batch.permute(0, 3, 1, 2), size, mode='bilinear')
                resized = resized.permute(0, 2, 3, 1)
            else:
                resized = F.interpolate(cropped_batch.unsqueeze(dim=1), size, mode='bilinear')
                resized = resized.squeeze(dim=1)
            probs = self.get_probs_(images=resized)
            prediction = probs.argmax(dim=1)
            if targeted:
                return (prediction == label) * 1.0
            else:
                return (prediction != label) * 1.0
        elif self.noise == 'stochastic':
            num_queries = 1  # TODO: this should be removed. num_queries is not supported by this function now
            probs = self.get_probs_(images=batch)
            rand_pred = torch.randint(self.n_classes-1, size=(len(batch), num_queries), device=self.device)
            # TODO: Review this step carefully. I think it is assumed that prediction = label
            rand_pred[rand_pred == label] = self.n_classes - 1
            prediction = probs.argmax(dim=1).view(-1, 1).repeat(1, num_queries)
            indices_to_flip = torch.rand(size=(len(batch), num_queries), device=self.device) < self.flip_prob
            prediction[indices_to_flip] = rand_pred[indices_to_flip]
            if targeted:
                return (prediction == label) * 1.0
            else:
                return (prediction != label) * 1.0

        elif self.noise == 'bayesian':
            probs = self.get_probs_(images=batch)
            probs = probs[:, label]
            # probs = probs.view(-1, 1).repeat(1, num_queries)
            if targeted:
                decisions = torch.bernoulli(probs)
            else:
                decisions = torch.bernoulli(1 - probs)
            return decisions
        else:
            raise RuntimeError(f'Unknown Noise type: {self.noise}')

    def decision_with_logits(self, batch, true_label):
        """
        Same as decision() but insteas of decision it returns logit vectors. Used for white-box attacks
        :return: decisions of shape = (len(batch), num_classes)
        """
        probs = self.get_probs_(images=batch)
        self.model_calls += batch.shape[0]
        if self.noise == 'deterministic':
            ans = torch.zeros_like(probs)
            ans[torch.arange(len(probs)), probs.argmax(axis=1)] = 1
            return ans
        elif self.noise == 'stochastic':
            ans = torch.ones_like(probs) * self.flip_prob / (self.n_classes - 1)
            ans[torch.arange(len(probs)), probs.argmax(axis=1)] = 1 - self.flip_prob
            return ans
        elif self.noise == 'bayesian':
            return probs
        else:
            raise RuntimeError(f'Unknown Noise type: {self.noise}')

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

    @DeprecationWarning
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
