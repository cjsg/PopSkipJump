class Diary(object):
    def __init__(self, image, label):
        self.true_label = label
        self.original = image

        self.initial_image = None
        self.initial_projection = None
        self.init_infomax = None

        self.calls_initialization = None
        self.calls_initial_bin_search = None

        self.epoch_start = None
        self.epoch_initialization = None
        self.epoch_initial_bin_search = None

        self.iterations = list()


class DiaryPage(object):
    def __init__(self):
        self.initial = None
        self.distance = None
        self.num_eval_det = None
        self.num_eval_prob = None
        self.delta = None
        self.approx_grad = None
        self.opposite = None
        self.bin_search = None
        self.info_max_stats = None
        self.calls: Calls = Calls()
        self.time: Time = Time()
        self.grad_estimate = None
        self.grad_true = None


class InfoMaxStats(object):
    def __init__(self, s, tmap, samples, e, n):
        self.s = s
        self.e = e
        self.n = n
        self.tmap = tmap
        self.samples = samples


class Time(object):
    def __init__(self):
        self.start = None
        self.num_evals = None
        self.approx_grad = None
        self.step_search = None
        self.bin_search = None
        self.end = None


class Calls(object):
    def __init__(self):
        self.start = None
        self.initial_projection = None
        self.approx_grad = None
        self.step_search = None
        self.bin_search = None
