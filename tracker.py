class Diary:
    def __init__(self, image, label):
        self.true_label = label
        self.original = image

        self.initial_image = None
        self.initial_projection = None

        self.calls_initialization = None
        self.calls_initial_bin_search = None

        self.epoch_start = None
        self.epoch_initialization = None
        self.epoch_initial_bin_search = None

        self.iterations = list()


class DiaryPage:
    def __init__(self):
        self.distance = None
        self.num_eval_det = None
        self.num_eval_prob = None
        self.approx_grad = None
        self.opposite = None
        self.bin_search = None
        self.info_max_stats = None
        self.calls: Calls = Calls()
        self.time: Time = Time()


class InfoMaxStats:
    def __init__(self, s, tmap, samples):
        self.s = s
        self.tmap = tmap
        self.samples = samples


class Time:
    def __init__(self):
        self.start = None
        self.num_evals = None
        self.approx_grad = None
        self.step_search = None
        self.bin_search = None
        self.end = None


class Calls:
    def __init__(self):
        self.start = None
        self.initial_projection = None
        self.approx_grad = None
        self.step_search = None
        self.bin_search = None