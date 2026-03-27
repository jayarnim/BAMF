class LinearAnnealer(object):
    def __init__(
        self,
        min,
        max,
        warmup,
    ):
        super().__init__()
        self.min = min
        self.max = max
        self.warmup = warmup

    def __call__(self, epoch):
        CURRENT = epoch / self.warmup
        MAX = self.max
        progress = min(CURRENT, MAX)
        beta = self.min + (self.max - self.min) * progress
        return beta