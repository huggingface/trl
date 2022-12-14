

class BaseTrainer(object):
    def __init__(self, config):
        self.config = config
    
    def step(self, *args):
        raise NotImplementedError("Not implemented")
    
    def loss(self, *args):
        raise NotImplementedError("Not implemented")
    
    def compute_rewards(self, *args):
        raise NotImplementedError("Not implemented")