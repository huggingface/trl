from trl.trainer.grpo_trainer import GRPOTrainer


class GRPOTrainerWithReplayBuffer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = kwargs.get("replay_buffer", None)