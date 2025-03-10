from rosellm.models import CausalModel


class GRPOTrainer:
    def __init__(
        self,
        model_path,
        reward_funcs,
        args,
        train_dataset,
        eval_dataset,
    ):
        self.model_path = model_path
        self.reward_funcs = reward_funcs
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = self._init_model()

    def _init_model(self):
        return CausalModel.from_pretrained(self.model_path, self.args.model)
