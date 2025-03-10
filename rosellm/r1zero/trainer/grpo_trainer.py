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
