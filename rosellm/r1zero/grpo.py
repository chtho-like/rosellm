import torch
from datasets import DatasetDict, load_dataset

from rosellm.config import Args, Parser
from rosellm.logging.logger import logger
from rosellm.r1zero.trainer.grpo_trainer import GRPOTrainer
from rosellm.utils import set_seed

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process "
    "in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, "
    "i.e., "
    "<think> reasoning process here </think>"
    "<answer> answer here </answer>"
)


def main(args: Args):
    logger.info(f"logging level: {args.training.logging_level}")
    logger.setLevel(args.training.logging_level)
    set_seed(args.training.seed)
    logger.info(f"set seed: {args.training.seed}")

    # Load dataset.
    dataset = load_dataset(args.dataset.path, args.dataset.name)
    logger.info(f"dataset: {dataset}")
    dataset = dataset.map(
        lambda example: {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ]
        }
    )
    torch_dtype = (
        args.model.torch_dtype
        if args.model.torch_dtype in ["auto", None]
        else getattr(torch, args.model.torch_dtype)
    )

    if isinstance(dataset, DatasetDict):
        # For DatasetDict, access by split name.
        train_dataset = dataset[args.dataset.train_split]
        eval_dataset = dataset[args.dataset.eval_split]
    else:
        # For single dataset, use the same dataset for train and eval.
        logger.warning(
            "Dataset is not a DatasetDict, using same dataset for train and eval."
        )
        train_dataset = dataset
        eval_dataset = dataset

    logger.info(f"initializing GRPOTrainer")
    trainer = GRPOTrainer(
        model_path=args.model.path,
        reward_funcs=["accuracy"],
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args_and_config()
    main(args)
