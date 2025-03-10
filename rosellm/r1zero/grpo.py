from datasets import load_dataset

from rosellm.config import Args, Parser
from rosellm.logging.logger import logger
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

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ]
        }

    dataset = dataset.map(make_conversation)


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args_and_config()
    main(args)
