from rosellm.config import Args, Parser
from rosellm.logging.logger import logger
from rosellm.utils import set_seed


def main(args: Args):
    set_seed(args.training.seed)
    logger.info(f"set seed: {args.training.seed}")


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args_and_config()
    main(args)
