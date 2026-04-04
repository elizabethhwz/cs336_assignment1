from experiments.train_llm_utils import parse_args, run_training


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
