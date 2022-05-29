import hydra
from evaluate_model import evaluate
from process import process_data
from train_model import train


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(config):
    process_data(config)
    train(config)
    evaluate(config)


if __name__ == "__main__":
    main()
