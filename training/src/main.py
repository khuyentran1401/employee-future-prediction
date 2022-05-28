import hydra
from process import process_data
from train_model import train


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(config):
    process_data(config)
    train(config)


if __name__ == "__main__":
    main()
