from src import (
    train,
    get_config
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for next thoughy generation training")
    parser.add_argument("--config_path", default="src/config/train.yaml", type=str, help="config path to training argument")
    args = parser.parse_args()
    config = get_config(args.config_path)
    train(config)