import click

from engine.trainer import Trainer
from utils.config_utils import load_config


@click.command()
@click.argument("config-path", type=click.Path(exists=True, dir_okay=False))
def main(config_path):
    config = load_config(config_path)
    trainer = Trainer(config)

    trainer.train()


if __name__ == "__main__":
    main()
