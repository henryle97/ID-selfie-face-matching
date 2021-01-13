from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.config import Cfg
from models.trainer import Trainer
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/base.yml")
    args = parser.parse_args()

    print("="*30 + "CONFIG INFOR" + "="*30)
    config = Cfg.load_config_from_file(args.config)
    print(config)

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
