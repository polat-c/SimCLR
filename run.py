from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper

## TESTING COMMENT


def main():
    #config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    config = yaml.load(open("config_cifar.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
