import json
import argparse
from trainer import train
import wandb

mimg_sweep_config = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "last_top_curve"},
    "parameters": {
        "lr_rate": {"values": [0.00005, 0.0001, 0.0005, 0.001, 0.005]},
    },
}

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    # wandb.init(project="PILOT-IMNR")
    # args["lr_rate"] = wandb.config.lr_rate

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/coso_cifar.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, default='0')
    
    return parser

if __name__ == '__main__':
    # normal train
    main()

    # use wandb sweep
    # sweep_id = wandb.sweep(sweep=mimg_sweep_config, project="PILOT-IMNR")
    # wandb.agent(sweep_id, function=main)
