import os
import yaml
import argparse
from .config_loader import ConfigLoader

def update_configs(configs, up):
    """ recursively update configs """
    for k, v in up.items():
        if k not in configs:
            configs[k] = v
        elif isinstance(v, dict):
            update_configs(configs[k], v)
        else:
            configs[k] = v

def parse_configure():
    parser = argparse.ArgumentParser(description='AgentRE')
    # parser.add_argument('--model', type=str, default="ZSL", help='Model name') 
    # parser.add_argument('--model', type=str, default="ZSL_COT", help='Model name') 
    # parser.add_argument('--model', type=str, default="FSL", help='Model name') 
    # parser.add_argument('--model', type=str, default="FSL_COT", help='Model name') 
    # parser.add_argument('--model', type=str, default="REACT_ZSL", help='Model name') 
    parser.add_argument('--model', type=str, default="REACT_FSL", help='Model name') 
    # parser.add_argument('--model', type=str, default="REACT", help='Model name') 
    # parser.add_argument('--model', type=str, default="REACT_Memory", help='Model name') 

    parser.add_argument('--logname', type=str, default=None, help='Log name')           # specific the log name
    parser.add_argument('--config_list', type=str, default=None, help='Config list')    # override configs that can be specified 
    
    parser.add_argument('--dataset', type=str, default="SciERC", choices=['DuIE2.0', 'SciERC'], help='Dataset name')

    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    args = parser.parse_args()

    if args.model == None:
        raise Exception("Please provide the model name through --model.")
    model_name = args.model.lower()
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")


    yml_fn = './config/modelconf/{}.yml'.format(model_name)
    configs = ConfigLoader().load_from(yml_fn)

    # model
    configs['model']['name'] = configs['model']['name'].lower()
    configs['model']['logname'] = configs['model']['name'] if (not args.logname) else args.logname
    # dataset
    if args.dataset is not None:
        configs['data']['name'] = args.dataset

    # read config_list and overwrite configs
    if args.config_list:
        config_list = args.config_list.split(',')
        for config_name in config_list:
            fn = f"./config/override/{config_name}.yml"
            if not os.path.exists(fn):
                raise Exception(f"Config file {fn} does not exist.")
            with open(fn, encoding='utf-8') as f:
                config_data = f.read()
                config = yaml.safe_load(config_data)
                update_configs(configs, config)

    return configs

configs = parse_configure()
