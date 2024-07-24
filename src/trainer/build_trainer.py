from config.configurator import configs
import importlib

def build_trainer(data_handler, logger):
    trainer_name = configs['train']['trainer'] if ('trainer' in configs['train'] and configs['train']['trainer']) else 'Trainer'
    # delete '_' in trainer name
    trainer_name = trainer_name.replace('_', '')

    trainers = importlib.import_module('trainer.trainer')
    for attr in dir(trainers):
        if attr.lower() == trainer_name.lower():
            return getattr(trainers, attr)(data_handler, logger)
    else:
        raise NotImplementedError('Trainer Class {} is not defined in {}'.format(trainer_name, 'trainer.trainer'))
