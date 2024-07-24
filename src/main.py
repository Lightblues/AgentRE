
from config.configurator import configs

from trainer.logger import Logger
from trainer.trainer import init_seed, Trainer
from data_utils.build_data_handler import build_data_handler
from data_utils.data_handler_re import DataHandlerRE
from models.build_model import build_model
from trainer.build_trainer import build_trainer

def main():
    # First Step: Create data_handler
    init_seed()
    data_handler:DataHandlerRE = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler)

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer:Trainer = build_trainer(data_handler, logger)

    # train
    if configs['train']['if_train']:
        # model.set_is_training(True)
        trainer.train(model)

    # predict
    if configs['train']['if_predict']:
        trainer.predict(model)

    # evaluate
    if configs['train']['if_evaluate']:
        trainer.evaluate(model)

main()


