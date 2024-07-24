import os
import logging
import datetime

from config.configurator import configs

def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

class Logger(object):
    def __init__(self, log_configs=True):
        """ logger 配置
        ./log/model_name/dataset_name.log
        """
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        
        # 设置日志文件
        # model_name = configs['model']['name']
        logname = configs['model']['logname']
        log_dir_path = './log/{}'.format(logname)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        dataset_name = configs['data']['name']
        log_file = logging.FileHandler('{}/{}_{}.log'.format(log_dir_path, dataset_name, get_local_time()), 'a', encoding='utf-8')
        # formatter = logging.Formatter('%(asctime)s - %(message)s')
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)
        if log_configs:
            self.log(configs)

    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)
    info = log      # alias!

    def log_loss(self, epoch_idx, loss_log_dict, mode='train', save_to_log=True, print_to_console=True):
        if type(mode) == int:
            epoch = mode
        elif mode == 'train':
            epoch = configs['train']['epoch']
        elif mode == 'kg':
            epoch = configs['train']['epoch_trans']
        message = '[Epoch {:3d} / {:3d}] '.format(epoch_idx, epoch)
        for loss_name in loss_log_dict:
            message += '{}: {:.4f} '.format(loss_name, loss_log_dict[loss_name])
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_eval(self, eval_result, k, data_type, save_to_log=True, print_to_console=True, epoch_idx=None):
        if epoch_idx is not None:
            message = 'Epoch {:3d} '.format(epoch_idx)
        else:
            message = ''

        for metric in eval_result:
            message += '{} ['.format(data_type)
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)