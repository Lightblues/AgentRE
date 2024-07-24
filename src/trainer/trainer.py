
import json
import random
import numpy as np
import pandas as pd
import torch
from config.configurator import configs
from trainer.logger import Logger
from trainer.metrics_v2 import EvaluatorRE
from datasets import Dataset
from .utils_trainer import DisabledSummaryWriter, log_exceptions
from data_utils.data_handler_re import DataHandlerRE
from models.base_model import BaseModel as Model            # just for lint
# from models.react_memory import ReAct_Memory as Model     # just for lint

def init_seed(seed=0):
    if 'seed' in configs['model']:
        seed = configs['model']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed

class Trainer(object):
    """ 
    属性: 
        logger: Logger, 同时输出到文件和控制台
    """
    data_handler:DataHandlerRE
    logger:Logger
    evaluator:EvaluatorRE = EvaluatorRE()
    def __init__(self, data_handler:DataHandlerRE, logger:Logger):
        self.data_handler = data_handler
        self.logger = logger

    @log_exceptions
    def predict(self, model:Model):
        ds_test:Dataset = self.data_handler.ds_test

        self.logger.info(f"Start predicting with {len(ds_test)} samples.")
        # ds_pred = ds.map(model.process_sample, with_indices=True, load_from_cache_file=False)     # 禁止使用 datasets 的缓存
        proces_res = []
        for idx, sample in enumerate(ds_test):
            res = model.process_sample(sample, idx)
            proces_res.append(res)
            golden = sample['spo_list']
            pred = json.dumps(res['spo_list_pred'], ensure_ascii=False)
            self.evaluator.add(golden, pred)
            metric_dict = self.evaluator.get_metric_dict()
            self.logger.info(f"idx={idx}, metric_dict={metric_dict}")
        df_pred = pd.concat([pd.DataFrame(proces_res), ds_test.to_pandas()], axis=1)
        ds_pred = Dataset.from_pandas(df_pred)
        self.logger.info(f"Finish predicting with {len(ds_pred)} samples.")
        self.evaluator.dump_audit_report(self.data_handler.data_meta.ofn_report)

        self.data_handler.ds_pred = ds_pred
        self.data_handler.save_results()
        return ds_pred

    @log_exceptions
    def train(self, model:Model):
        """ 对于带有 memory 的模型, 用这个接口 """
        ds_test:Dataset = self.data_handler.ds_test

        self.logger.info(f"Start predicting with {len(ds_test)} samples.")
        proces_res = []
        for idx, sample in enumerate(ds_test):
            res = model.train_sample(sample, idx)
            proces_res.append(res)
            # NEW: 增加评估器的评估
            golden = sample['spo_list']
            pred = json.dumps(res['spo_list_pred'], ensure_ascii=False)
            self.evaluator.add(golden, pred)
            metric_dict = self.evaluator.get_metric_dict()
            self.logger.info(f"idx={idx}, metric_dict={metric_dict}")
        df_pred = pd.concat([pd.DataFrame(proces_res), ds_test.to_pandas()], axis=1)
        ds_pred = Dataset.from_pandas(df_pred)
        self.logger.info(f"Finish predicting with {len(ds_pred)} samples.")
        self.evaluator.dump_audit_report(self.data_handler.data_meta.ofn_report)

        self.data_handler.ds_pred = ds_pred
        self.data_handler.save_results()    # TODO: 和 predict 保存到不同路径
        return ds_pred

    @log_exceptions
    def evaluate_v0(self, model):
        """ [v1] 参见 lang/DuIE/process_generated.py """
        from trainer.metrics import Metric
        self.data_handler.load_results()

        # 之前用于处理预测结果, 现在转移到生成逻辑中
        # ds:Dataset = self.data_handler.ds_pred
        # process_f = model.process_parse_generated
        # ds_pred_parsed = ds.map(process_f, with_indices=True, load_from_cache_file=False)     # 禁止使用 datasets 的缓存
        # self.data_handler.ds_pred_parsed = ds_pred_parsed
        # self.data_handler.save_results_parsed()

        ret_info = Metric(self.data_handler).evaluate()
        self.logger.info(f"Evaluation result: {ret_info}")
        return ret_info

    @log_exceptions
    def evaluate(self, model):
        """ [v2] 新版本的评估函数 
        读取 predict 的结果, 然后进行评估 (需要先运行 predict)
        """
        evaluator = EvaluatorRE()       # 新建一个评估器

        self.logger.info("Loading results...")
        self.data_handler.load_results()
        
        self.logger.info("Evaluating...")
        spo_list_pred = self.data_handler.ds_pred['spo_list_pred']
        spo_list_golden = self.data_handler.ds_test['spo_list']
        # spo_list_golden = [eval(x) for x in spo_list_golden]
        assert len(spo_list_pred) == len(spo_list_golden)
        for pred, golden in zip(spo_list_pred, spo_list_golden):
            evaluator.add(golden, json.dumps(pred, ensure_ascii=False))

        ret_info = evaluator.get_metric_dict()
        self.logger.info(f"Evaluation result: {ret_info}")

        evaluator.dump_audit_report(self.data_handler.data_meta.ofn_report)
        return ret_info

