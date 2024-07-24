""" 
modified from https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/DuIE
"""
import re, json
# from logging import getLogger
# logger = getLogger('train_logger')

from .eval_audit import FormatorUtils
from .eval_audit import AuditVoid, AuditBothEmpty, AuditLabelEmptyOnly, AuditPredEmptyOnly, AuditLong, AuditInsane, AuditRepeat, AuditRetard, AuditWhatever
from .eval_metric import MetricBase, MetricF1


class EvaluatorBase(FormatorUtils):
    """ 高层的评估器基类, 核心包括了审计和指标
    add: 添加新的数据, 输入的两个参数为 list/str, 最终的字符串都会通过 _extract 函数构建为标准化的 string, 用于 metric/audit
        _extract: 需要定义实现的抽取函数. 从输入的数据中提取出 y_truth, y_pred, 返回的两个参数为 set
        _format: 格式规范化
    dump_audit_report: 导出审计报告, json 字符串
        _init_audit: 初始化审计项目
    get_metric: 获取指标
    """
    def __init__(self):
        self.last = dict()
        self._init_audit()
        self._init_metric()
    
    def _init_metric(self):
        # must be overrided to init self.metric
        self.metric = MetricBase()

    def _extract(self, golden_list, predict_str: str):
        # must be overrided
        # return: y_truth, y_pred -> set
        raise NotImplementedError()

    def _init_audit(self):
        # override if necessary
        # 如果需要添加其他审计项目或者自定义实例化的话请考虑重载此方法
        self.audit = [
            AuditVoid(),
            AuditBothEmpty(),
            AuditLabelEmptyOnly(),
            AuditPredEmptyOnly(),
            AuditLong(),
            AuditInsane(),
            AuditRepeat(),
            AuditRetard(),
            AuditWhatever()
        ]
    
    def _update_audit(self):
        # override if necessary
        for audit in self.audit:
            audit.update(self.last)

    def add(self, golden_list, predict_str):
        """
        json_data应该为json-like object(即json.load解析出来的)，predict应该为单个字符串。
        """
        if isinstance(golden_list, str):
            golden_list = json.loads(golden_list)
        if not isinstance(predict_str, str):
            predict_str = json.dumps(predict_str)
        # add single case
        y_truth, y_pred = self._extract(golden_list, predict_str)
        self.metric.update(y_truth, y_pred)

        # audit
        # last中存储需要提交审计的所有可能会用到的信息
        self.last['json_data'] = golden_list
        self.last['predict'] = predict_str
        self.last['y_truth'] = y_truth
        self.last['y_pred'] = y_pred
        self.last['metric'] = self.metric

        self._update_audit()
    
    def add_batch(self, json_data, predict):
        for i, j in zip(json_data, predict):
            self.add(i, j)

    def get_metric(self) -> float:
        return self.metric.get_metric()

    def get_last_metric(self):
        return self.metric.get_last()

    def get_audit_report(self):
        '获取所有审计项结果报告，返回一个json-like object'
        return {
            a.get_name() : a.get_report()
            for a in self.audit
        }
    def dump_audit_report(self, fpath):
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(self.get_audit_report(), f, indent=4, ensure_ascii=False)


class EvaluatorRE(EvaluatorBase):
    keys = ['subject', 'predicate', 'object']
    def _init_metric(self):
        self.metric = MetricF1()

    # def _extract(self, json_data, predict):
    #     """ 两者都是字符串形式的json dict list, 提取出其中的json dict, 并转化为set形式的三元组 """
    #     y_truth = set()
    #     for triplet in self._format_json_dict(json_data):
    #         y_truth.add(self._format(triplet))
    #     y_pred = set()
    #     for triplet in self._format_json_dict(predict):
    #         y_pred.add(self._format(triplet))
    #     return y_truth, y_pred

    def _format_triplet(self, triplet):
        triplet_ = [triplet[k] for k in self.keys]
        triplet_str = "|".join(triplet_)
        return triplet_str

    def _extract(self, golden_list, predict_str):
        y_truth = set()
        for triplet in golden_list:
            triplet_str = self._format_triplet(triplet)
            y_truth.add(self._format(triplet_str))
        y_pred = set()
        predict_str = json.loads(predict_str)
        for triplet in predict_str:
            triplet_str = self._format_triplet(triplet)
            y_pred.add(self._format(triplet_str))
        return y_truth, y_pred

    def get_metric_dict(self):
        f1, recall, precision = self.metric.get_detail()
        f1 = round(f1, 4)
        recall = round(recall, 4)
        precision = round(precision, 4)
        return {
            "f1": f1,
            "recall": recall,
            "precision": precision
        }