

class MetricBase:
    """ 指标基类
    update: 加入新的数据, 更新指标, 输入参数为set
    get_metric: 获取指标
    """
    def __init__(self):
        raise NotImplementedError()
    def update(self, y_truth, y_pred):
        raise NotImplementedError()
    def get_metric(self):
        raise NotImplementedError()
    def get_last(self):
        raise NotImplementedError()

class MetricF1(MetricBase):
    def __init__(self):
        self.sum_TP = 0
        self.sum_FN = 0
        self.sum_FP = 0
        self.last_TP = None
        self.last_FN = None
        self.last_FP = None
    def update(self, y_truth: set, y_pred: set):
        # TP: 在truth中存在，且在pred中存在
        # FN: 在truth中存在，但在pred中不存在
        # FP: 在truth中不存在，但在pred中存在
        self.last_TP = len(y_truth & y_pred)
        self.last_FN = len(y_truth - y_pred)
        self.last_FP = len(y_pred - y_truth)
        self.sum_TP += self.last_TP
        self.sum_FN += self.last_FN
        self.sum_FP += self.last_FP
    def get_metric(self):
        # TP + FN 可能为0
        # TP + FP 可能为0
        TP = self.sum_TP
        FN = self.sum_FN
        FP = self.sum_FP
        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (recall + precision)
        self.recall = recall
        self.precision = precision
        return f1
    def get_detail(self):
        # if not hasattr(self, 'recall'):
        #     f1 = self.get_metric()
        f1 = self.get_metric()
        return f1, self.recall, self.precision
    def get_last(self):
        return self.last_TP, self.last_FN, self.last_FP

