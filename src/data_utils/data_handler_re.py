from config.configurator import configs
from modules.memory.memory import BaseMemory
from datasets import Dataset
import pandas as pd
import os
from easonsi.util.leetcode import *
from easonsi import utils
from enum import Enum
from logging import getLogger
logger = getLogger('train_logger')
rdir = os.path.dirname(os.path.dirname(
    os.path.dirname(__file__)
))


class DataName(Enum):
    """ standardized data name """
    DuIE2_0 = "DuIE2.0"
    SciERC = "SciERC"

class DataMeta:
    language: str = "zh"        # language
    model_name: str = "default" # model name

    ddir: str = ""              # data directory
    fn_schema: str              # schema filename
    fn_train: str               # train data filename
    fn_test: str                # test data filename

    odir: str = ""              # output directory
    ofn_pred: str               # output filename
    ofn_report: str             # output report filename

    def __init__(self, model_name:str="default") -> None:
        self.fn_schema = f"{self.ddir}/std_schema.json"
        self.fn_train = f"{self.ddir}/std_train.json"
        self.fn_test = f"{self.ddir}/std_test.json"

        self.model_name = model_name
        os.makedirs(f"{self.odir}/{self.model_name}", exist_ok=True)
        self.ofn_pred = f"{self.odir}/{self.model_name}/pred.json"
        self.ofn_report = f"{self.odir}/{self.model_name}/audit_report.json"



class DatMetaDuIE2_0(DataMeta):
    ddir = f"{rdir}/data/DuIE2.0"
    odir = f"{rdir}/out/DuIE2.0"
    def __init__(self, model_name):
        super().__init__(model_name)
        self.fn_test = f"{self.ddir}/std_sample.json"
        self.fn_train = f"{self.ddir}/std_dev.json"

class DatMetaSciERC(DataMeta):
    ddir = f"{rdir}/data/SciERC_sample_10000"
    odir = f"{rdir}/out/SciERC"
    def __init__(self, model_name):
        super().__init__(model_name)
        self.language = "en"


class DataHandlerRE:
    num_samples: int = -1           # Only use partial data, for debug
    num_samples_index: int = -1     # number of samples for index
    schema_dict: dict               # schema dict, Format as SciERC
    data_meta: DataMeta             # 

    ds_test: Dataset                # test data
    ds_pred: Dataset                # predicted
    ds_index: Dataset               # dataset for index

    correct_memory: BaseMemory      # memory for correct results
    reflexion_memory: BaseMemory    # memory for reflexion

    def __init__(self) -> None:
        model_name = configs['model']['name']
        if configs['data']['name'] == "DuIE2.0":
            self.data_meta = DatMetaDuIE2_0(model_name=model_name)
            self.data_name = DataName.DuIE2_0
        elif configs['data']['name'] == "SciERC":
            self.data_meta = DatMetaSciERC(model_name=model_name)
            self.data_name = DataName.SciERC
        else:
            raise Exception(f"Unknown dataset name {configs['data']['name']}")
        if 'num_samples' in configs['data'] and configs['data']['num_samples'] > 0:
            self.num_samples = configs['data']['num_samples']
        if 'num_samples_index' in configs['data'] and configs['data']['num_samples_index'] > 0:
            self.num_samples_index = configs['data']['num_samples_index']


    def load_data(self) -> None:
        # load evaluation data
        if self.num_samples > 0:                # Faster when only loading a few samples
            df_test = pd.read_json(self.data_meta.fn_test, lines=True, nrows=self.num_samples)
            self.ds_test = Dataset.from_pandas(df_test)
        else:
            self.ds_test = Dataset.from_json(self.data_meta.fn_test)
        # load index data
        if self.num_samples_index > 0:
            df_index = pd.read_json(self.data_meta.fn_train, lines=True, nrows=self.num_samples_index)
            self.ds_index = Dataset.from_pandas(df_index)
        else:
            self.ds_index = Dataset.from_json(self.data_meta.fn_train)
        # self.process_data()   # moved to -> class Processor
        self.load_schema()

    def load_schema(self) -> None:
        schemas = utils.LoadJsonl(self.data_meta.fn_schema)
        schema_dict = {}
        for schema in schemas:
            schema_dict[schema['predicate']] = schema
        self.schema_dict = schema_dict
    
    def get_relation_names(self) -> list:
        return list(self.schema_dict.keys())

    def save_results(self) -> None:
        df_pred = self.ds_pred.to_pandas()
        df_pred.to_json(self.data_meta.ofn_pred, orient='records', lines=True, force_ascii=False)
        logger.info(f"Save results to {self.data_meta.ofn_pred}")

    def load_results(self) -> None:
        self.ds_pred = Dataset.from_json(self.data_meta.ofn_pred)
        logger.info(f"Load results from {self.data_meta.ofn_pred}")



