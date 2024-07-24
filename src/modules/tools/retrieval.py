from modules.tools.base_tool import BaseTool
from config.configurator import configs
from modules.retrieval.index import DummyIndex, SimCSEIndex, BGEIndex, BaseIndex, MODE2INDEX
from modules.module_utils import format_sample, format_sample_str
from logging import getLogger
logger = getLogger('train_logger')
import json
from datasets import Dataset


class RetrieveExamples(BaseTool):
    name: str = "RetrieveExamples"
    description_en: str = "Retrieve examples from the training dataset. The input is a sentence. "
    description_zh: str = "从训练数据集中召回相似的例子来帮助判断. 传入参数是句子, 可以对于当前的句子改写作为输入. "

    mode: str               # dummy/simcse/bge
    k: int                  # the max k for retrieval
    ds_index = None         # dataset for retrieval
    index:BaseIndex = None  # the index model

    def init(self):
        self.mode = configs['tools']['RetrieveExamples']['mode']     # dummy/simcse/bge
        self.k = configs['tools']['RetrieveExamples']['k'] 

        self.ds_index = self.data_handler.ds_index
        logger.info(f"RetrieveExamples: mode={self.mode}, k={self.k}, ds_index={len(self.ds_index)}")
        self.build_index()
        logger.info(f"RetrieveExamples: index built.")
    
    def build_index(self):
        index_texts = self.generate_index_texts(self.ds_index)
        self.index = MODE2INDEX[self.mode]()
        self.index.add(index_texts)

    def generate_index_texts(self, ds:Dataset):
        index_texts = [format_sample_str(i) for i in ds]
        return index_texts

    def call(self, query):
        matched_idxs = self.index.query_indexs(query, top_k=self.k)
        samples = self.ds_index.select(matched_idxs)
        samples_str = [format_sample(i) for i in samples]
        return json.dumps(samples_str, ensure_ascii=False)
