from modules.retrieval.index import DummyIndex, SimCSEIndex, BGEIndex, BaseIndex, MODE2INDEX
from config.configurator import configs

class BaseMemory:
    """ memory ~ RetrieveExamples
    variables
        name: name for identificatiion
        num_memory_items: 
    functions
        init() -> None: 
        add(items) -> None: 
        query(query, top_k=5) -> list[str]: 
    """
    name: str = "BaseMemory"
    index: BaseIndex = None

    memory_idx_mode: str           # dummy/simcse/bge
    memory_k: int                  # top_k

    def __init__(self):
        self.init()

    def init(self):
        raise NotImplementedError

    @property
    def num_memory_items(self):
        return self.index.num_indexed_items

    def add(self, items) -> None:
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            raise Exception("The input should be a list of strings.")
        self.index.add(items)

    def query(self, query, top_k=-1) -> list[str]:
        if top_k == -1:
            top_k = self.memory_k
        top_k = min(top_k, self.num_memory_items)
        if top_k <= 0:
            return []
        matched_idxs = self.index.query_indexs(query, top_k)
        return self.index.get_texts(matched_idxs)

class CorrectMemory(BaseMemory):
    def init(self):
        self.memory_idx_mode = configs['memory']['CorrectMemory']['mode']
        self.memory_k = configs['memory']['CorrectMemory']['k']
        self.index = MODE2INDEX[self.memory_idx_mode]()

class ReflexionMemory(BaseMemory):
    def init(self):
        self.memory_idx_mode = configs['memory']['ReflexionMemory']['mode']
        self.memory_k = configs['memory']['ReflexionMemory']['k']
        self.index = MODE2INDEX[self.memory_idx_mode]()
