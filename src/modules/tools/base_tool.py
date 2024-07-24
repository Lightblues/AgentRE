from logging import getLogger
logger = getLogger('train_logger')
from data_utils.data_handler_re import DataHandlerRE

class BaseTool:
    """ 
    functions
        call(*args, **kwargs) -> str
        init() -> None
    """
    name: str = "BaseTool"
    language: str = "zh"
    # api: str = "BaseTool(args)"
    data_handler: DataHandlerRE
    description_en: str = "This is a base tool."
    description_zh: str = "这是一个基础工具."

    def __init__(self, data_handler=None):
        self.data_handler = data_handler
        self.language = data_handler.data_meta.language
        self.init()

    def call(self, *args, **kwargs) -> str:
        raise NotImplementedError(f"Method call is not implemented for {self.name}")

    def init(self):
        pass

    def get_description(self):
        if self.language == "zh":
            return self.description_zh
        else:
            return self.description_en
    @property
    def description(self):
        return self.get_description()

class Finish(BaseTool):
    name = "Finish"
    description_zh = "完成任务. 传入最终的结果作为输出. "
    description_en = "Finish the task. Return the final result. "

    def call(self, *args, **kwargs):
        return "Finish!"

