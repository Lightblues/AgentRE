
from modules.tools import GetTaskDescription
from logging import getLogger
from data_utils.data_handler_re import DataHandlerRE
from .prompt_zh import *
from .prompt_en import *
import json
from typing import List



class BasePormpter:
    def __init__(self, data_handler):
        self.data_handler:DataHandlerRE = data_handler
        self.logger = getLogger('train_logger')
        self.language = data_handler.data_meta.language


class PrompterReActFSL(BasePormpter):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        if self.language == "zh":
            self.TEMPLATE_REFLEXION = TEMPLATE_REFLEXION_ZH
            self.TEMPLATE_REACT = TEMPLATE_REACT_ZH
            self.FIRST_STEP = FIRST_STEP_ZH
            self.SECOND_STEP = SECOND_STEP_ZH
        elif self.language == "en":
            self.TEMPLATE_REFLEXION = TEMPLATE_REFLEXION_EN
            self.TEMPLATE_REACT = TEMPLATE_REACT_EN
            self.FIRST_STEP = FIRST_STEP_EN
            self.SECOND_STEP = SECOND_STEP_EN
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        self.SUFFIX = SUFFIX

    def get_react_prompt(self, text:str, tools_desc:str):
        return self.TEMPLATE_REACT.format(tools=tools_desc, text=text)
    def get_react_first_step(self, task_description:str):
        return self.FIRST_STEP.format(task_description=task_description)
    def get_react_second_step(self, text:str, retrieved_examples:str):
        return self.SECOND_STEP.format(text=text, retrieved_examples=retrieved_examples)
    def get_react_suffix(self):
        return SUFFIX


class PrompterReActMemory(BasePormpter):
    """ 相较于 PrompterReActFSL
    - second step 中的召回接口不同
    - 增加 Refelxion
    """
    def __init__(self, data_handler):
        super().__init__(data_handler)
        if self.language == "zh":
            self.TEMPLATE_REFLEXION = TEMPLATE_REFLEXION_ZH
            self.TEMPLATE_REACT = TEMPLATE_REACT_ZH
            self.FIRST_STEP = FIRST_STEP_ZH
            self.SECOND_STEP = SECOND_STEP_MEMORY_ZH
            self.TEMPLATE_SUMMAY = TEMPLATE_SUMMAY_ZH
        elif self.language == "en":
            self.TEMPLATE_REFLEXION = TEMPLATE_REFLEXION_EN
            self.TEMPLATE_REACT = TEMPLATE_REACT_EN
            self.FIRST_STEP = FIRST_STEP_EN
            self.SECOND_STEP = SECOND_STEP_MEMORY_EN
            self.TEMPLATE_SUMMAY = TEMPLATE_SUMMAY_EN
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        self.SUFFIX = SUFFIX

    def get_react_prompt(self, text:str, tools_desc:str):
        return self.TEMPLATE_REACT.format(tools=tools_desc, text=text)
    def get_react_first_step(self, task_description:str):
        return self.FIRST_STEP.format(task_description=task_description)
    def get_react_second_step(self, text:str, retrieved_examples:str):
        return self.SECOND_STEP.format(text=text, retrieved_examples=retrieved_examples)
    def get_react_suffix(self):
        return SUFFIX

    def get_reflexion_prompt(self, text:str, golden:str, pred:str):
        golden, pred = json.dumps(golden, ensure_ascii=False), json.dumps(pred, ensure_ascii=False)
        return self.TEMPLATE_REFLEXION.format(text=text, golden=golden, pred=pred)
    def get_summary_prompt(self, text:str, golden:str, history:List[str]):
        if isinstance(golden, list):
            golden = json.dumps(golden, ensure_ascii=False)
        history = "\n".join(history)
        return self.TEMPLATE_SUMMAY.format(text=text, golden=golden, history=history)


