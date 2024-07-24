
from config.configurator import configs
from clients.openai_client import OpenAIClient
from data_utils.data_handler_re import DataHandlerRE
from modules.prompt.prompter import BasePormpter
import re, json
from logging import getLogger

class BaseModel:
    """ 
    main functions
        process_sample(sample, idx) -> dict
            extract(text, idx) -> dict
        train_sample(sample, idx) -> dict
    utility functions
        parse_llm_output(text) -> tuple[int, list[dict]]
        query_llm(text, stop, temperature=None) -> str
        log_prompt(prompt) -> None
        set_training(is_training: bool) -> None
    """
    logger = getLogger('train_logger')
    data_handler: DataHandlerRE           # data handler
    prompter: BasePormpter                # prompter
    # is_training = True

    def __init__(self, data_handler=None):
        self.llm = self.load_llm()
        self.data_handler = data_handler

    def extract(self, text, idx):
        """ the main function for extraction
        return: 
            {
                "spo_list_pred": list[dict]  # the predicted triplets
                "errorCode": int             # error code
                "history": list[str]         # trajectories
                "final_output": str          # final output
            }
        """
        raise NotImplementedError

    def parse_llm_output(self, text: str) -> list[dict]:
        """ utlity function to parse LLM output
        text: str of json!
        """
        try:
            json_str = re.search(r'\{.*\}', text, flags=re.DOTALL).group()
            parsed = json.loads(json_str)           # JSON parse to get structured data
            spo_list = parsed['spo_list']           # update: result -> spo_list
            return 0, spo_list
        except Exception as e:
            spo_list = []
        return -1, spo_list

    def process_sample(self, sample, idx):
        """ function to process single sample
        input sample:
            {
                "text": str, 
                "spo_list": list[dict], 
            }
        """
        ret = self.extract(sample['text'], idx)
        return {
            "spo_list_pred": ret["spo_list_pred"],
            "errorCode": ret.get("errorCode", 0),
            "history": ret.get("history", []),
            "final_output": ret.get("final_output", "")
        }

    def parse_output(self, llm_output: str) -> tuple[int, list[dict]]:
        """ Parse the output of LLM
        """
        raise NotImplementedError


    def train_sample(self, samle, idx):
        raise NotImplementedError

    def load_llm(self):
        llm_configs = configs['llm']
        llm = OpenAIClient(
            model_name=llm_configs['model_name'],
            temperature=llm_configs['temperature'],
            max_tokens=llm_configs['max_tokens'],
        )
        return llm

    def query_llm(self, text, stop=None, temperature=None) -> str:
        res = self.llm.query_one(text, stop=stop, temperature=temperature)
        return res

    def log_prompt(self, prompt):
        self.logger.info(f"\n{'='*200}\nPrompt: {prompt}\n{'='*200}")

