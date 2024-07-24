
from config.configurator import configs
from models.base_model import BaseModel
import re, json
import importlib
from data_utils.data_handler_re import DataHandlerRE
from modules.module_utils import format_sample_str
from modules.prompt.prompter import PrompterReActFSL

SUCCESS = 0
NO_RESULT_WITHIN_MAX_ITERATIONS = -1
NO_VALID_RESULT_WITHIN_MAX_RETRY = -2



class ReAct_FSL(BaseModel):
    mode: str = configs["model"]["mode"] if "mode" in configs["model"] else "dummy"
    stop: str = ["Output:", "Observation:"]        # LLM stop
    max_iterations: int = configs["model"]["max_iterations"]
    max_retry: int = configs["model"]["max_retry"]
    history: list = []
    tools: dict = {}
    prompter: PrompterReActFSL


    def __init__(self, data_handler):
        super().__init__(data_handler)
        if configs['train']['if_predict'] or configs['train']['if_train']:
            self.init_tools()
        self.prompter = PrompterReActFSL(data_handler)

    def init_tools(self):
        tools_activated = []
        for tool_name in configs['tools'].keys():
            if configs['tools'][tool_name]['open']:
                tools_activated.append(tool_name)
        self.logger.info(f"Activated tools: {tools_activated}")
        module = importlib.import_module('modules.tools')
        for tool_name in tools_activated:
            tool = getattr(module, tool_name)(self.data_handler)
            self.tools[tool_name] = tool
        self.logger.info(f"Tools: {self.tools}")

    # @log_exceptions
    def extract(self, text, idx):
        debug = True

        text = json.dumps(text.strip(), ensure_ascii=False) 
        if debug: self.logger.info(f"[idx={idx}] Input: {text}")
        self.history = []

        for _ in range(self.max_iterations):
            prompt = self.generate_prompt(text)
            if idx < 5: self.log_prompt(prompt)
            for _ in range(self.max_retry):
                llm_output = self.query_llm(prompt, stop=self.stop, temperature=0.5)
                err_code, parsed_res = self.parse_output(llm_output)
                if err_code == -1:
                    if debug: self.logger.error(f"error in parse_output: {llm_output}")
                    continue
                thought, action_name, args = parsed_res
                if action_name not in self.tools:
                    if debug: self.logger.error(f"error action_name: {action_name}. llm_output: {llm_output}")
                    continue
                if action_name == "Finish":
                    err_code, spo_list = self.parse_llm_output(args)
                    if err_code == -1:
                        if debug: self.logger.error(f"error in parse_llm_output: {args}. llm_output: {llm_output}")
                        continue
                break
            else:
                self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")
                return {
                    "spo_list_pred": [],
                    "history": self.history.copy(),
                    "final_output": llm_output,
                    "errorCode": NO_VALID_RESULT_WITHIN_MAX_RETRY,
                }

            self.history.append(f"Thought: {thought}")
            if debug: self.logger.info(f"Thought: {thought}")
            if action_name == "Finish":
                err_code, spo_list = self.parse_llm_output(args)

                finish_output = json.dumps(args, ensure_ascii=False)
                self.history.append(f"Finish: {finish_output}")
                if debug: self.logger.info(f"Finish: {finish_output}")
                return {
                    "spo_list_pred": spo_list,
                    "history": self.history.copy(),
                    "final_output": llm_output,
                    "errorCode": err_code,
                }
            else:
                observation = self.tools[action_name].call(args)
                self.history.append(f"Action: {action_name}({args})")
                self.history.append(f"Observation: {observation}")
                if debug:
                    self.logger.info(f"Action: {action_name}({args})")
                    self.logger.info(f"Observation: {observation}")
        else:
            self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")
            return {
                "spo_list_pred": [],
                "history": self.history.copy(),
                "final_output": llm_output,
                "errorCode": NO_RESULT_WITHIN_MAX_ITERATIONS,
            }

    def generate_prompt(self, text):
        tools_desc = "\n".join([f"- {tool.name}: {tool.get_description()}" for tool in self.tools.values()])
        task_description = self.tools['GetTaskDescription'].call()
        retrieved_examples = self.tools['RetrieveExamples'].call(text)
        prompt = self.prompter.get_react_prompt(text, tools_desc) + \
            self.prompter.get_react_first_step(task_description) + \
            self.prompter.get_react_second_step(text, retrieved_examples)
        for history in self.history:
            prompt += history + "\n" 
        prompt += self.prompter.get_react_suffix()
        return prompt

    def parse_output(self, llm_output: str):
        try:
            # regex = r"(.*?)\nAction:(.*?)\nActionInput:[\s]*(.*)"
            regex = r"(.*?)Action:(.*?)\nActionInput:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            thought = match.group(1).strip()
            action = match.group(2).strip()
            args = match.group(3).strip()
            thought = json.dumps(thought, ensure_ascii=False)
            return 0, (thought, action, args)

        except Exception as e:
            return -1, None


