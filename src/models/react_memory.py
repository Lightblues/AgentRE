
from config.configurator import configs
from models.base_model import BaseModel
import re, json
import importlib
from modules.tools import MEMORY_NAME2TOOL
from trainer.metrics_v2 import EvaluatorRE
from modules.memory.memory import CorrectMemory, BaseMemory, ReflexionMemory
from data_utils.data_handler_re import DataHandlerRE
from modules.module_utils import format_sample_str
from modules.prompt.prompter import PrompterReActMemory

SUCCESS = 0
NO_RESULT_WITHIN_MAX_ITERATIONS = -1
NO_VALID_RESULT_WITHIN_MAX_RETRY = -2



class ReAct_Memory(BaseModel):
    mode: str = configs["model"]["mode"] if "mode" in configs["model"] else "dummy"
    stop: str = ["Output:", "Observation:"]        # LLM stop
    max_iterations: int = configs["model"]["max_iterations"]
    max_retry: int = configs["model"]["max_retry"]
    num_pre_history: int = configs["model"]["num_pre_history"]
    use_summary: bool = configs["model"]["use_summary"]
    debug: bool = configs["model"]["debug"]

    history: list = []              # for recording the history, CLEARED in each iteration
    tools: dict = {}                # list of tools
    memory_names: list = []         # list of memories
    prompter: PrompterReActMemory   # 

    # is_training = configs['train']['if_train']
    evaluator:EvaluatorRE = EvaluatorRE()

    def __init__(self, data_handler:DataHandlerRE):
        super().__init__(data_handler)
        if configs['train']['if_predict'] or configs['train']['if_train']:
            self.init_memorys()
            self.init_tools()
        self.prompter = PrompterReActMemory(data_handler)

    def init_tools(self):
        tools_activated = []
        for tool_name in configs['tools'].keys():
            if configs['tools'][tool_name]['open']:
                # NEW: remove RetrieveExamples
                if tool_name in ["RetrieveExamples"]:
                    continue
                tools_activated.append(tool_name)
        # NEW: set tools according to memory
        for memory_name in self.memory_names:
            tools_activated.append(MEMORY_NAME2TOOL[memory_name])
        self.logger.info(f"Activated tools: {tools_activated}")

        module = importlib.import_module('modules.tools')
        for tool_name in tools_activated:
            tool = getattr(module, tool_name)(self.data_handler)
            self.tools[tool_name] = tool
        self.logger.info(f"Tools: {self.tools}")
    
    # NEW: init memory
    def init_memorys(self):
        if configs['memory']['CorrectMemory']['open']:
            self.memory_names.append('CorrectMemory')
            self.data_handler.correct_memory = CorrectMemory()

            # init correct memory with few-shot
            num_samples_init = configs['memory']['CorrectMemory']['num_samples_init']
            if num_samples_init > 0:
                self.logger.info(f"Init correct memory with {num_samples_init} samples.")
                samples_ds = self.data_handler.ds_index.select(range(num_samples_init))
                samples_list = [samples_ds[i] for i in range(num_samples_init)]
                self.record_correct_memory(samples_list)
        if configs['memory']['ReflexionMemory']['open']:
            self.memory_names.append('ReflexionMemory')
            self.data_handler.reflexion_memory = ReflexionMemory()

    # @log_exceptions
    def extract(self, text, idx):
        debug = True

        text = json.dumps(text.strip(), ensure_ascii=False)
        if debug: self.logger.info(f"[idx={idx}] Input: {text}")
        history = []       # clear the history!

        # ReAct-fashion
        for _ in range(self.max_iterations):
            prompt = self.generate_prompt(text)
            if idx < 5: self.log_prompt(prompt)
            for _ in range(self.max_retry):
                llm_output = self.query_llm(prompt, stop=self.stop, temperature=0.5)    # can try different parameters
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
                    "history": history.copy(),
                    "final_output": llm_output,
                    "errorCode": NO_VALID_RESULT_WITHIN_MAX_RETRY,
                }

            history.append(f"Thought: {thought}")
            if debug: self.logger.info(f"Thought: {thought}")
            if action_name == "Finish":
                err_code, spo_list = self.parse_llm_output(args)

                finish_output = json.dumps(args, ensure_ascii=False)
                history.append(f"Finish: {finish_output}")
                if debug: self.logger.info(f"Finish: {finish_output}")
                return {
                    "spo_list_pred": spo_list,
                    "history": history.copy(),
                    "final_output": llm_output,
                    "errorCode": err_code,
                }
            else:
                observation = self.tools[action_name].call(args)
                history.append(f"Action: {action_name}({args})")
                history.append(f"Observation: {observation}")
                if debug:
                    self.logger.info(f"Action: {action_name}({args})")
                    self.logger.info(f"Observation: {observation}")
        else:
            self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")
            return {
                "spo_list_pred": [],
                "history": history.copy(),
                "final_output": llm_output,
                "errorCode": NO_RESULT_WITHIN_MAX_ITERATIONS,
            }

    # @log_exceptions
    def train_sample(self, sample, idx):
        err_code = SUCCESS
        spo_list_pred = []
        summary_str = ""
        self.history = []

        text_str = json.dumps(sample['text'].strip(), ensure_ascii=False)
        if self.debug: self.logger.info(f"[idx={idx}] Input: {text_str}")

        # Outer loop: ReAct-fashion action search with max_iterations limit
        for _ in range(self.max_iterations):
            # [0] prompt 
            prompt = self.generate_prompt(text_str)
            # if idx < 5: self.log_prompt(prompt)
            # Inner loop: try single action with max_retry limit
            err_code_, parsed_res = self.get_single_step(prompt)
            if err_code != 0:
                err_code = err_code_
                break
            thought, action_name, args = parsed_res

            # [1] thought
            self.history.append(f"Thought: {thought}")
            if self.debug: self.logger.info(f"Thought: {thought}")

            # [2] action
            if action_name == "Finish":
                err_code, spo_list_pred = self.parse_llm_output(json.loads(args))
                if err_code < 0:
                    self.logger.error(f"[ERROR] error in parse_llm_output: {args}")
                self.history.append(f"Finish: {args}")
                if self.debug: self.logger.info(f"Finish: {args}")
                # NEW: add refexion!
                f1 = self.get_eval_result(sample['spo_list'], spo_list_pred)
                if f1 < 1.0:
                    reflexion_text = self.get_reflexion(text_str, sample['spo_list'], spo_list_pred)
                    self.history.append(f"Reflexion: {reflexion_text}")
                    if self.debug: self.logger.info(f"Reflexion: {reflexion_text}")
                else:
                    pass
                # NOTE: break when Finsh?
                break
            else:
                # If not "Finish", exec and generate observation
                observation = self.tools[action_name].call(args)
                self.history.append(f"Action: {action_name}({args})")
                self.history.append(f"Observation: {observation}")
                if self.debug:
                    self.logger.info(f"Action: {action_name}({args})")
                    self.logger.info(f"Observation: {observation}")
        else:
            err_code = NO_RESULT_WITHIN_MAX_ITERATIONS
            self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")
        
        # NEW: record into CorrectMemory!
        self.record_correct_memory(sample)

        # NEW: summary!
        if self.use_summary:
            summary_str = self.get_summary(text_str, sample['spo_list'], self.history)
            self.history.append(f"Summary: {summary_str}")
            if self.debug: self.logger.info(f"Summary: {summary_str}")
        return {
            "spo_list_pred": spo_list_pred,
            "history": self.history.copy(),
            "summary": summary_str,
            "errorCode": err_code,
        }

    def get_single_step(self, prompt):
        """ Try get single step action with self.max_retry
        return: err_code, (thought, action_name, args) 
        """
        for _ in range(self.max_retry):
            llm_output = self.query_llm(prompt, stop=self.stop, temperature=0.5)
            # 1. parse the output
            err_code, parsed_res = self.parse_output(llm_output)
            if err_code == -1:
                if self.debug: self.logger.error(f"error in parse_output: {llm_output}")
                continue
            # 2. Action need to be valid
            thought, action_name, args = parsed_res
            if action_name not in self.tools:
                if self.debug: self.logger.error(f"error action_name: {action_name}. llm_output: {llm_output}")
                continue
            # 3. if "Finish", try to parse LLM output
            if action_name == "Finish":
                err_code, spo_list_pred = self.parse_llm_output(args)
                if err_code == -1:
                    if self.debug: self.logger.error(f"error in parse_llm_output: {args}. llm_output: {llm_output}")
                    continue
            return 0, (thought, action_name, json.dumps(args, ensure_ascii=False))
        else:
            self.logger.error(f"[ERROR] Failed to generate valid output after 5 iterations.")
            return NO_VALID_RESULT_WITHIN_MAX_RETRY, (None, None, None)

    def record_correct_memory(self, sample):
        """ 
        sample: {text, spo_list} """
        if isinstance(sample, list):
            index_texts = [format_sample_str(s) for s in sample]
            self.data_handler.correct_memory.add(index_texts)
        elif isinstance(sample, dict):
            index_text = format_sample_str(sample)
            self.data_handler.correct_memory.add(index_text)
        else:
            raise Exception(f"Unknown sample type: {type(sample)}")

    def get_reflexion(self, text, gloden, pred):
        """ {text, xxx} """
        # prompt = TEMPLATE_REFLEXION.format(text=text, golden=json.dumps(gloden, ensure_ascii=False), pred=json.dumps(pred, ensure_ascii=False))
        prompt = self.prompter.get_reflexion_prompt(text, gloden, pred)
        llm_output = self.query_llm(prompt, stop=self.stop, temperature=0).strip()
        reflexion = {
            "text": text,
            "golden": gloden,
            "pred": pred,
            "reflexion": llm_output,
        }
        return json.dumps(reflexion, ensure_ascii=False)

    def get_summary(self, text, golden, history):
        """ {text, golden, history} """
        prompt = self.prompter.get_summary_prompt(text, golden, history)
        llm_output = self.query_llm(prompt, stop=self.stop, temperature=0).strip()
        return json.dumps(llm_output, ensure_ascii=False)

    def get_eval_result(self, golden, pred):
        """ 
        return: 
            triplet_correct, triplet_wrong ?
        """
        self.evaluator.add(golden, pred)
        last_TP, last_FN, last_FP = self.evaluator.get_last_metric()
        f1 = round(last_TP / (last_TP + 0.5 * (last_FP + last_FN)), 4)
        # precision = round(last_TP / (last_TP + last_FP), 4)
        # recall = round(last_TP / (last_TP + last_FN), 4)
        # f1 = round(2 * precision * recall / (precision + recall), 4)
        return f1

    def generate_prompt(self, text):
        tools_desc = "\n".join([f"- {tool.name}: {tool.get_description()}" for tool in self.tools.values()])
        task_description = self.tools['GetTaskDescription'].call()
        retrieved_examples = self.tools['RetrieveCorrectMemory'].call(text)
        prompt = self.prompter.get_react_prompt(text, tools_desc) + \
            self.prompter.get_react_first_step(task_description) + \
            self.prompter.get_react_second_step(text, retrieved_examples)
        if len(self.history) == 0:
            self.history.append(f"Action: GetTaskDescription()")
            self.history.append(f"Observation: {task_description}")
            self.history.append(f"Action: RetrieveCorrectMemory({text})")
            self.history.append(f"Observation: {retrieved_examples}")
            if self.debug:
                self.logger.info(f"Action: GetTaskDescription()")
                self.logger.info(f"Observation: {task_description}")
                self.logger.info(f"Action: RetrieveCorrectMemory({text})")
                self.logger.info(f"Observation: {retrieved_examples}")
        for history in self.history[self.num_pre_history * 2:]:             # Action+Observation
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


