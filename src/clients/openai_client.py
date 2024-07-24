""" 
repo: https://github.com/openai/openai-python
API: https://platform.openai.com/docs/guides/text-generation
"""

from diskcache import Cache
cache = Cache('./cache/gpt.cache')
from config.configurator import configs

GENERATIVE_MODELAS = ["gpt-3.5-turbo-instruct", ]

class OpenAIClient:
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 4096

    history: list = []      # conversation history

    use_cache: bool = configs['llm']['use_cache'] if 'use_cache' in configs['llm'] else False
    # cache_key: str = configs['llm']['cache_key'] if 'cache_key' in configs['llm'] else 'default_cache_key'

    def __init__(self, model_name:str=None, temperature:float=None, max_tokens:int=None):
        from openai import OpenAI
        self.client = OpenAI()
        if model_name: self.model_name = model_name
        if temperature: self.temperature = temperature
        if max_tokens: self.max_tokens = max_tokens
        self.is_generative_model = self.model_name in GENERATIVE_MODELAS

    def query_chat(self, text, stop=None, temperature=None) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{ "role": "user", "content": text,}],
            model=self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            stop=stop
        )
        return chat_completion.choices[0].message.content

    def query_generative(self, text, stop=None, temperature=None) -> str:
        completion = self.client.completions.create(
            prompt=text,
            model=self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,

            stop=stop
        )
        return completion.choices[0].text.strip()

    def query_one(self, text, stop=None, temperature=None) -> str:
        cache_key_ = f"{self.model_name}_{text}"
        if self.use_cache and cache_key_ in cache:
            return cache.get(cache_key_)
        else:
            if self.is_generative_model:
                res = self.query_generative(text, stop=stop, temperature=temperature)
            else:
                res = self.query_chat(text, stop=stop, temperature=temperature)
            cache.set(cache_key_, res)
        return res
    def query_one_stream(self, text) -> None:
        """ Just for test! """
        stream = self.client.chat.completions.create(
            messages=[{ "role": "user", "content": text,}],
            model=self.model_name,
            temperature=self.temperature,
            stream=True,
            stop=["\nObservation:"]
        )
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")

    def chat(self, text, stop=None, temperature=None) -> str:
        self.history.append({"role": "user", "content": text})
        chat_completion = self.client.chat.completions.create(
            messages=self.history,
            model=self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            stop=stop
        )
        res = chat_completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": res})
        return res

    def clear_history(self):
        self.history = []

    def chat_with_history(self, history:list, stop=None, temperature=None) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=history,
            model=self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            stop=stop
        )
        res = chat_completion.choices[0].message.content
        return res
