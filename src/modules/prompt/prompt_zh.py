TEMPLATE_REACT_ZH = \
"""尽你所能, 从给定的句子中识别出符合规范格式要求的关系三元组. 

你可以使用以下的工具: 
{tools}

使用如下的形式进行回答:
Thought: 思考下一步应该做什么
Action: 执行的动作名称, 需要在上面的工具列表中
ActionInput: 执行的动作所传入的一个参数, 可以为空
Observation: 执行动作后返回的结果
... (上面的 Thought/Action/ActionInput/Observation 三个步骤可以重复多次, 直到执行Finish动作返回结果)

Begin! 
输入的句子是 `{text}`\n
"""
FIRST_STEP_ZH = \
"""Thought: 首先，我需要了解更多关于关系三元组抽取任务的定义和输出格式的信息。
Action: GetTaskDescription
ActionInput:
Observation: {task_description}\n"""
SECOND_STEP_ZH = \
"""Thought: 我可以先观察一些已经标注好的关系三元组，以便更好地理解这个任务。
Action: RetrieveExamples
ActionInput: {text}
Observation: {retrieved_examples}\n"""
SUFFIX = """Thought: """

SECOND_STEP_MEMORY_ZH = \
"""Thought: 我可以找到已有的正确的例子来帮助我理解这个任务。
Action: RetrieveCorrectMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""

TEMPLATE_REFLEXION_ZH = \
"""在关系抽取任务中, 对于输入的句子 `{text}`, 正确的结果应该是 `{golden}`. 但模型输出的结果是 `{pred}`. 
请你用一句话来总结错误的原因: """

TEMPLATE_SUMMAY_ZH = \
"""在关系抽取任务中, 对于输入的句子 `{text}`, 正确的结果应该是 `{golden}`. 下面是可以参考的抽取过程: 
```
{history}
```
假如你无法在抽取过程中执行这些 Action, 需要直接生成抽取结果, 请用一句话给出你的推理依据, 并给出最终的JSON抽取结果: """