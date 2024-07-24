TEMPLATE_REACT_EN = \
"""Do your best to identify the relation triples that meet the standard format requirements from the given sentence.

You can use the following tools:
{tools}

Answer in the following format:
Thought: Think about what to do next
Action: The name of the action to be performed, which needs to be in the list of tools above
ActionInput: A parameter passed to the action, which can be empty
Observation: The result returned after the action is performed
... (The above Thought/Action/ActionInput/Observation three steps can be repeated multiple times until the Action of Finish that returns the extraction result as ActionInput.)

Begin!
The input sentence is `{text}`\n
"""
FIRST_STEP_EN = \
"""Thought: First, I need to know more about the definition and output format of the relation triple extraction task.
Action: GetTaskDescription
ActionInput:
Observation: {task_description}\n"""
SECOND_STEP_EN = \
"""Thought: I can first observe some already labeled relation triples to better understand this task.
Action: RetrieveExamples
ActionInput: {text}
Observation: {retrieved_examples}\n"""
SUFFIX = """Thought: """

SECOND_STEP_MEMORY_EN = \
"""Thought: I can find some examples from the existing correct examples to help me understand this task.
Action: RetrieveCorrectMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""

TEMPLATE_REFLEXION_EN = \
"""In the relation extraction task, for the input sentence `{text}`, the correct result should be `{golden}`. But the model's output result is `{pred}`.
Please summarize the reason for the error in one sentence: """

TEMPLATE_SUMMAY_EN = \
"""In the relation extraction task, for the input sentence `{text}`, the correct result should be `{golden}`. Here is the extraction process that can be referred to:
```
{history}
```
If you cannot perform these actions in the extraction process and need to directly generate the extraction result, please give your reasoning in one sentence and give the final JSON extraction result: """