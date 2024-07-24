from modules.tools.base_tool import BaseTool
import json

class GetTaskDescription(BaseTool):
    name = "GetTaskDescription"
    # api = "GetTaskDescription()"
    description_zh = "获取关系三元组抽取的任务定义和输出格式. 函数没有参数. "
    description_en = "Get the task description and output format of relation triple extraction. The function has no parameters. "

    def call(self, args=None):
        if self.language == "zh":
            TASK_DESC = TASK_DESC_ZH
        elif self.language == "en":
            TASK_DESC = TASK_DESC_EN
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        relation_names = "|".join(self.data_handler.get_relation_names())
        task_desc = TASK_DESC.format(relation_names=relation_names)
        return json.dumps(task_desc, ensure_ascii=False)

class GetRelationDefinition(BaseTool):
    name = "GetRelationDefinition"
    # api = "GetRelationDefinition()"
    description_zh = "获取关系类别的定义. 传入需要查询的关系类别, 可以是多个类别, 例如 `创始人|号` 查询两个关系类别. "
    description_en = "Get the definition of the relation category. Pass in the relation category to be queried, which can be multiple categories, such as `feature of|conjunction` to query two relation categories. "

    def call(self, args=None):
        relation_names = args.strip().strip("\"").split('|')
        res = {}
        schema_dict = self.data_handler.schema_dict
        for relation_name in relation_names:
            if relation_name in schema_dict:
                res[relation_name] = schema_dict[relation_name]
            else:
                res[relation_name] = "Not Found"
        return json.dumps(res, ensure_ascii=False)


# ['毕业院校', '嘉宾', '配音', '主题曲', '代言人', '所属专辑', '父亲', '作者', '上映时间', '母亲', '专业代码', '占地面积', '邮政编码', '票房', '注册资本', '主角', '妻子', '编剧', '气候', '歌手', '获奖', '校长', '创始人', '首都', '丈夫', '朝代', '饰演', '面积', '总部地点', '祖籍', '人口数量', '制片人', '修业年限', '所在城市', '董事长', '作词', '改编自', '出品公司', '导演', '作曲', '主演', '主持人', '成立日期', '简称', '海拔', '号', '国籍', '官方语言’]
TASK_DESC_ZH = """目标: 根据给定的句子, 识别其中的关系三元组. 
输出形式: 
{{
    "spo_list": [
        {{"subject": "xxx", "predicate": "xxx", "object": "xxx"}}
    ]
}}

注意: 
1. 以JSON格式输出, 不要包含其他的任何内容. 
2. spo_list中输出抽取的三元组, 若存在多个关系三元组则输出多个. 关系类别必须要在以下列表中: `{relation_names}`"""

TASK_DESC_EN = """Objective: Identify the relationship triples in the given sentence.
Output format:
{{
    "spo_list": [
        {{"subject": "xxx", "predicate": "xxx", "object": "xxx"}}
    ]
}}
Note:
1. Output in JSON format, do not include any other content.
2. The spo_list outputs the extracted triples. If there are multiple relationship triples, output multiple. The predicate must be in the following list: `{relation_names}`"""
