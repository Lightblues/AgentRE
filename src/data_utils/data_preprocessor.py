""" Preprocessor for DuIE and SciERC! 

"""

from easonsi.util.leetcode import *
from easonsi import utils
from datasets import Dataset
import pandas as pd
import os, sys, json

class Processor:

    @staticmethod
    def f_process_duie_sample(sample):
        spo_list_new = []
        for spo in sample['spo_list']:
            spo_list_new.append({
                "subject": spo['subject'],
                "predicate": spo['predicate'],
                "object": spo['object']['@value']
            })
        return {'spo_list': spo_list_new}
    @staticmethod
    def f_process_duie_schema(schema) -> None:
        schema['object_type'] = schema['object_type']['@value']
        return schema

    def process_duie(self):
        ddir = "/home/ubuntu/work/agent/AgentIE/data/DuIE2.0"
        for fn in ['duie_sample.json', 'duie_dev.json']:
            ofn = f"{ddir}/std_{fn}"
            if os.path.exists(ofn):
                continue
            # ds = Dataset.from_json(f"{ddir}/{fname}")     # has bug!!!
            df = pd.read_json(f"{ddir}/{fn}", lines=True)
            ds = Dataset.from_pandas(df)
            ds_processed = ds.map(self.f_process_duie_sample)
            ds_processed.to_json(ofn, orient="records", lines=True, force_ascii=False)
            print(f"Saved to {ofn}")
        fn_schema = f"duie_schema.json"
        ds_schema = Dataset.from_json(f"{ddir}/{fn_schema}")
        ds_schema_processed = ds_schema.map(self.f_process_duie_schema)
        ds_schema_processed.to_json(f"{ddir}/std_{fn_schema}", orient="records", lines=True, force_ascii=False)
        print(f"Saved to {ddir}/std_{fn_schema}")


    @staticmethod
    def f_process_scierc_sample(sample):
        spo_list_new = []
        for spo in sample['spo_list']:
            spo_list_new.append({
                "subject": spo['head']['name'],
                "predicate": spo['type'],
                "object": spo['tail']['name']
            })
        return {'spo_list': spo_list_new}
    
    @staticmethod
    def f_process_scierc_schema(schema) -> None:
        schema_new = {
            "object_type": "Any",
            "predicate": schema['predicate'],
            "subject_type": "Any"
        }
        return schema_new

    def process_sciERC(self):
        ddir = "/home/ubuntu/work/agent/AgentIE/data/SciERC_sample_10000"
        for fn in ['test.json', 'train.json']:
            ofn = f"{ddir}/std_{fn}"
            if os.path.exists(ofn):
                continue
            d_list = utils.LoadJson(f"{ddir}/{fn}")
            ds = Dataset.from_dict({
                "text": [d['sentence'] for d in d_list],
                "spo_list": [d['relations'] for d in d_list]
            })
            ds_processed = ds.map(self.f_process_scierc_sample)
            ds_processed.to_json(ofn, orient="records", lines=True, force_ascii=False)
            print(f"Saved to {ofn}")
        schema_name_list = utils.LoadJson(f"{ddir}/labels.json")
        ds_schema = Dataset.from_dict({
            "predicate": schema_name_list
        })
        ds_schema_processed = ds_schema.map(self.f_process_scierc_schema)
        ds_schema_processed.to_json(f"{ddir}/std_schema.json", orient="records", lines=True, force_ascii=False)
        print(f"Saved to {ddir}/std_schema.json")
        


processor = Processor()
# processor.process_duie()
processor.process_sciERC()
print("Done!")