
# AgentRE

This repository contains code for paper "**AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction**".

## Run

1. download the datasets: you can download `SciERC` from [here](http://nlp.cs.washington.edu/sciIE/) and `DuIE2.0` from [here](https://aistudio.baidu.com/competition/detail/65/0/introduction).
2. process the datasets: see [data_preprocessor.py](src/data_utils/data_preprocessor.py).
3. prepare the python environment: see [requirements.txt](requirements.txt).
4. config and run: select or make your own config file in `src/config` folder, and run with [main.py](src/main.py).

```sh
# a sample bash script to run
bash run.sh
```


## Cite

```bib
@inproceedings{shi2024agentre,
  author = {Yuchen Shi and Guochao Jiang and Tian Qiu and Deqing Yang},
  title = {AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24)},
  year = {2024},
  month = {October},
  publisher = {ACM},
  address = {Boise, ID, USA},
  doi = {10.1145/3627673.3679791},
  isbn = {979-8-4007-0436-9/24/10},
}

```
