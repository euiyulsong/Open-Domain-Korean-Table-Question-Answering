# ODQA
TAPT, Instruction Tuning, and Preference Optimization for Open Domain Question Answering

## Directory Structure
```
├── README.md
├── refine.py #perform data cleaning
└── requirements.txt
```

## Install Required Packages
```pip install -r requirements.txt```

## Gold Data Preprocessing
### Clean Data
```python src/clean/clean.py -d ${DATASET_PATH}```
### Map negative pairs and filter data with more than 1 gold table for Retrieval
```python src/clean/map.py```
### Clean corpus and map rejection samples for Preference Optmization
```python src/clean/negatives.py```
### Random Train/Test split
```bash src/split/split.sh```

## Synthetic Data Generation
## Question Generation
```python src/synthetic/question_generation.py```
## Answer Generation
```python src/synthetic/answer_generation.py```

## Retrieval
### Training
#### Dense Retrieval
```bash src/retrieval/dense_finetuning.sh```
#### Cross Encoder Reranker
```bash src/retrieval/ce_finetuning.sh```
### Inferenec3
```python src/retrieval/retrieval.py```



## SFT

## Instruction Tuning

## Simple Preference Optimization
