# Open Domain Question Answering
RAFT, TAPT, Instruction Tuning, and Preference Optimization on [google/gemma-2b](https://huggingface.co/google/gemma-2b) for Open Domain Table Question Answering.

## Install Required Packages
```
pip install -r requirements.txt
python setup.py install
```

## Data Preprocessing
### Clean Data
```
python src/clean/clean.py # html parsing
python src/clean/map.py # map negative pairs and filter data with more than 1 gold table for Retrieval
python src/clean/negatives.py # clean corpus and map rejection samples for Preference Optmization
```
### Train/Test split
```
bash src/split/split.sh
```

### Synthetic Data Generation
```
python src/synthetic/question_generation.py # question generation
python src/synthetic/answer_generation.py #answer generation
python src/synthetic/refine_generation.py #self-refine generation
python src/synthetic/filter.py #consistent-based filtering (sort by l2 distance to the train label)
```

## Retrieval
### Data Preparation
```python src/clean/copy_train.py```
### Training
#### Dense Retrieval
```bash src/retrieval/dense_finetuning.sh```
#### Cross Encoder Reranker
```bash src/retrieval/ce_finetuning.sh```
### Inferenece
```python src/retrieval/retrieval.py```

## Reader
### Instruction Tuning 
```python src/trainer/sft.py --dataset_name ${INSTRUCTION_DATASET} --model_name ${BASE_MODEL}```

### Simple Preference Optimization
```python src/trainer/simpo.py```

### Odds Ratio Preference Optimization
```python src/trainer/orpo.py```
