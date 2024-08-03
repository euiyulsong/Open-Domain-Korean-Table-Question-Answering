# Open Domain Question Answering
RAFT, TAPT, Instruction Tuning, and Preference Optimization on [google/gemma-2b](https://huggingface.co/google/gemma-2b) for Open Domain Question Answering
## Performance
### Hyperparameters
| Hp | Retrieval | Reranker | Reader | 
| ------------- | ------------- | ------------- | ------------- |
| lr |  |  |  |
| bs |  |  |  |
| scheduler |  |  |  |
| gpu | rtx 4090  | rtx 4090  | rtx 4090  |
| epoch |  |  |  |
| input maxlength |  |  |  |
| batch size |   |  |  |
| hard neg | 8 |  |  |
| τ | 0.02 |  |  |
### Retrieval
| Method | R-Precision | Recall@5 | TPS | Training Time |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| BM25 | 67.018 | 91.265 | 0.007s | 0s |
| Dense |  73.343 | 95.03 | 0.44s | 858.0121s |
| Cross Encoder | 86.747 | 96.084 | 1.30s | 800.346s |

### Reader

#### google/gemma-2b (float16)
| Method | EM | F-1 | Rouge-L | Training Time | 
| ------------- | ------------- | ------------- |------------- | ------------- |
| SFT (Close-book) | 0 |  |  | 51.81 |
| TAPT + SFT (Close-book) | 0 |   |  | 514.2573 + 51.81 |
| SFT (Open-book) | 0 |   |  | 335.8971 |
| Instruction Tuning | 0 | 3.267  | 60.462 | 8,017.02 |
| Instruction Tuning + Synthetic SFT | 0  | 3.353  | 61.273 | 3034.3011 |
| Instruction Tuning + Synthetic SFT + SFT  |  0 |   |  | 335.8971 |
| Instruction Tuning + Synthetic SFT + SFT + SIMPO |  0 |   |  | 2069.1573 |



## Directory Structure
```
├── README.md
├── refine.py #perform data cleaning
└── requirements.txt
```

## Install Required Packages
```pip install -r requirements.txt```

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



## SFT

## Instruction Tuning

## Simple Preference Optimization
