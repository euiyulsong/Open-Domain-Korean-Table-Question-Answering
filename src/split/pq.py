import logging
import argparse
import json
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sklearn.mixture import GaussianMixture
import numpy as np
class ProductQuantizer:
    def __init__(self, num_subvectors, num_clusters):
        self.num_subvectors = num_subvectors
        self.num_clusters = num_clusters
        self.codebooks = []

    def fit(self, X):
        n, d = X.shape
        subvector_dim = d // self.num_subvectors
        self.codebooks = []

        for i in range(self.num_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim
            subvector = X[:, start:end]

            kmeans = GaussianMixture(n_components=self.num_clusters)
            kmeans.fit(subvector)
            self.codebooks.append(kmeans)

    def encode(self, X):
        n, d = X.shape
        subvector_dim = d // self.num_subvectors
        encoded = np.zeros((n, self.num_subvectors), dtype=np.int32)

        for i in range(self.num_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim
            subvector = X[:, start:end]

            kmeans = self.codebooks[i]
            encoded[:, i] = kmeans.predict(subvector)

        return encoded

    def decode(self, encoded):
        n, _ = encoded.shape
        subvector_dim = self.codebooks[0].cluster_centers_.shape[1]
        d = subvector_dim * self.num_subvectors
        decoded = np.zeros((n, d))

        for i in range(self.num_subvectors):
            start = i * subvector_dim
            end = (i + 1) * subvector_dim

            kmeans = self.codebooks[i]
            centers = kmeans.cluster_centers_
            decoded[:, start:end] = centers[encoded[:, i]]

        return decoded

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", help="Input filename to clean", type=str, default="/mnt/c/Users/thddm/Documents/dataset/cleaned_dataset.jsonl", required=False)
    parser.add_argument("-o", "--output_dir", help="Output filename for mapping product quantized embedding", type=str, default="/mnt/c/Users/thddm/Documents/dataset/pq_cleaned_dataset.jsonl", required=False)
    args = parser.parse_args()
    input_dir = open(args.input_dir, "r")
    output_dir = open(args.output_dir, "w", encoding="utf-8")
    model = BGEM3FlagModel("BAAi/bge-m3", use_fp16=True)
    inputs = []
    questions = set()
    caches = []
    for i in tqdm(input_dir):
        i = json.loads(i)
        caches.append(i)
        current = f"Question: " + i['question'] + "\nContext: " + i['table'] + "\nAnswer: " + i['answer']
        if i['question'] in questions:
            raise Exception('Duplicated question exists')
        questions.add(i['question'])
        inputs.append(current)
    embeddings = model.encode(inputs, max_length=8192)['dense_vecs']
    pq = ProductQuantizer(num_subvectors=8, num_clusters=512)
    pq.fit(embeddings)
    encoded_X = pq.encode(embeddings)

    for pq, (cache, embedding) in tqdm(zip(encoded_X, zip(caches, embeddings))):
        cache['pq'] = [int(p) for p in list(pq)]
        output_dir.write(json.dumps(cache, ensure_ascii=False) + "\n")
    output_dir.close()

