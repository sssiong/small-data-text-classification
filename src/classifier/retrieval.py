from dataclasses import dataclass
from collections import Counter
from typing import List

import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.sklearn import SKLearnVectorStore
from tqdm import tqdm


@dataclass
class RetrievalClassifier:
    k: int = 4
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'

    def _build_vector_store(self, texts: List[str], metadata: List[dict] = None):
        # https://www.sbert.net/docs/pretrained_models.html
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vector_store = SKLearnVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadata,
            serializer='parquet',
        )

    def fit(self, texts: List[str], y: pd.DataFrame):
        self.feature_names_in_ = y.columns.to_list()
        self.y_cnts = y.sum()
        y_list = y.apply(lambda row: row[row==1].index.to_list(), axis=1)
        self._build_vector_store(texts, [{'labels': l} for l in y_list])
        return self    

    def predict_proba(self, texts: List[str]) -> pd.DataFrame:
        counts = []
        for text in tqdm(texts):
            results = self.vector_store.similarity_search_with_relevance_scores(text, k=self.k)
            counter = Counter([
                label for result in results
                for label in result[0].metadata['labels']
            ])
            counts.append(counter)
        counts = pd.DataFrame(counts).fillna(0.)

        # scale between 0 and 1
        counts /= self.k

        # prepare output
        score = pd.DataFrame(columns=self.feature_names_in_)
        for name in self.feature_names_in_:
            if name in counts.columns:
                score[name] = counts[name]
            else:
                score[name] = 0.
        return score
    
    def predict(self, texts: List[str]) -> pd.DataFrame:
        y_score = self.predict_proba(texts)
        return (y_score >= 0.5).astype(int)
