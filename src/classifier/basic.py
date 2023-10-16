from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

@dataclass
class BasicClassifier:

    def fit(self, texts: List[str], y: pd.DataFrame):
        self.clf_ = OneVsRestClassifier(Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
        ]))
        self.clf_.fit(texts, y)
        return self
    
    def predict_proba(self, texts: List[str]) -> pd.DataFrame:
        return self.clf_.predict_proba(texts)
    
    def predict(self, texts: List[str]) -> pd.DataFrame:
        return self.clf_.predict(texts)
    