import time
from typing import Any, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

def evaluate_clf(clf: Any, X: List[str], y_true: pd.DataFrame):

    t = time.time()
    name = clf.__class__.__name__
    y_pred = clf.predict(X)
    print(f'{name} predict completed in {time.time()-t:.1f}s')

    params = {'y_true': y_true, 'y_pred': y_pred}
    overall = {
        'clf': name,
        'accuracy': accuracy_score(**params),
        'f1_score': f1_score(**params, average='macro'),
        'precision': precision_score(**params, average='macro'),
        'recall': recall_score(**params, average='macro'),
    }

    target_names = y_true.columns.to_list()
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names, output_dict=True)
    report = pd.DataFrame(report).T.assign(clf=name).reset_index(names='label').query('label == @target_names')
    return overall, report
