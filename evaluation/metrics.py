# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:08:01 2025
# 评估指标
@author: dragon
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, model, X_test, y_test):
        """评估模型性能并返回指标"""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
            
            return metrics
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return None
    