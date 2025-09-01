# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:08:39 2025
# 可视化
@author: dragon
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

class Visualization:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt
    
    def plot_roc_curve(self, y_true, y_pred_proba, title="ROC Curve"):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        return plt
        
    def plot_metrics_comparison(self, metrics_results):
        """绘制模型性能对比图"""
        metrics_df = pd.DataFrame(metrics_results).T
        metrics_df[['accuracy', 'precision', 'recall', 'f1']].plot(
            kind='bar', figsize=(12, 6)
        )
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/metrics_comparison.png')
        plt.close()
    
    def plot_feature_importance(self, models, feature_names):
        """
        绘制每个模型的特征重要性图（如果模型支持）
        :param models: 字典，包含模型名称和模型对象
        :param feature_names: 列表，特征名称
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        for model_name, model in models.items():
            try:
                # 检查模型是否有feature_importances_属性（树模型）
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    plt.figure(figsize=(10, 6))
                    plt.title(f"Feature Importance - {model_name}")
                    plt.bar(range(len(importances)), importances[indices])
                    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
                    plt.tight_layout()
                    plt.savefig(f"results/feature_importance_{model_name}.png")
                    plt.close()
                    # 检查模型是否有coef_属性（线性模型）
                elif hasattr(model, 'coef_'):
                    coef = model.coef_[0]  # 假设二分类，取第一个类的系数
                    indices = np.argsort(np.abs(coef))[::-1]
                    plt.figure(figsize=(10, 6))
                    plt.title(f"Feature Coefficients - {model_name}")
                    plt.bar(range(len(coef)), np.abs(coef)[indices])  # 取绝对值用于重要性排序
                    plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=45)
                    plt.tight_layout()
                    plt.savefig(f"results/feature_coefficients_{model_name}.png")
                    plt.close()
                else:
                    # 对于不支持特征重要性的模型（如SVM with RBF），跳过或使用SHAP
                    print(f"Model {model_name} does not support feature importances or coefficients.")
            except Exception as e:
                print(f"Error plotting feature importance for {model_name}: {e}")
