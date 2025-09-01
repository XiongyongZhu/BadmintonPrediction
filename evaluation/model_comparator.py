# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 09:14:17 2025
模型评估比较器
@author: dragon
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
from config.paths import PathConfig

logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
    
    def evaluate_all_models(self, trained_models, X_test, y_test):
        """评估所有模型并返回指标"""
        metrics_results = {}
        for model_name, model in trained_models.items():
            metrics = self.evaluator.evaluate(model, X_test, y_test)
            if metrics is not None:
                metrics_results[model_name] = metrics
                # 输出详细分类报告
                y_pred = model.predict(X_test)
                logger.info(f"\n=== {model_name} 的分类报告 ===")
                logger.info(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
                if 'roc_auc' in metrics:
                    logger.info(f"AUC: {metrics['roc_auc']:.4f}")
                if 'average_precision' in metrics:
                    logger.info(f"Average Precision: {metrics['average_precision']:.4f}")
            else:
                logger.error(f"评估模型 {model_name} 失败")
        return metrics_results
    
    def generate_performance_report(self, metrics_results):
        """生成性能报告"""
        performance_data = []
        for model_name, metrics in metrics_results.items():
            performance_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'ROC AUC': metrics.get('roc_auc', 'N/A'),
                'Train Time': metrics.get('train_time', 'N/A')
            })
        
        report_df = pd.DataFrame(performance_data)
        report_path = os.path.join(PathConfig.RESULTS_DIR, 'model_performance_comparison.csv')
        report_df.to_csv(report_path, index=False)
        logger.info(f"性能报告已保存: {report_path}")
        return report_path
    
    def plot_comparison_chart(self, metrics_results, metric_name='Accuracy'):
        """绘制模型比较图表"""
        try:
            model_names = list(metrics_results.keys())
            metric_values = [metrics[metric_name.lower()] for metrics in metrics_results.values()]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(model_names, metric_values, color='skyblue')
            plt.title(f'模型 {metric_name} 比较')
            plt.ylabel(metric_name)
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, metric_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.4f}', ha='center', va='bottom')
            
            chart_path = os.path.join(PathConfig.RESULTS_DIR, f'{metric_name}_comparison.png')
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"{metric_name} 比较图表已保存: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"生成比较图表失败: {e}")
            return None
