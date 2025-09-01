# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:09:02 2025
# 报告生成
@author: dragon
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import os
import shap
from datetime import datetime
from config.paths import PathConfig
from .explainability_report import ExplainabilityReport

logger = logging.getLogger(__name__)

class ComprehensiveReportGenerator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
    
    def generate_comprehensive_report(self, models, metrics_results, X_test, y_test, feature_names):
        """生成综合报告"""
        try:
            # 1. 生成性能比较报告
            performance_df = self._create_performance_dataframe(metrics_results)
            
            # 2. 生成特征重要性报告
            feature_importance_df = self._create_feature_importance_report(models, feature_names)
            
            # 3. 生成最佳模型识别
            best_model_info = self._identify_best_model(metrics_results)
            
            # 4. 保存所有报告
            report_data = {
                'performance': performance_df,
                'feature_importance': feature_importance_df,
                'best_model': best_model_info,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self._save_reports(report_data)
            
            logger.info("综合报告生成完成")
            return report_data
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return None
    
    def _create_performance_dataframe(self, metrics_results):
        """创建性能数据框"""
        performance_data = []
        for model_name, metrics in metrics_results.items():
            performance_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1': f"{metrics['f1']:.4f}",
                'ROC_AUC': f"{metrics.get('roc_auc', 'N/A'):.4f}" if metrics.get('roc_auc') else 'N/A'
            })
        return pd.DataFrame(performance_data)
    
    def add_explainability_section(self, model, X_test, y_test, output_path=None):
        """
        添加模型可解释性分析章节
        
        参数:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            output_path: 报告输出路径
        """
        # 初始化解释器
        explainer = ExplainabilityReport(
            feature_names=X_test.columns.tolist()
        )
        
        # 生成全局解释
        global_report = explainer.generate_shap_global_report(model, X_test)
        
        # 生成局部解释（以第一个测试样本为例）
        sample_instance = X_test.iloc[0].values
        local_explanation = explainer.generate_local_explanation(
            model, sample_instance, X_test
        )
        
        # 生成可读报告
        readable_report = explainer.generate_readable_report(local_explanation)
        
        # 将报告内容添加到现有报告中
        self.report_content += "\n\n" + readable_report
        
        # 保存特征重要性可视化
        plt.figure(figsize=(10, 8))
        shap.summary_plot(global_report['shap_values'], X_test, show=False)
        plt.savefig(f"{output_path}/feature_importance_summary.png", bbox_inches='tight')
        plt.close()
        
        return self.report_content
    
    def _create_feature_importance_report(self, models, feature_names):
        """创建特征重要性报告"""
        # 实现特征重要性分析逻辑
        pass
    
    def _identify_best_model(self, metrics_results):
        """识别最佳模型"""
        best_model = None
        best_score = 0
        
        for model_name, metrics in metrics_results.items():
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_model = model_name
        
        return {'model': best_model, 'accuracy': best_score}
    
    def _save_reports(self, report_data):
        """保存报告"""
        # 保存性能报告
        performance_path = os.path.join(PathConfig.REPORTS_DIR, 'performance_summary.csv')
        report_data['performance'].to_csv(performance_path, index=False)
        
        # 保存最佳模型信息
        best_model_path = os.path.join(PathConfig.REPORTS_DIR, 'best_model.txt')
        with open(best_model_path, 'w') as f:
            f.write(f"最佳模型: {report_data['best_model']['model']}\n")
            f.write(f"准确率: {report_data['best_model']['accuracy']:.4f}\n")
            f.write(f"生成时间: {report_data['timestamp']}")

    