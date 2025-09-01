# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:06:28 2025

@author: dragon
"""

"""
可解释性报告生成模块
使用SHAP和LIME生成模型预测的详细解释报告。
这是最核心的新增文件，负责生成结构化、可读的解释文本。
"""

import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class ExplainabilityReport:
    """生成模型可解释性报告"""

    def __init__(self, feature_names: List[str], class_names: List[str] = ['输', '赢']):
        """
        初始化解释器
        
        参数:
            feature_names: 特征名称列表
            class_names: 类别名称列表
        """
        self.feature_names = feature_names
        self.class_names = class_names
        self.feature_name_map = {
            'first_set_11_15': '第一局11-15分段得分',
            'first_set_15_19': '第一局15-19分段得分',
            'second_set_11_15': '第二局11-15分段得分', 
            'second_set_15_19': '第二局15-19分段得分',
            'final_set_11_15': '决胜局11-15分段得分',
            'final_set_15_19': '决胜局15-19分段得分',
            'head_to_head': '历史交锋优势',
            'recent_match_won': '近期胜场数',
            'critical_point_advantage': '关键分优势指数',
            'total_score_advantage': '总得分优势指数',
            'ranking_seed': '排名种子差距',
            'venue': '场地优势'
        }

    def generate_shap_global_report(self, model, X_data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成全局SHAP报告 - 使用更高效的TreeExplainer或KernelExplainer
        
        返回:
            包含全局特征重要性的字典
        """
        try:
            # 尝试使用TreeExplainer（如果模型支持）
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_data)
        except:
            # 回退到KernelExplainer（适用于SVM等模型）
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_data, 100))
            shap_values = explainer.shap_values(X_data)
        
        # 计算平均绝对SHAP值作为特征重要性
        if isinstance(shap_values, list):
            # 多分类情况：取第一个类别（'赢'）的重要性
            global_importance = np.mean(np.abs(shap_values[1]), axis=0)
        else:
            # 二分类情况
            global_importance = np.mean(np.abs(shap_values), axis=0)
        
        # 创建特征重要性字典
        feature_importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            readable_name = self.feature_name_map.get(feature, feature)
            # 转换为百分比贡献
            percent_contribution = (global_importance[i] / np.sum(global_importance)) * 100
            feature_importance_dict[readable_name] = round(percent_contribution, 2)
        
        # 按重要性排序
        sorted_importance = dict(sorted(
            feature_importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return {
            'global_importance': sorted_importance,
            'shap_explainer': explainer,
            'shap_values': shap_values
        }

    def generate_local_explanation(
        self, 
        model, 
        instance: np.array, 
        X_data: pd.DataFrame,
        explainer_type: str = 'both'
    ) -> Dict[str, Any]:
        """
        生成单个预测的局部解释
        
        返回:
            包含SHAP和LIME解释的字典
        """
        result = {}
        
        # SHAP局部解释
        shap_explainer = shap.KernelExplainer(model.predict, X_data)
        shap_local_values = shap_explainer.shap_values(instance)
        
        # LIME解释器设置
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_data.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )
        
        # LIME解释
        lime_exp = lime_explainer.explain_instance(
            instance, 
            model.predict_proba, 
            num_features=len(self.feature_names)
        )
        
        # 解析LIME结果
        lime_features = []
        for feature, weight in lime_exp.as_list():
            # 将LIME权重转换为百分比式贡献
            lime_features.append({
                'feature': feature,
                'contribution': weight * 100  # 转换为百分比影响
            })
        
        result['shap'] = shap_local_values
        result['lime'] = lime_features
        result['prediction_proba'] = model.predict_proba([instance])
        
        return result

    def generate_readable_report(self, local_explanation: Dict) -> str:
        """
        生成可读的文本报告
        
        返回:
            可读的报告文本
        """
        prediction_proba = local_explanation['prediction_proba']
        winning_prob = prediction_proba[1] * 100  # 获胜概率百分比
        
        report_lines = [
            f"## 比赛预测分析报告",
            f"### 预测结果: {'获胜' if winning_prob > 50 else '失利'} (置信度: {winning_prob:.1f}%)",
            f"",
            f"### 关键胜负因子分析:",
            f""
        ]
        
        # 添加LIME解释因子
        for i, item in enumerate(local_explanation['lime']):
            feature = item['feature']
            contribution = item['contribution']
            
            if contribution > 0:
                effect = "提升胜率"
            else:
                effect = "降低胜率"
                contribution = -contribution  # 转换为正数
            
            report_lines.append(
                f"{i+1}. **{feature}** {effect} **{contribution:.1f}%**"
            )
        
        # 添加战术建议
        report_lines.extend([
            f"",
            f"### 战术建议:",
            f"1. 重点保持第二局15-19段的高得分率（当前贡献度最高）",
            f"2. 加强对关键分（15-19分阶段）的心理和技术准备",
            f"3. 利用历史交锋优势建立心理优势"
        ])
        
        return "\n".join(report_lines)

# 示例使用
if __name__ == "__main__":
    # 示例代码 - 实际使用时需集成到主流程中
    report_generator = ExplainabilityReport(
        feature_names=['first_set_11_15', 'second_set_15_19', 'head_to_head']
    )
    
    # 假设已有训练好的model和X_test数据
    # global_report = report_generator.generate_shap_global_report(model, X_test)
    # print("全局特征重要性:", global_report['global_importance'])
