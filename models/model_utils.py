# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 22:33:36 2025

@author: dragon
"""

from config.paths import PathConfig
import shap
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelUtils:
    def __init__(self):
        self.path_config = PathConfig()
    
    def save_model(self, model, filename):
        """保存模型"""
        filepath = self.path_config.TRAINED_MODELS_DIR / filename
        joblib.dump(model, filepath)
        logger.info(f"模型已保存: {filepath}")
    
    def load_model(self, filename):
        """加载模型"""
        filepath = self.path_config.TRAINED_MODELS_DIR / filename
        model = joblib.load(filepath)
        logger.info(f"模型已加载: {filepath}")
        return model
    
    def get_feature_importance(self, model, feature_names):
        """获取特征重要性"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        else:
            logger.warning("该模型不支持特征重要性分析")
            return {}
    
    def explain_model(model, X_test, feature_names=None):
        """
        使用SHAP解释模型预测，确保维度匹配
    
        参数:
            model: 训练好的模型
            X_test: 测试数据集
            feature_names: 需要解释的具体样本（可选）
        """
        # 确保使用训练时的特征子集
        if hasattr(model, 'feature_names_in_'):
            # 使用模型训练时的特征顺序
            X_test = X_test[:, model.feature_names_in_]
            
        # 创建解释器（根据模型类型选择合适的方法）
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)  # 树模型
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.KernelExplainer(model.predict, X_test)
            shap_values = explainer.shap_values(X_test)
    
        # 处理多维SHAP值（多分类情况）
        if isinstance(shap_values, list):
            # 多分类：取第一个类别的值
            shap_values = shap_values
            
        return shap_values
