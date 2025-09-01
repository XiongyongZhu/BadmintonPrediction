# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 17:54:40 2025
可解释性模块
@author: dragon
"""

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import shap
import gc
import logging

logger = logging.getLogger(__name__)

class ExplainabilityAnalyzer:
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.global_explanation = None  # 初始化属性
    
    def pre_check(self, model, instance):
        """在执行SHAP计算前进行预检查"""
        # 检查输入实例的特征维度
        # 确保实例是二维数组：如果是一维，则重塑为二维
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
            
        if instance.shape[1] != len(self.feature_names):
            logger.error(f"输入实例特征维度不匹配: {instance.shape[1]} != {len(self.feature_names)}")
            return False
        
        # 检查模型类型和兼容性      
        model_type = type(model).__name__
        logger.info(f"模型类型: {model_type}")
        
        # 检查模型是否支持SHAP解释
        if not self._is_model_supported(model):
            logger.warning(f"模型 {model_type} 可能不支持SHAP解释")
            return False
        
        return True
    
    def _is_model_supported(self, model):
        """检查模型是否支持SHAP解释"""
        model_type = type(model).__name__
        supported_models = ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier',
                           'SVC', 'LogisticRegression', 'GaussianNB']
        return model_type in supported_models
    
    def generate_global_report(self, model, X_data, max_samples=50):
        """生成全局特征解释报告，限制样本数量以加速计算"""
        try:
            # 随机采样，减少计算量
            if len(X_data) > max_samples:
                sample_indices = np.random.choice(len(X_data), max_samples, replace=False)
                if isinstance(X_data, pd.DataFrame):
                    X_sampled = X_data.iloc[sample_indices]
                else:
                    X_sampled = X_data[sample_indices]
            else:
                X_sampled = X_data
        
            # 创建SHAP解释器
            if isinstance(model, (XGBClassifier, RandomForestClassifier, LGBMClassifier)):
                explainer = shap.TreeExplainer(model)
                # 获取SHAP值，明确处理二分类
                shap_values_multi = explainer.shap_values(X_sampled, check_additivity=False)
            
                # 处理树解释器的常见输出格式
                if isinstance(shap_values_multi, list):
                    # 通常是二分类或多分类，列表长度为 n_classes
                    if len(shap_values_multi) == 2:
                        # 二分类问题，取正类（索引1）
                        shap_values = shap_values_multi
                    else:
                        # 多分类问题，这里简化处理，取第一个类别的SHAP值
                        shap_values = shap_values_multi
                        logger.warning("多分类检测到，默认使用第一个类别的SHAP值生成全局报告。")
                elif isinstance(shap_values_multi, np.ndarray) and len(shap_values_multi.shape) == 3:
                    # 处理三维数组 (n_classes, n_samples, n_features)
                    logger.info(f"处理三维SHAP值数组: {shap_values_multi.shape}")
                    
                    # 对于二分类问题，取第二个类别（索引1）的SHAP值
                    if shap_values_multi.shape[0] == 2:  # 第一个维度是类别数
                        shap_values = shap_values_multi[1]  # 形状为 (n_samples, n_features)
                    elif shap_values_multi.shape[2] == 2:  # 最后一个维度是类别数
                        shap_values = shap_values_multi[:, :, 1]  # 形状为 (n_samples, n_features)
                    else:
                        # 默认取第一个类别
                        shap_values = shap_values_multi
                        logger.warning("无法确定类别维度，默认使用第一个类别的SHAP值")
                else:
                    # 如果返回的是二维数组，直接使用（可能是回归或特定设置）
                    shap_values = shap_values_multi
            else:
                # 对于非树模型，使用KernelExplainer
                background_data = shap.sample(X_sampled, min(30, len(X_sampled)))
                explainer = shap.KernelExplainer(model.predict, background_data)
                shap_values = explainer.shap_values(X_sampled, nsamples=30)
                
                # 处理KernelExplainer返回的多分类情况
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1]  # 取最后一个元素
                elif len(shap_values.shape) == 3:
                    shap_values = shap_values[0]  # 取第一个类别
        
            # 检查SHAP值是否为二维数组
            if len(shap_values.shape) != 2:
                logger.error(f"SHAP值维度异常: {shap_values.shape}，期望二维数组")
                return None
        
            # 检查特征维度是否匹配
            if shap_values.shape[1] != len(self.feature_names):
                logger.error(f"SHAP值特征维度不匹配: {shap_values.shape[1]} != {len(self.feature_names)}")
                return None
        
            # 计算特征重要性（绝对值平均）
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            global_importance = pd.Series(mean_abs_shap, index=self.feature_names)
            total_importance = global_importance.sum()
            global_impr_percent = global_importance / total_importance * 100
        
            return {
                'global_importance': global_impr_percent,
                'explainer': explainer,
                'shap_values': shap_values
            }
        except Exception as e:
            logger.error(f"生成全局报告失败: {e}")
            return None

    def generate_local_report(self, model, instance, background_data):
        """生成单个样本的局部解释报告"""
        # 先进行预检查
        if not self.pre_check(model, instance):
            return None
        
        try:
            # 确保实例是二维数组
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)
                
            # 减少背景数据样本数以提高性能
            background_data_sampled = shap.sample(background_data, min(100, len(background_data)))
            logger.info("减少背景数据样本数以提高性能")
        
            # 根据模型类型选择合适的SHAP解释器
            model_type = type(model).__name__
        
            if 'RandomForest' in model_type or 'XGB' in model_type or 'LGBM' in model_type:
                explainer = shap.TreeExplainer(model)
                shap_values_multi = explainer.shap_values(instance, check_additivity=False)
                
                # 处理三维数组情况 (样本数, 特征数, 类别数)
                if isinstance(shap_values_multi, np.ndarray) and len(shap_values_multi.shape) == 3:
                    logger.info(f"处理三维SHAP值数组: {shap_values_multi.shape}")
                
                    # 对于二分类问题，通常取第二个类别（索引1）的SHAP值
                    if shap_values_multi.shape[2] == 2:  # 二分类
                        shap_values = shap_values_multi[0, :, 1]  # 取第一个样本，所有特征，第二个类别
                    else:
                        # 多分类情况，取第一个类别
                        shap_values = shap_values_multi[0, :, 0]
                
                    logger.info(f"提取后的SHAP值形状: {shap_values.shape}")
                else:
                    # 处理其他格式（列表或二维数组）
                    if isinstance(shap_values_multi, list):
                        if len(shap_values_multi) == 2:
                            shap_values_single_sample = shap_values_multi[1]  # 形状应为 (1, n_features)
                        elif len(shap_values_multi) == 1:
                            shap_values_single_sample = shap_values_multi
                        else:
                            shap_values_single_sample = shap_values_multi
                    elif isinstance(shap_values_multi, np.ndarray) and len(shap_values_multi.shape) == 2:
                        shap_values_single_sample = shap_values_multi
                    else:
                        shap_values_single_sample = shap_values_multi
                
                    # 确保是二维数组 (1, n_features)
                    if len(shap_values_single_sample.shape) == 2:
                        # 然后提取一维数组
                        shap_values = shap_values_single_sample[0]  # 取第一个样本的SHAP值
                    else:
                        # 如果已经是一维，则直接使用
                        shap_values = shap_values_single_sample
                
                    logger.info(f"TreeExplainer processed local SHAP values shape: {shap_values.shape}")
            else:
                # 对于非树模型，使用KernelExplainer
                explainer = shap.KernelExplainer(model.predict, background_data_sampled)
                shap_values = explainer.shap_values(instance, nsamples=30)
                logger.info(f"KernelExplainer shap_values initial shape: {np.array(shap_values).shape}")
            
                # 处理多分类情况
                if isinstance(shap_values, list):
                    if len(shap_values) == 2:
                        shap_values = shap_values[1]  # 取正类
                    else:
                        shap_values = shap_values
                elif len(shap_values.shape) == 3:
                    if shap_values.shape[2] == 2:
                        shap_values = shap_values[:, :, 1]
                    else:
                        shap_values = shap_values
            logger.info("确保shap_values是一维数组（一个样本的SHAP值）")
            # 确保shap_values是一维数组（一个样本的SHAP值）
            if len(shap_values.shape) > 1:
                shap_values = shap_values.flatten()
            logger.info(f"Final shap_values shape: {shap_values.shape}")
            
            # 检查SHAP值维度与特征数量是否匹配
            if len(shap_values) != len(self.feature_names):
                logger.error(f"SHAP值维度与特征名数量不匹配: {len(shap_values)} != {len(self.feature_names)}")
                return None
        
            # 生成解释报告
            report_lines = []
            for i, feature in enumerate(self.feature_names):
                # 确保sh_val是标量而不是数组
                sh_val = float(shap_values[i]) if hasattr(shap_values[i], 'item') else shap_values[i]
                
                if sh_val > 0:
                    report_lines.append(f"特征 '{feature}' 对胜负结果有正向影响，贡献值: {sh_val:.4f}")
                else:
                    report_lines.append(f"特征 '{feature}' 对胜负结果有负向影响，贡献值: {sh_val:.4f}")
        
            return "\n".join(report_lines)
        except Exception as e:
            logger.error(f"生成局部报告失败: {e}", exc_info=True)
            return None
    
    def generate_suggestion(self, feature, effect_type):
        """根据特征和影响类型生成战术建议"""
        suggestions_map = {
            'head_to_head': {
                1: "利用历史交锋优势增强信心",
                -1: "克服历史交锋劣势，专注于本次比赛"
            },
            'venue': {
                1: "利用主场优势创造有利条件",
                -1: "克服客场劣势，调整适应场地条件"
            },
            'ranking_seed': {
                1: "利用排名优势保持稳定发挥",
                -1: "克服排名劣势，积极冲击对手"
            },
            'recent_match_won': {
                1: "保持近期良好状态",
                -1: "调整状态，提升近期比赛表现"
            },
            'second_set_15_19': {
                1: "继续保持第二局15-19阶段的得分能力",
                -1: "改进第二局15-19阶段的策略，减少失误"
            },
            'first_set_11_15': {
                1: "保持第一局11-15阶段的领先优势",
                -1: "提高第一局11-15阶段的状态稳定性"
            },
            'momentum_indicator': {
                1: "利用当前的比赛势头保持优势",
                -1: "调整策略打断对手的进攻势头"
            }
        }
    
        return suggestions_map.get(feature, {}).get(effect_type, 
               "关注此因素的表现以提高胜率" if effect_type > 0 else 
               "改善此因素的表现以提升胜率")

    
    def cleanup(self):
        """清理内存"""
        gc.collect()


