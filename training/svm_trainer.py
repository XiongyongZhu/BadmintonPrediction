# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 09:11:26 2025
SVM模型训练
@author: dragon
"""

from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
import logging
import shap
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

class SVMTrainer:
    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.cv_results = None
        self.feature_importances_ = None
        self.feature_names = None
    
    def train(self, X_train, y_train, f1_scorer, feature_names=None):
        """优化后的SVM训练方法，支持特征重要性"""
        try:
            # 保存特征名称用于后续分析
            self.feature_names = feature_names
            
            # 定义参数网格
            param_grid = {
                'C': [1.846],       # 论文中的最佳惩罚系数
                'gamma': [0.034],   # 论文中的最佳gamma值
                'kernel': ['rbf']   # 径向基核函数
            }
            
            base_model = SVC(
                C=1.846, gamma=0.034, kernel='rbf',
                probability=True, class_weight='balanced'
            )
            
            # 交叉验证策略
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # 使用HalvingGridSearchCV
            grid_search = HalvingGridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=f1_scorer,
                n_jobs=-1,
                random_state=42
            )
            
            # 训练模型
            grid_search.fit(X_train, y_train)
            
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_results = grid_search.cv_results_
            
            # 计算SVM的特征重要性
            self._calculate_feature_importance(X_train, y_train)
            
            logger.info(f"SVM训练完成，最佳参数: {self.best_params}")
            return self.best_model
            
        except Exception as e:
            logger.error(f"SVM训练失败: {str(e)}")
            raise
    
    def _calculate_feature_importance(self, X_train, y_train):
        """修复维度不匹配问题的特征重要性计算"""
        try:
            n_features = X_train.shape
            
            # 方法1: 对于线性核，使用系数绝对值
            if hasattr(self.best_model, 'coef_') and self.best_model.coef_ is not None:
                if self.best_model.coef_.ndim == 2:  # 多分类情况
                    # 取所有类别的平均重要性
                    coef_importance = np.mean(np.abs(self.best_model.coef_), axis=0)
                else:  # 二分类情况
                    coef_importance = np.abs(self.best_model.coef_[0])
                
                # 确保维度匹配
                if len(coef_importance) == n_features:
                    self.feature_importances_ = coef_importance
                    logger.info("使用SVM系数计算特征重要性")
                    return
            
            # 方法2: 排列重要性（更可靠的方法）
            try:
                # 使用较小的样本集计算排列重要性以避免计算开销
                sample_size = min(100, len(X_train))
                if sample_size > 0:
                    perm_importance = permutation_importance(
                        self.best_model, 
                        X_train[:sample_size], 
                        y_train[:sample_size],
                        n_repeats=5,
                        random_state=42,
                        n_jobs=-1
                    )
                    self.feature_importances_ = perm_importance.importances_mean
                    logger.info("使用排列重要性计算SVM特征重要性")
                    return
            except Exception as perm_error:
                logger.warning(f"排列重要性计算失败: {perm_error}")
            
            # 方法3: SHAP值（备选方案）
            try:
                if hasattr(self.best_model, 'predict_proba'):
                    # 使用小样本计算SHAP值
                    sample_size = min(50, len(X_train))
                    explainer = shap.KernelExplainer(
                        self.best_model.predict_proba, 
                        shap.sample(X_train, min(20, len(X_train)))
                    )
                    shap_values = explainer.shap_values(X_train[:sample_size])
                    
                    if isinstance(shap_values, list):
                        # 多分类：取第一个类别的平均绝对值
                        shap_importance = np.mean(np.abs(shap_values[0]), axis=0)
                    else:
                        shap_importance = np.mean(np.abs(shap_values), axis=0)
                    
                    self.feature_importances_ = shap_importance
                    logger.info("使用SHAP值计算SVM特征重要性")
                    return
            except Exception as shap_error:
                logger.warning(f"SHAP值计算失败: {shap_error}")
            
            # 方法4: 默认均匀重要性（最后备选）
            self.feature_importances_ = np.ones(n_features) / n_features
            logger.info("使用均匀分布作为SVM特征重要性")
            
        except Exception as e:
            logger.warning(f"SVM特征重要性计算失败: {str(e)}")
            # 确保返回一个合理的重要性数组
            n_features = X_train.shape[1] if hasattr(X_train, 'shape') else 1
            self.feature_importances_ = np.ones(n_features) / n_features
    
    def get_feature_importances(self):
        """获取特征重要性，确保返回正确格式"""
        if self.feature_importances_ is None:
            # 返回默认的重要性数组
            if self.feature_names:
                n_features = len(self.feature_names)
            else:
                n_features = 1
            return np.ones(n_features) / n_features
        return self.feature_importances_
    
    def get_feature_importance_df(self):
        """返回带特征名称的重要性DataFrame"""
        importances = self.get_feature_importances()
        
        if self.feature_names and len(self.feature_names) == len(importances):
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def get_best_params(self):
        """返回最佳参数"""
        return self.best_params
    
    def get_cv_results(self):
        """返回交叉验证结果"""
        return self.cv_results
