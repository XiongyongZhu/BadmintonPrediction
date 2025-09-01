# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 09:11:26 2025

@author: dragon
"""

import xgboost as xgb
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
import logging
# import numpy as np

logger = logging.getLogger(__name__)

class XGBoostTrainer:
    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.cv_results = None
        self.feature_importances_ = None
    
    def train(self, X_train, y_train, f1_scorer, imbalance_ratio):
        """使用网格搜索训练XGBoost模型并找到最佳参数"""
        try:
            # 创建基础模型
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=imbalance_ratio,  # 使用辅助函数计算权重
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            # 参数网格
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.2, 1.5],
                'scale_pos_weight': [imbalance_ratio]  # 添加类别权重
            }
            
            # 交叉验证策略
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # 使用HalvingGridSearchCV
            grid_search = HalvingGridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=f1_scorer,  # 使用F1分数作为评估标准
                n_jobs=-1,
                random_state=42
            )
            
            # 执行网格搜索
            grid_search.fit(X_train, y_train)
            
            # 保存结果和最佳模型
            self.best_params = grid_search.best_params_
            self.best_model = grid_search.best_estimator_
            self.cv_results = {
                'best_score': grid_search.best_score_,
                'all_results': grid_search.cv_results_
            }
            
            # 确保特征重要性可用
            self._ensure_feature_importances()
            
            logger.info(f"XGBoost训练完成，最佳参数: {self.best_params}")
            return self.best_model
            
        except Exception as e:
            logger.error(f"XGBoost训练失败: {str(e)}")
            raise
    
    def _ensure_feature_importances(self):
        """确保特征重要性可用"""
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importances_ = self.best_model.feature_importances_
            else:
                self.feature_importances_ = None
        except Exception as e:
            logger.warning(f"XGBoost特征重要性确保失败: {str(e)}")
            self.feature_importances_ = None
    
    def get_best_params(self):
        """返回最佳参数"""
        return self.best_params
    
    def get_best_model(self):
        """返回最佳模型"""
        return self.best_model
    
    def get_feature_importances(self):
        """获取特征重要性"""
        return self.feature_importances_
    
    def get_cv_results(self):
        """返回交叉验证结果"""
        return self.cv_results
