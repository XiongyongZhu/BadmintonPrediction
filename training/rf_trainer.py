# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:29:49 2025
随机森林训练
@author: dragon
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class RandomForestTrainer:
    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.cv_results = None
    
    def train(self, X_train, y_train):
        """训练随机森林模型"""
        # 使用网格搜索优化参数
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        # 交叉验证策略
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='f1')
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        self.cv_results = {
            'best_score': grid_search.best_score_,
            'all_results': grid_search.cv_results_
        }
        return self.best_model
        
    def get_best_params(self):
        """返回最佳参数"""
        return self.best_params
    
    def get_best_model(self):
        """返回最佳模型"""
        return self.best_model
    
    def get_cv_results(self):
        """返回交叉验证结果"""
        return self.cv_results
