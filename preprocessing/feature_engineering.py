# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:04:43 2025
# 特征工程
@author: dragon
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, k_features=10, requires_label=False):
        self.k_features = k_features
        self.pipeline = None
        self.selected_features = None
        self.feature_means_ = None
        self.feature_stds_ = None
        self.requires_label = requires_label
        
    def get_numeric_features(self, data):
        """获取数值型特征列"""
        numeric_features = [
            'first_set_11_15', 'first_set_15_19',
            'second_set_11_15', 'second_set_15_19',
            'final_set_11_15', 'final_set_15_19',
            'head_to_head', 'venue', 'ranking_seed',
            'recent_match_won'
        ]
        return [feat for feat in numeric_features if feat in data.columns]
    
    def create_features(self, data):
        """创建衍生特征"""
        df = data.copy()
        numeric_features = self.get_numeric_features(df)
        
        # 创建交互特征
        if all(col in df.columns for col in ['first_set_15_19', 'second_set_15_19', 'final_set_15_19']):
            df['critical_point_advantage'] = (
                df['first_set_15_19'] + df['second_set_15_19'] + df['final_set_15_19']
            )
            numeric_features.append('critical_point_advantage')
            
        if all(col in df.columns for col in ['first_set_11_15', 'second_set_11_15', 'final_set_11_15']):
            df['total_score_advantage'] = (
                df['first_set_11_15'] + df['second_set_11_15'] + df['final_set_11_15']
            )
            numeric_features.append('total_score_advantage')
            
        return df, numeric_features
    
    def build_pipeline(self):
        """构建特征工程管道"""
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
            ('pca', PCA(n_components=0.95))
        ]
    
        # 如果不需要标签数据，使用无监督特征选择
        if not hasattr(self, 'requires_label') or not self.requires_label:
            steps[2] = ('selector', SelectKBest(score_func=f_classif, k=10))
    
        return Pipeline(steps)
        # return Pipeline([
        #     ('imputer', SimpleImputer(strategy='median')),
        #     ('scaler', StandardScaler()),
        #     ('selector', SelectKBest(score_func=f_classif, k=self.k_features))
        # ])
    
    def fit(self, X, y=None):
        # 首先标准化特征
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 使用ANOVA F-value选择最重要的特征
        self.selector = SelectKBest(score_func=f_classif, k=self.k_features)
        self.selector.fit(X_scaled, y)
        return self
    
    def transform(self, data):
        """执行特征工程"""
        try:
            # 移除会导致数据泄露的特征
            if 'second_set_15_19' in data.columns:
                data = data.drop(['second_set_15_19', 'final_set_15_19', 
                                  'first_set_15_19', 'final_set_11_15'], axis=1)
        
            # 创建早期比赛阶段特征（避免数据泄露）
            df, numeric_features = self.create_early_stage_features(data)
        
            # 构建并应用管道（只使用transform，不使用fit_transform）
            # 检查管道是否已经拟合
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                self.pipeline = self.build_pipeline()
                # 如果没有标签数据，使用无监督方法
                if 'win_loss' in df.columns:
                    X_transformed = self.pipeline.fit_transform(df[numeric_features], df['win_loss'])
                else:
                    X_transformed = self.pipeline.fit_transform(df[numeric_features])
            else:
                # 管道已经拟合，只进行转换
                X_transformed = self.pipeline.transform(df[numeric_features])
        
            # # 构建并拟合管道
            # self.pipeline = self.build_pipeline()
            # X_transformed = self.pipeline.fit_transform(df[numeric_features], df['win_loss'])
        
            # 获取选择的特征名称
            selector = self.pipeline.named_steps['selector']
            self.selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
        
            logger.info(f"特征工程完成，选择特征: {self.selected_features}")
            return pd.DataFrame(X_transformed, columns=self.selected_features)
        
        except Exception as e:
            logger.error(f"特征工程失败: {str(e)}")
            raise

    def create_early_stage_features(self, data):
        """
        创建早期比赛阶段特征，避免使用后期数据
        """
        df = data.copy()
    
        # 只使用比赛早期数据（前10分）
        early_features = []
        for period in ['first_set', 'second_set']:
            for minute_range in ['0_5', '5_10']:
                feature_name = f"{period}_{minute_range}"
                if feature_name in df.columns:
                    early_features.append(feature_name)
        
        # 添加静态特征
        static_features = ['head_to_head', 'venue', 'ranking_seed', 'recent_match_won']
        static_features = [f for f in static_features if f in df.columns]
        
        numeric_features = early_features + static_features
        
        return df, numeric_features



