# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 09:09:51 2025
模型训练管理器
@author: dragon
"""


import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, make_scorer
from training.svm_trainer import SVMTrainer
from training.xgb_trainer import XGBoostTrainer
import lightgbm as lgb
import psutil
import time

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.metrics_results = {}
        self.performance_stats = {}  # 新增性能统计字典
    
    def train_svm(self, X_train, y_train, f1_scorer, feature_names):
        """训练SVM模型"""
        try:
            logger.info("开始训练 SVM 模型...")
            svm_trainer = SVMTrainer()
            svm_model = svm_trainer.train(X_train, y_train, f1_scorer, feature_names)
            logger.info("SVM 模型训练完成")
            return svm_model
        except Exception as e:
            logger.error(f"SVM 模型训练失败: {e}")
            return None
    
    def train_xgboost(self, X_train, y_train, f1_scorer, imbalance_ratio, feature_names):
        """训练XGBoost模型"""
        try:
            logger.info("开始训练 XGBoost 模型...")
            xgb_trainer = XGBoostTrainer()
            xgb_model = xgb_trainer.train(X_train, y_train, f1_scorer, imbalance_ratio)
            logger.info("XGBoost 模型训练完成")
            return xgb_model
        except Exception as e:
            logger.error(f"XGBoost 模型训练失败: {e}")
            return None
    
    def train_random_forest(self, X_train, y_train):
        """训练随机森林模型"""
        try:
            logger.info("开始训练随机森林模型...")
            rf_model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            logger.info("随机森林模型训练完成")
            return rf_model
        except Exception as e:
            logger.error(f"随机森林模型训练失败: {e}")
            return None
    
    def train_logistic_regression(self, X_train, y_train):
        """训练逻辑回归模型"""
        try:
            logger.info("开始训练逻辑回归模型...")
            lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
            lr_model.fit(X_train, y_train)
            logger.info("逻辑回归模型训练完成")
            return lr_model
        except Exception as e:
            logger.error(f"逻辑回归模型训练失败: {e}")
            return None
    
    def train_naive_bayes(self, X_train, y_train):
        """训练朴素贝叶斯模型"""
        try:
            logger.info("开始训练高斯朴素贝叶斯模型...")
            nb_model = GaussianNB()
            nb_model.fit(X_train, y_train)
            logger.info("高斯朴素贝叶斯模型训练完成")
            return nb_model
        except Exception as e:
            logger.error(f"高斯朴素贝叶斯模型训练失败: {e}")
            return None
    
    def train_lightgbm(self, X_train, y_train, imbalance_ratio):
        """训练LightGBM模型"""
        try:
            logger.info("开始训练 LightGBM 模型...")
            lgb_model = lgb.LGBMClassifier(
                objective='binary', random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=imbalance_ratio
            )
            lgb_model.fit(X_train, y_train)
            logger.info("LightGBM 模型训练完成")
            return lgb_model
        except Exception as e:
            logger.warning(f"LightGBM 模型训练失败或未配置: {e}")
            return None
    
    def train_all_models(self, X_train, y_train, imbalance_ratio, feature_names):
        """训练所有支持的模型"""
        model_instances = {}
        f1_scorer = make_scorer(f1_score, average='macro')
        # 开始监控
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss
        
        # 训练各个模型
        model_instances['SVM'] = self.train_svm(X_train, y_train, f1_scorer, feature_names)
        model_instances['XGBoost'] = self.train_xgboost(X_train, y_train, f1_scorer, imbalance_ratio, feature_names)
        model_instances['RandomForest'] = self.train_random_forest(X_train, y_train)
        model_instances['LogisticRegression'] = self.train_logistic_regression(X_train, y_train)
        model_instances['GaussianNB'] = self.train_naive_bayes(X_train, y_train)
        model_instances['LightGBM'] = self.train_lightgbm(X_train, y_train, imbalance_ratio)
        
        # 结束监控
        end_time = time.time()
        mem_after = process.memory_info().rss
            
        # 记录性能指标
        model_type = 'All'
        self.performance_stats[model_type] = {
            'training_time': end_time - start_time,
            'memory_usage': (mem_after - mem_before) / (1024**3)  # 转换为GB
        }
            
        logger.info(f"{model_type}训练完成 - "
                    f"耗时: {self.performance_stats[model_type]['training_time']:.2f}秒, "
                    f"内存占用: {self.performance_stats[model_type]['memory_usage']:.2f}GB")
        
        # 过滤掉训练失败的模型
        return {k: v for k, v in model_instances.items() if v is not None}
