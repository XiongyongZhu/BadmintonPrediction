# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 23:39:33 2025

@author: dragon
"""

from sklearn.model_selection import train_test_split
from config.paths import PathConfig
from preprocessing.data_loader import BadmintonDataLoader
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.data_balancer import DataBalancer
from preprocessing.simulation import MatchSimulator
from evaluation.explainability import ExplainabilityAnalyzer
from evaluation.tratical_report_generator import TacticalReportGenerator
from training.model_manager import ModelManager
from evaluation.model_comparator import ModelComparator
from evaluation.report_generator import ComprehensiveReportGenerator
from evaluation.metrics import ModelEvaluator
from evaluation.visualization import Visualization
from evaluation.explainability import ExplainabilityAnalyzer
from utils.monitoring import MemoryMonitor
import numpy as np
import logging
import psutil
import time
import traceback

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.txt", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 设置第三方库日志级别
    for lib in ['shap', 'matplotlib', 'sklearn', 'xgboost', 'lightgbm', 'fonttools']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def main():
    """主程序入口"""
    try:
        # 初始化配置
        path_config = PathConfig()
        path_config.ensure_directories()
        
        # 初始化组件
        data_loader = BadmintonDataLoader()
        feature_engineer = FeatureEngineer()
        data_balancer = DataBalancer(sampler_type='smotetomek', sampling_strategy=0.5, random_state=42)
        evaluator = ModelEvaluator()
        model_manager = ModelManager()
        model_comparator = ModelComparator(evaluator)
        model_comparator = ModelComparator(evaluator)
        report_generator = ComprehensiveReportGenerator(evaluator)
        
        # 启动内存监控
        monitor = MemoryMonitor()
        monitor.start()
        
        logger = setup_logging()
        logger.info("开始羽毛球比赛预测系统")
        
        # 1. 数据加载与预处理
        logger.info("开始数据加载与预处理...")
        raw_data = data_loader.load_data()
        processed_data = feature_engineer.transform(raw_data)
        
        # 2. 数据准备
        feature_columns = [col for col in [
            'first_set_11_15', 'first_set_15_19', 'second_set_11_15', 
            'second_set_15_19', 'final_set_11_15', 'final_set_15_19',
            'head_to_head', 'venue', 'ranking_seed', 'recent_match_won'
        ] if col in processed_data.columns]
        
        X = processed_data[feature_columns]
        y = raw_data['win_loss']

        # 3. 划分数据集（80%训练，20%测试）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("原始数据分布:", np.bincount(y_train))
        
        # 4. 只对训练集进行平衡处理
        logger.info("对训练集进行平衡处理...")
        X_res, y_res = data_balancer.balance(X_train, y_train)
        print("平衡后数据分布:", np.bincount(y_res))
        X_train = X_res
        y_train = y_res
        imbalance_ratio = data_balancer.calculate_imbalance_ratio(y_train)
        
        # 5. 训练所有模型
        logger.info("开始训练所有模型...")
        trained_models = model_manager.train_all_models(X_train, y_train, imbalance_ratio, feature_columns)
        
        # 6. 评估所有模型
        logger.info("开始评估所有模型...")
        metrics_results = model_comparator.evaluate_all_models(trained_models, X_test, y_test)
        
        # === 可解释性分析部分 ===
        logger.info("开始可解释性分析...")
        explainer = ExplainabilityAnalyzer(feature_names=feature_columns)
        sample_idx = 0
        sample_instance = X_test.iloc[sample_idx].values
        
        # 为每个模型生成解释报告
        for model_name, model in trained_models.items():
            try:
                logger.info(f"为模型 {model_name} 生成解释报告...")
                # 全局解释 - 限制样本数量为50以提高性能
                explainer.generate_global_report(model, X_test.iloc[:50])
                # 局部解释
                local_explanation = explainer.generate_local_report(model, sample_instance, X_test)
            except Exception as e:
                logger.error(f"为模型 {model_name} 生成解释报告时出错: {e}")
                continue
        
        # 7. 生成综合报告
        logger.info("生成综合报告...")
        report_generator.generate_comprehensive_report(
            trained_models, 
            metrics_results, 
            X_test, 
            y_test,
            feature_names=feature_columns  # 添加特征名称参数
        )
        
        # 8. 可视化结果
        logger.info("生成可视化...")
        visualizer = Visualization()
        visualizer.plot_metrics_comparison(metrics_results)
        visualizer.plot_feature_importance(trained_models, feature_columns)
        
        logger.info("羽毛球比赛预测系统完成！")
        
        # 9. 系统功能演示
        best_model = trained_models['XGBoost']
        demo_system_capability(best_model, logger, feature_engineer, X_test, y_test)
        
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        traceback.print_exc()
    finally:
        monitor.stop()

def demo_system_capability(model, logger, feature_engineer, X_test, y_test):
    """
    演示系统预测能力，生成论文图3和图4的可视化图表
    """
    logger.info("开始系统能力演示...")
    
    try:
        # 创建MatchSimulator实例并获取仿真数据
        simulator = MatchSimulator()
        demo_features, demo_label = simulator.simulate_hayashi_case()

        # 标记特征工程对象不需要标签数据
        feature_engineer.requires_label = False
        processed_demo_features = feature_engineer.transform(demo_features)
        
        # 开始监控
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss
        # 模型预测
        demo_pred = model.predict_proba(processed_demo_features)[0, 1]
        # 结束监控
        end_time = time.time()
        mem_after = process.memory_info().rss
        prediction_time = end_time - start_time
        memory_usage = (mem_after - mem_before) / (1024**3)
        
        # 返回结果和性能指标
        logger.info(f"单次预测完成 - "
                    f"耗时: {prediction_time:.4f}秒, "
                    f"内存占用: {memory_usage:.6f}GB")
        
        # 输出预测结果
        logger.info(f"山口茜获胜概率预测: {demo_pred:.2%}")
        logger.info(f"实际结果: {'山口茜获胜' if demo_label.iloc[0] == 1 else '戴资颖获胜'}")
        
        # 生成SHAP解释（对应论文图3和图4）
        import shap
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # 开始监控
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss
        # 根据模型类型选择合适的解释器
        if hasattr(model, 'tree_') or 'XGB' in type(model).__name__:
            # 树模型使用TreeExplainer
            explainer = shap.TreeExplainer(model)
        else:
            # 非树模型使用KernelExplainer
            explainer = shap.KernelExplainer(model.predict_proba, X_test.iloc[:100])
        
        # 计算SHAP值
        shap_values = explainer.shap_values(processed_demo_features)
        # 处理不同类型的SHAP值输出
        if isinstance(shap_values, list):
            # 多分类情况：取第一个类别的值
            shap_values_positive = shap_values
            expected_value = explainer.expected_value
        elif len(shap_values.shape) == 3:
            # 三维数组 (n_classes, n_samples, n_features)
            shap_values_positive = shap_values[0]  # 取第一个类别
            expected_value = explainer.expected_value
        else:
            # 二维数组 (n_samples, n_features)
            shap_values_positive = shap_values
            expected_value = explainer.expected_value
        # 结束监控
        end_time = time.time()
        mem_after = process.memory_info().rss
        prediction_time = end_time - start_time
        memory_usage = (mem_after - mem_before) / (1024**3)
        
        # 返回结果和性能指标
        logger.info(f"SHAP解释完成 - "
                    f"耗时: {prediction_time:.4f}秒, "
                    f"内存占用: {memory_usage:.6f}GB")
        
        # 生成图3(a): 全局特征重要性（条形图）
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, processed_demo_features, plot_type="bar", show=False)
        # plt.title("全局特征重要性分析 (图3a)", fontproperties="SimHei")  # 使用中文字体
        plt.tight_layout()
        plt.savefig("results/global_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("已生成图3(a): 全局特征重要性分析")
        
        # 生成图3(b): SHAP蜂群图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, processed_demo_features, show=False)
        # plt.title("特征作用方向分析 (图3b)", fontproperties="SimHei")  # 使用中文字体
        plt.tight_layout()
        plt.savefig("results/feature_direction_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("已生成图3(b): 特征作用方向分析")
        
        # 生成图4: SHAP力导向图（局部解释）
        plt.figure(figsize=(10, 6))
        # 获取第一个样本的解释
        sample_index = 0
        sample_features = processed_demo_features.iloc[sample_index]
        
        # 处理不同的SHAP值格式
        if isinstance(shap_values_positive, np.ndarray) and len(shap_values_positive.shape) == 2:
            # 二维数组：取第一个样本的解释
            sample_shap = shap_values_positive[sample_index]
        else:
            # 其他格式：直接使用
            sample_shap = shap_values_positive
        
        # 绘制力导向图
        shap.force_plot(expected_value, sample_shap, sample_features,
                       matplotlib=True, show=False)
        
        # plt.title("山口茜vs戴资颖预测归因分析 (图4)", fontproperties="SimHei")
        plt.tight_layout()
        plt.savefig("results/local_explanation_force_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("已生成图4: SHAP力导向图")
        
        # 生成战术报告
        from evaluation.explainability import ExplainabilityAnalyzer
        explainer_obj = ExplainabilityAnalyzer(feature_names=feature_engineer.selected_features)
        # 获取第一个样本的特征值（一维数组）
        sample_instance = processed_demo_features.iloc[0].values
        # 生成局部解释报告
        local_report = explainer_obj.generate_local_report(model, sample_instance, processed_demo_features)
        
        if local_report:
            logger.info("战术分析报告:\n" + local_report)
        else:
            logger.warning("无法生成战术分析报告")
        
    except Exception as e:
        logger.error(f"演示系统失败: {e}")
        raise


if __name__ == "__main__":
    main()
