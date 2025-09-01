# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 09:11:16 2025
数据仿真模块
用于生成符合论文案例研究的仿真数据，以演示系统功能。
@author: dragon
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple#, Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchSimulator:
    """比赛数据仿真器"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def simulate_hayashi_case(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        仿真2021年世锦赛决赛案例（山口茜 vs 戴资颖）
        返回: (特征DataFrame, 标签Series)
        """
        # 根据论文3.4节描述的案例构建特征向量
        # 注意：这是一个仿真数据，用于功能演示
        simulation_data = {
            'first_set_11_15': [5],    # 仿真值
            'first_set_15_19': [3],    # 仿真值
            'second_set_11_15': [4],   # 仿真值
            'second_set_15_19': [2],  # 关键特征：山口关键分净胜2分
            'final_set_11_15': [0],    # 非三局比赛
            'final_set_15_19': [0],    # 非三局比赛
            'head_to_head': [-1],      # 历史交锋略处下风
            'venue': [0],              # 中立场地
            'ranking_seed': [-5],      # 仿真排名差距
            'recent_match_won': [2]    # 近期状态良好
        }
        
        # 创建特征DataFrame
        features_df = pd.DataFrame(simulation_data)
        
        # 创建标签（已知山口茜获胜）
        label = pd.Series([1], name='win_loss')  # 1表示获胜
        
        logger.info("已生成山口茜案例仿真数据")
        return features_df, label

    def simulate_common_patterns(self, n_samples: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """
        生成常见比赛模式的仿真数据
        用于系统演示和测试
        
        参数:
            n_samples: 总样本数量
            
        返回:
            Tuple[pd.DataFrame, pd.Series]: 特征数据和标签
        """
        logger.info(f"开始生成 {n_samples} 个常见比赛模式的仿真数据")
        
        # 定义四种常见比赛模式及其比例
        patterns = [
            self._generate_dominant_win,    # 强势碾压局: 30%
            self._generate_close_win,       # 险胜局: 25%
            self._generate_comeback_win,    # 逆转局: 20%
            self._generate_upset_win        # 爆冷局: 25%
        ]
        pattern_ratios = [0.3, 0.25, 0.2, 0.25]
        
        # 计算每种模式的样本数量
        pattern_counts = [int(n_samples * ratio) for ratio in pattern_ratios]
        # 调整总数以确保精确匹配
        pattern_counts[-1] = n_samples - sum(pattern_counts[:-1])
        
        # 生成各种模式的数据
        all_features = []
        all_labels = []
        
        for pattern_func, count in zip(patterns, pattern_counts):
            if count > 0:
                features, labels = pattern_func(count)
                all_features.append(features)
                all_labels.append(labels)
        
        # 合并所有数据
        features_df = pd.concat(all_features, ignore_index=True)
        labels_series = pd.concat(all_labels, ignore_index=True)
        
        # 打乱数据顺序
        indices = np.random.permutation(len(features_df))
        features_df = features_df.iloc[indices].reset_index(drop=True)
        labels_series = labels_series.iloc[indices].reset_index(drop=True)
        
        logger.info(f"已完成 {len(features_df)} 个样本的仿真数据生成")
        return features_df, labels_series
    
    def _generate_dominant_win(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """生成强势碾压局的仿真数据"""
        features = {
            'first_set_11_15': np.random.randint(3, 8, n_samples),
            'first_set_15_19': np.random.randint(2, 6, n_samples),
            'second_set_11_15': np.random.randint(3, 8, n_samples),
            'second_set_15_19': np.random.randint(2, 6, n_samples),
            'final_set_11_15': np.zeros(n_samples),  # 两局结束比赛
            'final_set_15_19': np.zeros(n_samples),  # 两局结束比赛
            'head_to_head': np.random.randint(1, 4, n_samples),  # 历史占优
            'venue': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),  # 大概率主场
            'ranking_seed': np.random.randint(5, 20, n_samples),  # 排名领先
            'recent_match_won': np.random.randint(2, 5, n_samples)  # 近期状态好
        }
        
        labels = np.ones(n_samples)  # 全部获胜
        
        return pd.DataFrame(features), pd.Series(labels, name='win_loss')
    
    def _generate_close_win(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """生成险胜局的仿真数据"""
        features = {
            'first_set_11_15': np.random.randint(-2, 3, n_samples),
            'first_set_15_19': np.random.randint(-2, 3, n_samples),
            'second_set_11_15': np.random.randint(-2, 3, n_samples),
            'second_set_15_19': np.random.randint(1, 4, n_samples),  # 关键分略占优
            'final_set_11_15': np.random.randint(-2, 3, n_samples),
            'final_set_15_19': np.random.randint(0, 3, n_samples),  # 决胜局关键分略优
            'head_to_head': np.random.randint(-2, 2, n_samples),  # 历史交锋接近
            'venue': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),  # 中立或主场
            'ranking_seed': np.random.randint(-5, 5, n_samples),  # 排名接近
            'recent_match_won': np.random.randint(-1, 2, n_samples)  # 近期状态一般
        }
        
        labels = np.ones(n_samples)  # 全部获胜
        
        return pd.DataFrame(features), pd.Series(labels, name='win_loss')
    
    def _generate_comeback_win(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """生成逆转局的仿真数据"""
        features = {
            'first_set_11_15': np.random.randint(-5, -1, n_samples),  # 首局落后
            'first_set_15_19': np.random.randint(-4, 0, n_samples),  # 首局关键分落后
            'second_set_11_15': np.random.randint(0, 4, n_samples),  # 次局开始追分
            'second_set_15_19': np.random.randint(1, 5, n_samples),  # 次局关键分占优
            'final_set_11_15': np.random.randint(1, 5, n_samples),  # 决胜局领先
            'final_set_15_19': np.random.randint(2, 6, n_samples),  # 决胜局关键分占优
            'head_to_head': np.random.randint(-3, 1, n_samples),  # 历史交锋略处下风
            'venue': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # 大概率客场
            'ranking_seed': np.random.randint(-15, -5, n_samples),  # 排名落后
            'recent_match_won': np.random.randint(-2, 1, n_samples)  # 近期状态一般或较差
        }
        
        labels = np.ones(n_samples)  # 全部获胜（逆转）
        
        return pd.DataFrame(features), pd.Series(labels, name='win_loss')
    
    def _generate_upset_win(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """生成爆冷局的仿真数据"""
        features = {
            'first_set_11_15': np.random.randint(-1, 3, n_samples),
            'first_set_15_19': np.random.randint(-1, 3, n_samples),
            'second_set_11_15': np.random.randint(-1, 3, n_samples),
            'second_set_15_19': np.random.randint(1, 4, n_samples),  # 关键分略占优
            'final_set_11_15': np.random.randint(-1, 3, n_samples),
            'final_set_15_19': np.random.randint(1, 4, n_samples),  # 决胜局关键分略优
            'head_to_head': np.random.randint(-4, -1, n_samples),  # 历史交锋明显劣势
            'venue': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  # 大概率客场
            'ranking_seed': np.random.randint(-25, -10, n_samples),  # 排名明显落后
            'recent_match_won': np.random.randint(-3, 0, n_samples)  # 近期状态差
        }
        
        labels = np.ones(n_samples)  # 全部获胜（爆冷）
        
        return pd.DataFrame(features), pd.Series(labels, name='win_loss')

    def save_simulation_data(self, features: pd.DataFrame, labels: pd.Series, filepath: str):
        """保存仿真数据到CSV文件"""
        # 合并特征和标签
        data = features.copy()
        data['win_loss'] = labels
        
        # 保存到CSV
        data.to_csv(filepath, index=False)
        logger.info(f"仿真数据已保存至: {filepath}")
        
        return data

# 使用示例
if __name__ == "__main__":
    simulator = MatchSimulator()
    
    # 生成山口茜案例数据
    demo_features, demo_label = simulator.simulate_hayashi_case()
    simulator.save_simulation_data(demo_features, demo_label, "../data/simulation/hayashi_case_2021.csv")
    
    # 生成常见模式数据
    common_features, common_labels = simulator.simulate_common_patterns(200)
    simulator.save_simulation_data(common_features, common_labels, "../data/simulation/common_patterns.csv")
    
    print("山口茜案例数据:")
    print(demo_features)
    print("\n标签:")
    print(demo_label)
    
    print("\n常见模式数据统计:")
    print(f"样本数量: {len(common_features)}")
    print(f"获胜比例: {common_labels.mean():.2%}")