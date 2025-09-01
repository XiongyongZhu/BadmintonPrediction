# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 10:39:58 2025
战术报告生成模块
专门用于将模型预测和SHAP分析结果转化为可执行的战术建议
@author: dragon
"""

import numpy as np
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger(__name__)

class TacticalReportGenerator:
    """专门生成战术建议的报告生成器"""
    
    def __init__(self, feature_names: List[str], class_names: List[str] = ['输', '赢']):
        """
        初始化战术报告生成器
        
        参数:
            feature_names: 特征名称列表
            class_names: 类别名称列表
        """
        self.feature_names = feature_names
        self.class_names = class_names
        
        # 特征名称映射到中文描述
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
        
        # 战术建议规则库
        self.tactical_rules = {
            'second_set_15_19': {
                'positive': [
                    "继续保持第二局15-19阶段的得分能力，这是您的最大优势",
                    "在第二局中段(15-19分)保持进攻压力，这是决定比赛走向的关键时期",
                    "第二局中段的稳定发挥是您获胜的重要保障，继续保持"
                ],
                'negative': [
                    "需要改进第二局15-19阶段的策略，减少不必要的失误",
                    "第二局中段(15-19分)是您的薄弱环节，需要加强此阶段的专注力",
                    "考虑调整第二局中段的战术布置，提高此阶段的得分效率"
                ]
            },
            'head_to_head': {
                'positive': [
                    "利用历史交锋优势增强信心，您对这位对手有心理优势",
                    "过往交手记录显示您更适应这位对手的打法，继续保持",
                    "历史战绩表明您对这位对手有克制作用，发挥这一优势"
                ],
                'negative': [
                    "克服历史交锋劣势，专注于本次比赛而非过往结果",
                    "虽然历史交锋处于下风，但每场比赛都是新的开始",
                    "不要被过往交手记录影响，专注于当前比赛的战术执行"
                ]
            },
            'first_set_11_15': {
                'positive': [
                    "保持第一局11-15阶段的领先优势，这是建立信心的关键",
                    "第一局中前段的表现稳定，为您后续比赛奠定了良好基础",
                    "第一局11-15分的良好发挥展示了您的竞技状态，继续保持"
                ],
                'negative': [
                    "提高第一局11-15阶段的状态稳定性，避免过早落后",
                    "第一局中前段需要更加专注，避免不必要的失误",
                    "加强第一局11-15分的开局表现，为整场比赛建立优势"
                ]
            },
            'momentum_indicator': {
                'positive': [
                    "利用当前的比赛势头保持优势，不要给对手喘息机会",
                    "比赛势头在您这边，乘胜追击扩大优势",
                    "保持良好的比赛节奏，利用势头控制比赛进程"
                ],
                'negative': [
                    "调整策略打断对手的进攻势头，重新掌握比赛节奏",
                    "需要改变战术以扭转不利的比赛势头",
                    "对手势头正盛，考虑通过暂停或战术调整打断其节奏"
                ]
            },
            'recent_match_won': {
                'positive': [
                    "近期良好的获胜记录表明您状态正佳，保持信心",
                    "连胜势头是您的优势，带着自信进入比赛",
                    "近期表现稳定，这是您技术水平和心理素质的体现"
                ],
                'negative': [
                    "虽然近期战绩不佳，但每场比赛都是新的机会",
                    "不要受近期结果影响，调整心态专注当前比赛",
                    "通过技术分析和训练弥补近期的不足，重拾信心"
                ]
            }
        }
        
        # 通用战术建议
        self.general_advice = [
            "注意保持体能分配，特别是在比赛后半段",
            "根据对手特点调整战术，扬长避短",
            "关键分处理要果断，减少犹豫",
            "保持心理稳定性，不受个别得失分影响",
            "充分利用场地条件，适应环境因素"
        ]
    
    def generate_tactical_report(self, 
                                model_prediction: Dict[str, Any],
                                shap_values: np.array,
                                feature_values: Dict[str, float],
                                match_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成完整的战术报告
        
        参数:
            model_prediction: 模型预测结果，包含概率和类别
            shap_values: SHAP值数组，形状为(n_features,)
            feature_values: 特征实际值字典
            match_context: 比赛上下文信息（可选）
            
        返回:
            包含详细战术报告的字典
        """
        # 确定预测结果
        winning_prob = model_prediction.get('winning_probability', 0.5)
        predicted_class = 1 if winning_prob > 0.5 else 0
        
        # 分析特征贡献
        feature_contributions = self._analyze_feature_contributions(shap_values, feature_values)
        
        # 生成针对性建议
        specific_advice = self._generate_specific_advice(feature_contributions)
        
        # 生成综合报告
        report = {
            'prediction_summary': {
                'predicted_outcome': self.class_names[predicted_class],
                'winning_probability': winning_prob,
                'confidence_level': self._get_confidence_level(winning_prob)
            },
            'key_factors': self._identify_key_factors(feature_contributions, top_n=3),
            'specific_advice': specific_advice,
            'general_advice': np.random.choice(self.general_advice, 2, replace=False).tolist(),
            'matchup_analysis': self._generate_matchup_analysis(feature_values) if match_context else None,
            'visualization_data': self._prepare_visualization_data(feature_contributions)
        }
        
        # 生成文本报告
        report['text_report'] = self._generate_text_report(report)
        
        logger.info("战术报告生成完成")
        return report
    
    def _analyze_feature_contributions(self, shap_values: np.array, feature_values: Dict[str, float]) -> List[Dict]:
        """
        分析各特征的贡献度
        
        返回:
            包含特征贡献分析的列表
        """
        contributions = []
        
        for i, feature in enumerate(self.feature_names):
            if i >= len(shap_values):
                continue
                
            shap_val = shap_values[i]
            actual_val = feature_values.get(feature, 0)
            
            # 计算相对贡献度（百分比）
            contribution = {
                'feature': feature,
                'shap_value': shap_val,
                'actual_value': actual_val,
                'contribution_percent': abs(shap_val) / (np.sum(np.abs(shap_values)) + 1e-10) * 100,
                'direction': 'positive' if shap_val > 0 else 'negative',
                'description': self.feature_name_map.get(feature, feature)
            }
            
            contributions.append(contribution)
        
        # 按贡献度排序
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return contributions
    
    def _generate_specific_advice(self, feature_contributions: List[Dict]) -> List[str]:
        """
        生成针对性的战术建议
        
        返回:
            战术建议列表
        """
        advice_list = []
        
        for contrib in feature_contributions[:5]:  # 只考虑前5个最重要特征
            feature = contrib['feature']
            direction = contrib['direction']
            
            if feature in self.tactical_rules:
                # 从规则库中选择一条随机建议
                advice_options = self.tactical_rules[feature][direction]
                selected_advice = np.random.choice(advice_options)
                
                # 添加贡献度信息
                advice_with_context = f"{selected_advice} (贡献度: {contrib['contribution_percent']:.1f}%)"
                advice_list.append(advice_with_context)
        
        return advice_list
    
    def _identify_key_factors(self, feature_contributions: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        识别关键影响因素
        
        返回:
            关键因素列表
        """
        return feature_contributions[:top_n]
    
    def _generate_matchup_analysis(self, feature_values: Dict[str, float]) -> Dict[str, Any]:
        """
        生成对阵分析
        
        返回:
            对阵分析字典
        """
        # 这里可以根据实际特征值生成更详细的对阵分析
        analysis = {
            'historical_advantage': "略有优势" if feature_values.get('head_to_head', 0) > 0 else "稍有劣势",
            'current_form': "状态良好" if feature_values.get('recent_match_won', 0) > 0 else "状态一般",
            'key_stage': "第二局中段" if abs(feature_values.get('second_set_15_19', 0)) > 1 else "第一局开局"
        }
        
        return analysis
    
    def _get_confidence_level(self, winning_prob: float) -> str:
        """
        根据获胜概率确定置信水平
        
        返回:
            置信水平描述
        """
        if winning_prob > 0.7:
            return "高置信度"
        elif winning_prob > 0.6:
            return "中等置信度"
        elif winning_prob > 0.55:
            return "低置信度"
        else:
            return "非常不确定"
    
    def _prepare_visualization_data(self, feature_contributions: List[Dict]) -> Dict[str, Any]:
        """
        准备可视化数据
        
        返回:
            可视化数据字典
        """
        # 提取前10个最重要特征
        top_features = [contrib['description'] for contrib in feature_contributions[:10]]
        top_contributions = [contrib['contribution_percent'] for contrib in feature_contributions[:10]]
        directions = [1 if contrib['direction'] == 'positive' else -1 for contrib in feature_contributions[:10]]
        
        return {
            'top_features': top_features,
            'contributions': top_contributions,
            'directions': directions
        }
    
    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """
        生成文本格式的战术报告
        
        返回:
            文本报告字符串
        """
        lines = []
        
        # 标题
        lines.append("羽毛球比赛战术分析报告")
        lines.append("=" * 50)
        lines.append("")
        
        # 预测摘要
        lines.append("预测摘要:")
        lines.append(f"- 预测结果: {report_data['prediction_summary']['predicted_outcome']}")
        lines.append(f"- 获胜概率: {report_data['prediction_summary']['winning_probability']:.1%}")
        lines.append(f"- 置信水平: {report_data['prediction_summary']['confidence_level']}")
        lines.append("")
        
        # 关键因素
        lines.append("关键影响因素:")
        for i, factor in enumerate(report_data['key_factors']):
            lines.append(f"{i+1}. {factor['description']}: {factor['contribution_percent']:.1f}% "
                       f"({'正向' if factor['direction'] == 'positive' else '负向'})")
        lines.append("")
        
        # 具体建议
        lines.append("针对性战术建议:")
        for i, advice in enumerate(report_data['specific_advice']):
            lines.append(f"{i+1}. {advice}")
        lines.append("")
        
        # 一般建议
        lines.append("一般性建议:")
        for i, advice in enumerate(report_data['general_advice']):
            lines.append(f"{i+1}. {advice}")
        lines.append("")
        
        # 对阵分析（如果有）
        if report_data['matchup_analysis']:
            lines.append("对阵分析:")
            lines.append(f"- 历史交锋: {report_data['matchup_analysis']['historical_advantage']}")
            lines.append(f"- 近期状态: {report_data['matchup_analysis']['current_form']}")
            lines.append(f"- 关键阶段: {report_data['matchup_analysis']['key_stage']}")
            lines.append("")
        
        # 报告生成时间
        lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)
    
    def generate_visualization(self, visualization_data: Dict[str, Any], save_path: str = None):
        """
        生成战术建议可视化图表
        
        参数:
            visualization_data: 可视化数据
            save_path: 保存路径（可选）
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = visualization_data['top_features']
        contributions = visualization_data['contributions']
        directions = visualization_data['directions']
        
        # 创建颜色映射（绿色表示正向，红色表示负向）
        colors = ['green' if d > 0 else 'red' for d in directions]
        
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, contributions, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # 从上到下显示
        ax.set_xlabel('contributions (%)')                 # 贡献度
        ax.set_title('Feature Contribution Analysis')      # 特征贡献度分析
        
        # 添加数值标签
        for i, v in enumerate(contributions):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图表已保存: {save_path}")
        
        return fig

# 使用示例
if __name__ == "__main__":
    # 初始化战术报告生成器
    feature_names = ['second_set_15_19', 'head_to_head', 'first_set_11_15', 'recent_match_won']
    report_generator = TacticalReportGenerator(feature_names)
    
    # 模拟数据
    model_prediction = {
        'winning_probability': 0.72,
        'predicted_class': 1
    }
    
    shap_values = np.array([0.142, 0.095, 0.078, 0.055])  # SHAP值
    feature_values = {
        'second_set_15_19': 3.2,
        'head_to_head': 2.5,
        'first_set_11_15': 1.8,
        'recent_match_won': 4
    }
    
    # 生成战术报告
    tactical_report = report_generator.generate_tactical_report(
        model_prediction, shap_values, feature_values
    )
    
    # 打印文本报告
    print(tactical_report['text_report'])
    
    # 生成可视化
    fig = report_generator.generate_visualization(
        tactical_report['visualization_data'],
        save_path="tactical_analysis.png"
    )
    
    plt.show()