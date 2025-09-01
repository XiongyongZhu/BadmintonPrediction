# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:16:08 2025

@author: dragon
"""

# data_acquisition/badminton_data_acquirer.py
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
import random
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Tuple

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BadmintonDataAcquirer:
    """羽毛球比赛数据获取器"""
    
    def __init__(self):
        # 羽毛球数据API端点
        self.apis = {
            'bwf_tournaments': ' https://bwf.tournamentsoftware.com/sport/tournaments.aspx ',
            'bwf_results': ' https://bwf.tournamentsoftware.com/sport/matches.aspx ',
            'badminton_api': ' https://api.badminton-api.com/v1/matches ',  # 假设的API
            'sports_data': ' https://api.sportsdata.io/v3/badminton/scores/json '
        }
        
        # 预定义的重要赛事
        self.important_tournaments = [
            'All England Open Badminton Championships',
            'BWF World Championships',
            'Indonesia Open',
            'China Open',
            'Japan Open',
            'Malaysia Open',
            'Denmark Open',
            'French Open',
            'India Open',
            'Singapore Open'
        ]
        
        # 球员名单（用于模拟数据）
        self.male_players = [
            'Viktor Axelsen', 'Kento Momota', 'Anders Antonsen', 'Anthony Ginting',
            'Lee Zii Jia', 'Chou Tien Chen', 'Jonatan Christie', 'Lakshya Sen',
            'Kunlavut Vitidsarn', 'Srikanth Kidambi'
        ]
        
        self.female_players = [
            'Akane Yamaguchi', 'Tai Tzu Ying', 'Chen Yufei', 'An Se Young',
            'Pusarla Venkata Sindhu', 'Ratchanok Intanon', 'He Bingjiao',
            'Michelle Li', 'Nozomi Okuhara', 'Pornpawee Chochuwong'
        ]
        
        self.doubles_teams = [
            'Marcus Gideon/Kevin Sukamuljo', 'Mohammad Ahsan/Hendra Setiawan',
            'Hiroyuki Endo/Yuta Watanabe', 'Lee Yang/Wang Chi Lin',
            'Takuro Hoki/Yugo Kobayashi', 'Greysia Polii/Apriyani Rahayu',
            'Chen Qing Chen/Jia Yi Fan', 'Kim So Yeong/Kong Hee Yong',
            'Yuki Fukushima/Sayaka Hirota', 'Mayu Matsumoto/Wakana Nagahara'
        ]

    def fetch_from_bwf_website(self, year: int) -> pd.DataFrame:
        """从BWF网站获取比赛数据"""
        matches = []
        
        try:
            # 构建URL
            url = f"{self.apis['bwf_tournaments']}?year={year}"
            logger.info(f"尝试从BWF网站获取{year}年数据: {url}")
            
            # 发送请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                logger.info(f"成功获取{year}年BWF页面")
                
                # 解析HTML内容
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找赛事链接
                tournament_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if 'id=' in href and ('tournament' in href or 'event' in href):
                        tournament_id = re.search(r'id=([0-9A-F-]+)', href)
                        if tournament_id:
                            tournament_links.append({
                                'id': tournament_id.group(1),
                                'name': link.get_text().strip(),
                                'url': f" https://bwf.tournamentsoftware.com {href}"
                            })
                
                logger.info(f"找到{len(tournament_links)}个赛事")
                
                # 获取每个赛事的比赛数据
                for tournament in tournament_links[:5]:  # 限制数量以避免请求过多
                    try:
                        tournament_matches = self._fetch_tournament_matches(tournament)
                        matches.extend(tournament_matches)
                        time.sleep(1)  # 礼貌延迟
                    except Exception as e:
                        logger.error(f"获取赛事{tournament['name']}数据失败: {e}")
                        continue
            else:
                logger.warning(f"BWF网站返回状态码: {response.status_code}")
                
        except Exception as e:
            logger.error(f"从BWF网站获取数据失败: {e}")
        
        return pd.DataFrame(matches) if matches else pd.DataFrame()

    def _fetch_tournament_matches(self, tournament: Dict) -> List[Dict]:
        """获取特定赛事的比赛数据"""
        matches = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(tournament['url'], headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找比赛表格
                match_tables = soup.find_all('table', class_='matches')
                
                for table in match_tables:
                    rows = table.find_all('tr')
                    for row in rows[1:]:  # 跳过表头
                        cols = row.find_all('td')
                        if len(cols) >= 5:
                            try:
                                match_data = {
                                    'tournament_id': tournament['id'],
                                    'tournament_name': tournament['name'],
                                    'date': cols[0].get_text().strip(),
                                    'player_a': cols[2].get_text().strip(),
                                    'player_b': cols[3].get_text().strip(),
                                    'score': cols[4].get_text().strip(),
                                    'category': self._determine_category(cols[1].get_text().strip())
                                }
                                matches.append(match_data)
                            except Exception as e:
                                logger.warning(f"解析比赛行时出错: {e}")
                                continue
        except Exception as e:
            logger.error(f"获取赛事{tournament['name']}详情失败: {e}")
        
        return matches

    def _determine_category(self, text: str) -> str:
        """根据文本确定比赛类别"""
        text = text.lower()
        if 'men' in text and 'single' in text:
            return 'MS'
        elif 'women' in text and 'single' in text:
            return 'WS'
        elif 'men' in text and 'double' in text:
            return 'MD'
        elif 'women' in text and 'double' in text:
            return 'WD'
        elif 'mixed' in text:
            return 'XD'
        else:
            return 'Unknown'

    def fetch_from_sports_api(self, api_key: str = None) -> pd.DataFrame:
        """从体育数据API获取比赛数据"""
        matches = []
        
        try:
            # 如果没有提供API密钥，使用模拟数据
            if not api_key:
                logger.warning("未提供API密钥，使用模拟数据")
                return self.generate_mock_data(100)
            
            # 构建请求URL和头
            url = f"{self.apis['sports_data']}/Matches"
            headers = {'Ocp-Apim-Subscription-Key': api_key}
            
            # 发送请求
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"从体育API获取到{len(data)}条比赛数据")
                
                # 转换数据格式
                for match in data:
                    matches.append({
                        'match_id': match.get('MatchId'),
                        'tournament_name': match.get('Tournament'),
                        'date': match.get('DateTime'),
                        'player_a': match.get('PlayerA'),
                        'player_b': match.get('PlayerB'),
                        'score': match.get('Score'),
                        'category': match.get('Category')
                    })
            else:
                logger.warning(f"体育API返回状态码: {response.status_code}")
                
        except Exception as e:
            logger.error(f"从体育API获取数据失败: {e}")
        
        return pd.DataFrame(matches) if matches else pd.DataFrame()

    def generate_mock_data(self, num_matches: int = 200) -> pd.DataFrame:
        """生成模拟羽毛球比赛数据"""
        matches = []
        
        # 生成赛事日期（过去2年）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        for i in range(num_matches):
            # 随机选择赛事和日期
            tournament = random.choice(self.important_tournaments)
            match_date = start_date + timedelta(days=random.randint(0, 730))
            
            # 随机选择比赛类别
            category = random.choice(['MS', 'WS', 'MD', 'WD', 'XD'])
            
            # 根据类别选择选手
            if category == 'MS':
                player_a = random.choice(self.male_players)
                player_b = random.choice([p for p in self.male_players if p != player_a])
            elif category == 'WS':
                player_a = random.choice(self.female_players)
                player_b = random.choice([p for p in self.female_players if p != player_a])
            else:
                team_a = random.choice(self.doubles_teams)
                team_b = random.choice([t for t in self.doubles_teams if t != team_a])
                player_a, player_b = team_a, team_b
            
            # 生成比分（羽毛球比赛通常是三局两胜）
            score = self._generate_score()
            
            matches.append({
                'match_id': f"MID{1000 + i}",
                'tournament_name': tournament,
                'date': match_date.strftime('%Y-%m-%d'),
                'player_a': player_a,
                'player_b': player_b,
                'score': score,
                'category': category,
                'duration': f"{random.randint(30, 120)}分钟"
            })
        
        return pd.DataFrame(matches)

    def _generate_score(self) -> str:
        """生成羽毛球比赛比分"""
        sets = []
        # 随机决定比赛局数（2或3局）
        num_sets = random.choice([2, 3])
        
        for i in range(num_sets):
            # 羽毛球每局21分制
            score_a = random.randint(15, 21)
            # 确保比分合理（对手得分在15-21之间，且与赢家比分接近）
            score_b = random.randint(15, 21)
            
            # 确保有一方达到21分且领先至少2分
            if abs(score_a - score_b) < 2:
                if score_a > score_b:
                    score_a = 21
                    score_b = min(19, score_b)
                else:
                    score_b = 21
                    score_a = min(19, score_a)
            
            sets.append(f"{score_a}-{score_b}")
        
        return ", ".join(sets)

    def process_raw_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """处理原始数据，提取特征"""
        if raw_df.empty:
            logger.warning("原始数据为空，生成模拟特征数据")
            return self.generate_features_data(200)
        
        # 这里添加实际的数据处理逻辑
        # 由于原始数据格式可能不同，这里只是一个示例
        
        processed_data = []
        for _, row in raw_df.iterrows():
            # 解析比分字符串
            scores = row['score'].split(', ')
            set_results = []
            
            for score in scores:
                try:
                    a, b = score.split('-')
                    set_results.append((int(a), int(b)))
                except:
                    continue
            
            # 提取特征（这里只是示例，实际特征工程会更复杂）
            if len(set_results) >= 2:
                feature_row = {
                    'first_set_11_15': set_results[0] - set_results[1],
                    'first_set_15_19': set_results[0] - set_results[1],
                    'second_set_11_15': set_results[0] - set_results[1] if len(set_results) > 1 else 0,
                    'second_set_15_19': set_results[0] - set_results[1] if len(set_results) > 1 else 0,
                    'final_set_11_15': set_results[0] - set_results[1] if len(set_results) > 2 else 0,
                    'final_set_15_19': set_results[0] - set_results[1] if len(set_results) > 2 else 0,
                    'head_to_head': random.randint(-5, 5),
                    'venue': random.randint(0, 1),
                    'ranking_seed': random.randint(-50, 50),
                    'recent_match_won': random.randint(-3, 3),
                    'win_loss': random.randint(0, 1)
                }
                processed_data.append(feature_row)
        
        return pd.DataFrame(processed_data)

    def generate_features_data(self, num_samples: int = 200) -> pd.DataFrame:
        """生成特征数据（模拟）"""
        data = []
        
        for i in range(num_samples):
            row = {
                'first_set_11_15': random.randint(-10, 10),
                'first_set_15_19': random.randint(-10, 10),
                'second_set_11_15': random.randint(-10, 10),
                'second_set_15_19': random.randint(-10, 10),
                'final_set_11_15': random.randint(-10, 10),
                'final_set_15_19': random.randint(-10, 10),
                'head_to_head': random.randint(-5, 5),
                'venue': random.randint(0, 1),
                'ranking_seed': random.randint(-50, 50),
                'recent_match_won': random.randint(-3, 3),
                'win_loss': random.randint(0, 1)
            }
            data.append(row)
        
        return pd.DataFrame(data)

    def acquire_data(self, years: List[int] = [2023, 2024]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """主数据获取函数"""
        raw_data = pd.DataFrame()
        processed_data = pd.DataFrame()
        
        logger.info("开始获取羽毛球比赛数据...")
        
        # 尝试从BWF网站获取数据
        for year in years:
            try:
                year_data = self.fetch_from_bwf_website(year)
                if not year_data.empty:
                    raw_data = pd.concat([raw_data, year_data], ignore_index=True)
            except Exception as e:
                logger.error(f"获取{year}年数据失败: {e}")
        
        # 如果从网站获取失败，生成模拟数据
        if raw_data.empty:
            logger.warning("无法从网站获取数据，生成模拟数据")
            raw_data = self.generate_mock_data(200)
        
        # 处理数据以提取特征
        processed_data = self.process_raw_data(raw_data)
        
        return raw_data, processed_data

    def save_data(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame, 
                 raw_path: str = 'badminton_raw_data.csv', 
                 processed_path: str = 'badminton_data.csv'):
        """保存数据到CSV文件"""
        try:
            raw_df.to_csv(raw_path, index=False, encoding='utf-8-sig')
            processed_df.to_csv(processed_path, index=False, encoding='utf-8-sig')
            logger.info(f"原始数据已保存到: {raw_path}")
            logger.info(f"处理后的数据已保存到: {processed_path}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")

# 使用示例
if __name__ == "__main__":
    acquirer = BadmintonDataAcquirer()
    
    try:
        # 获取数据
        raw_data, processed_data = acquirer.acquire_data(years=[2023, 2024])
        
        # 显示数据信息
        print(f"获取到 {len(raw_data)} 条原始比赛数据")
        print(f"生成 {len(processed_data)} 条特征数据")
        
        # 保存数据
        acquirer.save_data(raw_data, processed_data)
        
        # 显示数据预览
        print("\n原始数据预览:")
        print(raw_data.head())
        
        print("\n特征数据预览:")
        print(processed_data.head())
        
    except Exception as e:
        print(f"数据获取过程中出现错误: {e}")
        # 生成模拟数据作为备用
        mock_raw = acquirer.generate_mock_data(100)
        mock_processed = acquirer.generate_features_data(100)
        acquirer.save_data(mock_raw, mock_processed)
        print("已生成模拟数据用于测试流程")




