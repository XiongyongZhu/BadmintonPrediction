# -*- coding: utf-8 -*-
"""路径配置
@author: dragon
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PathConfig:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
    RAW_DATA = DATA_DIR / "raw/badminton_dataset.csv"
    DATA_VERSION = DATA_DIR / "raw/version.txt"
    PROCESSED_DATA = DATA_DIR / "processed/processed_data.csv"
    
    # 添加缺失的路径属性
    REPORTS_DIR = BASE_DIR / "reports"
    RESULTS_DIR = BASE_DIR / "results"  # 添加RESULTS属性
    TS = BASE_DIR / "temp"  # 如果存在的话
    
    # 字体路径（如果存在）
    FONTS_DIR = BASE_DIR / "fonts"
    SIMHEI_FONT = FONTS_DIR / "simhei.ttf" if FONTS_DIR.exists() else None
    
    # 其他必要的目录
    def __init__(self):
        # 确保所有目录都存在
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保所有必要的目录都存在"""
        directories = [
            self.DATA_DIR / "raw",
            self.DATA_DIR / "processed",
            self.MODELS_DIR,
            self.TRAINED_MODELS_DIR,
            self.REPORTS_DIR,
            self.RESULTS_DIR,  # 确保results目录存在
            self.FONTS_DIR if hasattr(self, 'FONTS_DIR') else None
        ]
        
        for directory in directories:
            if directory and not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建目录: {directory}")
