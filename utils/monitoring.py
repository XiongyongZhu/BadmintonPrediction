# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 17:31:37 2025

@author: dragon
"""

import psutil
import time
import logging
from threading import Thread

logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, interval=5.0, warning_threshold=0.6, critical_threshold=0.8):
        self.interval = interval
        self.warning_threshold = warning_threshold  # 60%内存使用警告
        self.critical_threshold = critical_threshold  # 80%内存使用临界值
        self.running = False
        
    def get_memory_usage(self):
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # MB
        
    def get_system_memory_usage(self):
        """获取系统内存使用百分比"""
        memory = psutil.virtual_memory()
        return memory.percent / 100.0  # 返回0-1之间的比例
        
    def monitor(self):
        """监控内存使用"""
        while self.running:
            # 获取进程内存使用量
            process_memory_usage = self.get_memory_usage()
            
            # 获取系统内存使用百分比
            system_memory_percent = self.get_system_memory_usage()
            
            # 记录进程内存使用量（调试信息）
            logger.debug(f"当前进程内存使用: {process_memory_usage:.2f}MB")
            
            # 基于系统内存百分比进行检查
            if system_memory_percent > self.critical_threshold:
                logger.critical(f"系统内存使用超过临界值: {system_memory_percent*100:.1f}% ({process_memory_usage:.2f}MB 进程使用)")
            elif system_memory_percent > self.warning_threshold:
                logger.warning(f"系统内存使用超过警告值: {system_memory_percent*100:.1f}% ({process_memory_usage:.2f}MB 进程使用)")
            else:
                logger.debug(f"系统内存使用正常: {system_memory_percent*100:.1f}%")
            
            time.sleep(self.interval)
    
    def start(self):
        """启动监控"""
        self.running = True
        self.thread = Thread(target=self.monitor)
        self.thread.daemon = True
        self.thread.start()
        logger.info("内存监控已启动")
    
    def stop(self):
        """停止监控"""
        self.running = False
        logger.info("内存监控已停止")
