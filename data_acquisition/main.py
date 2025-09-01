# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 14:32:41 2025

@author: dragon
"""

from data_acquisition.badminton_data_acquirer import BadmintonDataAcquirer

def main():
    # 创建数据获取器实例
    acquirer = BadmintonDataAcquirer()
    
    # 获取数据
    raw_data, processed_data = acquirer.acquire_data(years=[2023, 2024])
    
    # 保存数据
    acquirer.save_data(
        raw_data, 
        processed_data,
        raw_path='data/raw/badminton_dataset.csv',
        processed_path='data/processed/badminton_features.csv'
    )
    
    print("数据获取和保存完成！")
    print(f"原始数据形状: {raw_data.shape}")
    print(f"特征数据形状: {processed_data.shape}")

if __name__ == "__main__":
    main()
