import math
import numpy as np

def calculate_angle(a, b, c):
    """
    計算三個點之間的夾角 (單位：度)
    a: 固定點 (肩膀) [x, y]
    b: 轉軸點 (髖部) [x, y] -> 我們要算這個角的角度
    c: 固定點 (膝蓋) [x, y]
    """
    # 轉成 numpy array 比較方便計算
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    # 使用 arctan2 計算角度 (這是寫程式算角度的標準做法)
    # 算出 線段BC 和 線段BA 的弧度
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    
    # 把弧度轉成角度 (3.14 -> 180度)
    angle = np.abs(radians * 180.0 / np.pi)
    
    # 如果角度大於 180，我們通常看內角 (例如 200度 改看 160度)
    if angle > 180.0:
        angle = 360 - angle
        
    return angle