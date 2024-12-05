import numpy as np

def normalize_angle(angle):
    """将角度标准化到 [-pi, pi] 范围内。"""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def process_headings(headings):
    """处理方向角序列，消除从 -pi 到 pi 的跳变。"""
    processed_headings = [headings[0]]
    for i in range(1, len(headings)):
        diff = headings[i] - processed_headings[-1]
        if diff > np.pi:
            diff -= 2.0 * np.pi
        elif diff < -np.pi:
            diff += 2.0 * np.pi
        processed_headings.append(processed_headings[-1] + diff)
    return processed_headings

# 示例方向角序列
headings = [-3.125, 3.120]
processed_headings = process_headings(headings)

print("原始方向角:", headings)
print("处理后的方向角:", processed_headings)
