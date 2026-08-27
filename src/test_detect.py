"""
test_detect.py — 第 1 周第 4 天用
作用：用官方自带的、已经训练好的免费模型，跑一张测试图，看看「目标检测」长什么样。
运行：在 (marine) 环境下执行  python test_detect.py
"""
from ultralytics import YOLO

# yolo11n.pt 是官方现成模型，第一次运行会自动下载（约 5MB）
model = YOLO("yolo11n.pt")

# 用 Ultralytics 官方的一张示例图（公交车/行人），也会自动下载
result = model("https://ultralytics.com/images/bus.jpg")

# 把检测结果保存成图片，打开它就能看到框
result[0].save("my_first_detection.jpg")
print("完成！打开文件夹里的 my_first_detection.jpg 看看效果。")
