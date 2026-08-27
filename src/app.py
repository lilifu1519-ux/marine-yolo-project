"""
app.py — 第 3 周第 15~16 天用
作用：用 Gradio 做一个网页 Demo，上传图片就能看到检测框和目标计数。
运行：在 (marine) 环境下，在本文件所在目录执行  python app.py
      然后浏览器打开它打印的网址（一般是 http://127.0.0.1:7860）
"""
import gradio as gr
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "best.pt"

model = YOLO(str(MODEL_PATH))

CLEAN_NAMES = {
    0: "Trash",
    1: "Other",
    2: "Garbage",
    3: "Marine Life",
}
model.model.names = CLEAN_NAMES


def detect(img, conf):
    if img is None:
        return None, {"提示": "请先上传一张图片"}
    r = model.predict(img, conf=conf, device="cpu")[0]
    count = len(r.boxes)
    classes = [CLEAN_NAMES[int(c)] for c in r.boxes.cls.tolist()]
    info = {"框出目标数量": count, "类别": classes}
    return r.plot(), info


ui = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="上传一张海洋图片"),
        gr.Slider(0.2, 0.9, 0.45, step=0.05, label="置信度阈值（越高框越少越准）"),
    ],
    outputs=[
        gr.Image(label="检测结果"),
        gr.JSON(label="统计信息"),
    ],
    title="海洋目标检测 Demo",
    description="上传一张图片，自动框出目标并统计数量。基于 YOLO11n。",
)

if __name__ == "__main__":
    ui.launch()
