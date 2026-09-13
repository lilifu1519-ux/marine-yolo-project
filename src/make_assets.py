"""make_assets.py — 生成 README 里展示的检测效果图
运行：conda activate marine 后，
      cd D:\Trae\marine-yolo-project\src
      python make_assets.py
会在 assets/ 下生成 detect_marine_life.jpg 和 detect_garbage.jpg。
（这个脚本用不到 README，只在想重新生成效果图时跑。）
"""
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "best.pt"
SAMPLES_DIR = PROJECT_DIR / "samples"
OUT_DIR = PROJECT_DIR / "assets"
OUT_DIR.mkdir(exist_ok=True)

model = YOLO(str(MODEL_PATH))
model.model.names = {0: "Trash", 1: "Other", 2: "Garbage", 3: "Marine Life"}

marine_img = list(SAMPLES_DIR.glob("nm_*"))[0]
r1 = model.predict(str(marine_img), conf=0.3, device="cpu")[0]
r1.save(str(OUT_DIR / "detect_marine_life.jpg"))
print("已生成 assets/detect_marine_life.jpg")

garbage_img = list(SAMPLES_DIR.glob("multiplegarbage*"))[0]
r2 = model.predict(str(garbage_img), conf=0.3, device="cpu")[0]
r2.save(str(OUT_DIR / "detect_garbage.jpg"))
print("已生成 assets/detect_garbage.jpg")