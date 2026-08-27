"""
predict_local.py — 第 2 周第 11 天用
作用：用你自己训练好的模型（best.pt），检测你电脑上的一张图片，并把结果存下来。
运行：在 (marine) 环境下，在本文件所在目录执行  python predict_local.py
      或指定图片：python predict_local.py --img D:/Trae/marine-yolo-project/test/images/某张图.jpg
"""
import sys
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "best.pt"
TEST_IMG_DIR = PROJECT_DIR / "test" / "images"
SAMPLES_DIR = PROJECT_DIR / "samples"
OUTPUT_NAME = SCRIPT_DIR / "own_result.jpg"


def find_default_image():
    """优先用 test/images/，没有就用 samples/（clone 后的场景）"""
    search_dirs = [TEST_IMG_DIR, SAMPLES_DIR]
    for d in search_dirs:
        if d.exists():
            for ext in (".jpg", ".jpeg", ".png"):
                files = list(d.glob(f"*{ext}"))
                if files:
                    return str(files[0])
    print("找不到测试图片。请用 --img 指定一张图片路径，或把图片放到 samples/ 文件夹。")
    sys.exit(1)


def main():
    if not MODEL_PATH.exists():
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        print("请把 best.pt 放到 models/ 文件夹里。")
        sys.exit(1)

    args = sys.argv[1:]
    img_path = None
    if "--img" in args:
        i = args.index("--img")
        img_path = args[i + 1]
    if not img_path:
        img_path = find_default_image()

    print(f"使用的图片：{img_path}")
    print(f"使用的模型：{MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    CLEAN_NAMES = {0: "Trash", 1: "Other", 2: "Garbage", 3: "Marine Life"}
    model.model.names = CLEAN_NAMES

    result = model(img_path, device="cpu")[0]

    result.save(str(OUTPUT_NAME))
    count = len(result.boxes)
    print(f"完成！检测到 {count} 个目标，结果保存在：{OUTPUT_NAME}")


if __name__ == "__main__":
    main()
