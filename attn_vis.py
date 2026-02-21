import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from glob import glob
import csv
import re


def load_and_label_image(path, label, idx, size=(512, 512)):
    try:
        img = Image.open(path).convert("RGB").resize(size)
    except:
        img = Image.new("RGB", size, (0, 0, 0))
        label = "MISSING"

    draw = ImageDraw.Draw(img)

    # 상단 영역을 크게 확보 (번호 + 라벨)
    header_height = 60
    draw.rectangle([(0, 0), (img.width, header_height)], fill=(0, 0, 0))

    # 글꼴 크기 설정 (폰트는 시스템에 따라 다를 수 있음)
    try:
        big_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)  # 큰 번호용
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)     # 폴더명/라벨용
    except:
        big_font = None
        small_font = None

    # 번호 크게 위에
    draw.text((10, 5), str(idx), fill=(255, 255, 0), font=big_font)   # 노란색 크게
    # 라벨은 아래쪽에 작게
    draw.text((10, 35), label, fill=(255, 255, 255), font=small_font)

    return img


def parse_lomoe_number(fname: str):
    """fname가 'LOMOE_<num>_xxx.png' 형태일 때 <num>을 int로 반환."""
    m = re.match(r"^LOMOE_(\d+)_", fname)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


def match_and_concatenate_images(
    base_dir,
    size=(512, 512),
    input_root="/mnt/hdd1/chaewon/image_editing/dataset/LoMOE-Bench/images",
    save_mapping_csv=True,
):
    """
    - 왼쪽 맨 앞에 'INPUT' 열을 추가.
    - 각 열 상단 라벨은 '[idx] name' 형식.
    - 결과는 base_dir/combined_results 폴더에 저장.
    """
    subfolders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

    # 모든 파일명 집합 만들기
    all_images = {}
    for sub in subfolders:
        sub_path = os.path.join(base_dir, sub)
        images = sorted(glob(os.path.join(sub_path, "*.png")))
        for img_path in images:
            fname = os.path.basename(img_path)
            if fname not in all_images:
                all_images[fname] = {}
            all_images[fname][sub] = img_path

    # 저장 경로: base_dir/combined_results
    save_dir = os.path.join(base_dir, "combined_results")
    os.makedirs(save_dir, exist_ok=True)

    # 열 인덱스 매핑: 0은 INPUT, 1..N은 subfolders
    col_labels = ["INPUT"] + subfolders

    if save_mapping_csv:
        map_csv = os.path.join(save_dir, "column_mapping.csv")
        with open(map_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["col_index", "name_or_folder"])
            for idx, name in enumerate(col_labels):
                w.writerow([idx, name])

    for fname, paths in all_images.items():
        imgs = []

        # 0) INPUT 이미지
        lomoe_num = parse_lomoe_number(fname)
        if lomoe_num is not None:
            init_path = os.path.join(input_root, str(lomoe_num), "init_image.png")
        else:
            init_path = None
        img_input = load_and_label_image(init_path, "INPUT", 0, size)
        imgs.append(img_input)

        # 1..N) 각 서브폴더 이미지들
        for idx, sub in enumerate(subfolders, start=1):
            img_path = paths.get(sub, None)
            img = load_and_label_image(img_path, sub, idx, size)
            imgs.append(img)

        # 가로 이어붙이기
        total_width = size[0] * len(imgs)
        merged_img = Image.new("RGB", (total_width, size[1]))
        for idx, img in enumerate(imgs):
            merged_img.paste(img, (idx * size[0], 0))

        merged_img.save(os.path.join(save_dir, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match and concatenate images from subfolders")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing subfolders with images")
    parser.add_argument("--input_root", type=str, default="/mnt/hdd1/chaewon/image_editing/dataset/LoMOE-Bench/images", help="Root directory for input images")
    parser.add_argument("--size", type=int, default=512, help="Image size (width and height)")
    args = parser.parse_args()

    match_and_concatenate_images(
        base_dir=args.base_dir,
        size=(args.size, args.size),
        input_root=args.input_root,
    )
    print("Done!!")
