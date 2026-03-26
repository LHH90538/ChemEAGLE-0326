# 国内环境请先在终端输入 export HF_ENDPOINT=https://hf-mirror.com  解决Huggingface连接问题

import os
import zipfile
from huggingface_hub import hf_hub_download

# 模型仓库
MODEL_REPO = "CYF200127/ChemEAGLEModel"

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def download_single_zip(filename: str, local_path: str):
    ensure_parent_dir(local_path)
    print(f"正在下载: {filename}")

    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename,
        local_dir=os.path.dirname(local_path),
        local_dir_use_symlinks=False,
        force_download=True
    )
    print(f"✅ 下载完成: {local_path}\n")

def extract_zip(zip_path, extract_dir):
    """自动解压zip到对应文件夹"""
    if not os.path.exists(zip_path):
        print(f"⚠️ 不存在 {zip_path}，跳过解压")
        return

    print(f"正在解压: {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ 解压完成\n")

def download_all_missing_zips():
    targets = [
        ("biobert-large-cased.zip", "./biobert-large-cased.zip"),
        ("cre_models_v0.1.zip", "./cre_models_v0.1.zip"),
    ]

    print("==== 开始下载 ChemEAGLE 缺失的 ZIP 文件 ====\n")
    print("ℹ️ Tesseract OCR 不再从 HF 下载，请在 Linux 上通过系统包安装：")
    print("   sudo apt update && sudo apt install -y tesseract-ocr\n")
    for fname, lpath in targets:
        download_single_zip(fname, lpath)

    # ====================== 自动解压 ======================
    print("\n==== 开始自动解压所有 ZIP ====\n")
    extract_zip("biobert-large-cased.zip", "biobert-large-cased")
    extract_zip("cre_models_v0.1.zip", "cre_models_v0.1")

    print("✅ 所有 ZIP 下载并解压完成！")
    print("📁 模型目录已就绪：")
    print("  - cre_models_v0.1/")
    print("  - biobert-large-cased/")

if __name__ == "__main__":
    download_all_missing_zips()