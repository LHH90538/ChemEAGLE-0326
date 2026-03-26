# 国内环境请先在终端输入 export HF_ENDPOINT=https://hf-mirror.com  解决Huggingface连接问题

import os
from huggingface_hub import hf_hub_download, snapshot_download


MODEL_REPO = "CYF200127/ChemEAGLEModel"
CHEMRXN_REPO = "amberwang/chemrxnextractor-training-modules"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def download_file(repo_id: str, filename: str, local_path: str) -> str:
    ensure_parent_dir(local_path)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.dirname(os.path.abspath(local_path)),
        local_dir_use_symlinks=False,
    )
    # hf_hub_download 在 local_dir 模式下会直接将文件放到 local_dir
    # 这里统一返回目标路径，便于输出和后续检查
    target_path = os.path.abspath(local_path)
    if os.path.exists(target_path):
        return target_path
    return os.path.abspath(downloaded)


def download_weights() -> None:
    targets = [
        # 原有 ckpt 权重
        ("molnextr.pth", "./molnextr.pth"),
        ("rxn.ckpt", "./rxn.ckpt"),
        ("moldet.ckpt", "./moldet.ckpt"),
        ("corefdet.ckpt", "./corefdet.ckpt"),
        ("ner.ckpt", "./ner.ckpt"),
    ]

    print("开始下载 ChemEAGLE 权重...")
    for filename, local_path in targets:
        print(f"- 下载 {filename} -> {local_path}")
        resolved = download_file(MODEL_REPO, filename, local_path)
        print(f"  完成: {resolved}")

    chemrxn_dir = "./chemrxnextractor-training-modules"
    print(f"- 下载 chemrxnextractor 目录 -> {chemrxn_dir}")
    snapshot_download(
        repo_id=CHEMRXN_REPO,
        local_dir=chemrxn_dir,
        local_dir_use_symlinks=False,
    )
    print(f"  完成: {os.path.abspath(chemrxn_dir)}")
    print("全部权重下载完成。")


if __name__ == "__main__":
    download_weights()