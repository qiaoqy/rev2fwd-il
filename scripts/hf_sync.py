#!/usr/bin/env python3
"""
HuggingFace Hub Sync Tool
Upload/download datasets and model checkpoints

Login first: huggingface-cli login

Usage:


    # Upload model (entire experiment folder, ignores lerobot_dataset)
    python scripts/hf_sync.py upload-model runs/ditflow_piper_0210_B
    python scripts/hf_sync.py upload-model runs/ditflow_piper_0210_A

    python scripts/hf_sync.py upload-model runs/diffusion_piper_teleop_B_0205
    python scripts/hf_sync.py upload-model runs/dp_pickplace_piper_0210_A
    python scripts/hf_sync.py upload-model runs/dp_pickplace_piper_0210_B
    
    # Download model (auto-appends model name as subdirectory)
    python scripts/hf_sync.py download-model diffusion_piper_teleop_A --local_dir /media/qiyuan/SSDQQY/runs       # -> runs/diffusion_piper_teleop_A
    python scripts/hf_sync.py download-model diffusion_piper_teleop_B_0205 --local_dir /media/qiyuan/SSDQQY/runs       # -> /media/qiyuan/SSDQQY/diffusion_piper_teleop_A
    python scripts/hf_sync.py download-model ditflow_piper_teleop_B_0206 --local_dir /media/qiyuan/SSDQQY/runs
    python scripts/hf_sync.py download-model ditflow_piper_0210_A --local_dir /media/qiyuan/SSDQQY/runs # pick& place with new camera setup
    python scripts/hf_sync.py download-model dp_pickplace_piper_0210_A --local_dir /media/qiyuan/SSDQQY/runs

    # Upload dataset
    python scripts/hf_sync.py upload-dataset data/pick_place_piper_A
    
    # Download dataset (auto-appends dataset name as subdirectory)
    python scripts/hf_sync.py download-dataset pick_place_piper_A --local_dir /media/qiyuan/SSDQQY/           # -> /media/qiyuan/SSDQQY/pick_place_piper_A



    
    # List all repos
    python scripts/hf_sync.py list
"""

import argparse
import os
import time
from pathlib import Path
from functools import wraps

from huggingface_hub import (
    HfApi, 
    snapshot_download, 
    upload_folder,
    list_repo_files,
    whoami,
)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def retry_on_timeout(func):
    """Decorator to retry function on timeout/connection errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, TimeoutError, OSError) as e:
                if attempt < MAX_RETRIES:
                    print(f"Warning: Connection timeout (attempt {attempt}/{MAX_RETRIES}). Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Error: Failed after {MAX_RETRIES} attempts: {e}")
                    raise
            except Exception as e:
                # Check if it's a requests timeout
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    if attempt < MAX_RETRIES:
                        print(f"Warning: Connection issue (attempt {attempt}/{MAX_RETRIES}). Retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f"Error: Failed after {MAX_RETRIES} attempts: {e}")
                        raise
                else:
                    raise
        return None
    return wrapper


@retry_on_timeout
def get_username():
    """Get current HuggingFace username"""
    try:
        info = whoami()
        return info["name"]
    except Exception as e:
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise
        print(f"Error: Please login to HuggingFace Hub first")
        print(f"Run: huggingface-cli login")
        raise SystemExit(1)


@retry_on_timeout
def _upload_with_retry(folder_path: str, repo_id: str, repo_type: str):
    """Upload folder with retry on timeout"""
    ignore = ["*.pyc", "__pycache__", ".git", "wandb/*", "lerobot_dataset/*"] if repo_type == "model" else ["*.pyc", "__pycache__", ".git"]
    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=ignore,
    )


@retry_on_timeout
def _download_with_retry(repo_id: str, repo_type: str, local_dir: str, revision: str):
    """Download snapshot with retry on timeout"""
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        revision=revision,
    )


def upload_model(local_path: str, repo_name: str = None, private: bool = False):
    """Upload model checkpoints to HuggingFace Hub"""
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"Error: Path does not exist {local_path}")
        return
    
    username = get_username()
    
    # Smart naming: if folder is "checkpoints", use parent folder name
    if repo_name:
        final_repo_name = repo_name
    elif local_path.name == "checkpoints":
        final_repo_name = local_path.parent.name
    else:
        final_repo_name = local_path.name
    
    repo_id = f"{username}/{final_repo_name}"
    
    api = HfApi()
    
    # Create repo if not exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"Repo: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return
    
    # Upload folder
    print(f"Uploading {local_path} -> {repo_id}")
    _upload_with_retry(str(local_path), repo_id, "model")
    print(f"Upload complete! https://huggingface.co/{repo_id}")


def download_model(repo_name: str, local_dir: str = None, revision: str = "main"):
    """Download model checkpoints from HuggingFace Hub"""
    username = get_username()
    
    # Add username prefix if repo_name doesn't contain /
    if "/" not in repo_name:
        repo_id = f"{username}/{repo_name}"
    else:
        repo_id = repo_name
    
    # Auto-append repo name as subdirectory
    model_name = repo_name.split('/')[-1]
    if local_dir:
        local_dir = str(Path(local_dir) / model_name)
    else:
        local_dir = f"runs/{model_name}"
    
    print(f"Downloading {repo_id} -> {local_dir}")
    _download_with_retry(repo_id, "model", local_dir, revision)
    print(f"Download complete! {local_dir}")


def upload_dataset(local_path: str, repo_name: str = None, private: bool = False):
    """Upload dataset to HuggingFace Hub"""
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"Error: Path does not exist {local_path}")
        return
    
    username = get_username()
    repo_name = repo_name or local_path.name
    repo_id = f"{username}/{repo_name}"
    
    api = HfApi()
    
    # Create dataset repo
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"Repo: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return
    
    # Upload
    print(f"Uploading {local_path} -> {repo_id}")
    _upload_with_retry(str(local_path), repo_id, "dataset")
    print(f"Upload complete! https://huggingface.co/datasets/{repo_id}")


def download_dataset(repo_name: str, local_dir: str = None, revision: str = "main"):
    """Download dataset from HuggingFace Hub"""
    username = get_username()
    
    if "/" not in repo_name:
        repo_id = f"{username}/{repo_name}"
    else:
        repo_id = repo_name
    
    # Auto-append repo name as subdirectory
    dataset_name = repo_name.split('/')[-1]
    if local_dir:
        local_dir = str(Path(local_dir) / dataset_name)
    else:
        local_dir = f"data/{dataset_name}"
    
    print(f"Downloading {repo_id} -> {local_dir}")
    _download_with_retry(repo_id, "dataset", local_dir, revision)
    print(f"Download complete! {local_dir}")


def list_repos():
    """List all repos for current user"""
    username = get_username()
    api = HfApi()
    
    print(f"\n=== {username}'s Models ===")
    models = api.list_models(author=username)
    for model in models:
        print(f"  - {model.id}")
    
    print(f"\n=== {username}'s Datasets ===")
    datasets = api.list_datasets(author=username)
    for ds in datasets:
        print(f"  - {ds.id}")


def list_files(repo_id: str, repo_type: str = "model"):
    """List files in a repo"""
    try:
        files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
        print(f"\n=== {repo_id} ({repo_type}) Files ===")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Hub Sync Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # upload-model
    p_upload_model = subparsers.add_parser("upload-model", help="Upload model checkpoints")
    p_upload_model.add_argument("local_path", help="Local path")
    p_upload_model.add_argument("--repo_name", help="Repo name (default: folder name)")
    p_upload_model.add_argument("--private", action="store_true", help="Make repo private")
    
    # download-model
    p_download_model = subparsers.add_parser("download-model", help="Download model checkpoints")
    p_download_model.add_argument("repo_name", help="Repo name or full repo_id")
    p_download_model.add_argument("--local_dir", help="Local save path")
    p_download_model.add_argument("--revision", default="main", help="Branch/revision")
    
    # upload-dataset
    p_upload_dataset = subparsers.add_parser("upload-dataset", help="Upload dataset")
    p_upload_dataset.add_argument("local_path", help="Local path")
    p_upload_dataset.add_argument("--repo_name", help="Repo name (default: folder name)")
    p_upload_dataset.add_argument("--private", action="store_true", help="Make repo private")
    
    # download-dataset
    p_download_dataset = subparsers.add_parser("download-dataset", help="Download dataset")
    p_download_dataset.add_argument("repo_name", help="Repo name or full repo_id")
    p_download_dataset.add_argument("--local_dir", help="Local save path")
    p_download_dataset.add_argument("--revision", default="main", help="Branch/revision")
    
    # list
    p_list = subparsers.add_parser("list", help="List all repos")
    
    # list-files
    p_list_files = subparsers.add_parser("list-files", help="List repo files")
    p_list_files.add_argument("repo_id", help="Repo ID")
    p_list_files.add_argument("--type", choices=["model", "dataset"], default="model")
    
    args = parser.parse_args()
    
    if args.command == "upload-model":
        upload_model(args.local_path, args.repo_name, private=args.private)
    elif args.command == "download-model":
        download_model(args.repo_name, args.local_dir, args.revision)
    elif args.command == "upload-dataset":
        upload_dataset(args.local_path, args.repo_name, private=args.private)
    elif args.command == "download-dataset":
        download_dataset(args.repo_name, args.local_dir, args.revision)
    elif args.command == "list":
        list_repos()
    elif args.command == "list-files":
        list_files(args.repo_id, args.type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
