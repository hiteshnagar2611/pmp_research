"""
File utilities: path handling, file operations, and I/O helpers.

Features:
  - Path utilities
  - File reading/writing
  - Directory management
  - Batch file operations
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union
import json
import pickle
import shutil
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Path Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if needed.

    Args:
        path: directory path

    Returns:
        path: Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_files(
    directory: Union[str, Path],
    pattern: str = '*',
    recursive: bool = False,
) -> List[Path]:
    """
    Get files matching pattern in directory.

    Args:
        directory: directory path
        pattern: glob pattern (e.g., '*.pdb')
        recursive: search recursively

    Returns:
        files: list of file paths
    """
    directory = Path(directory)

    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    return sorted(files)


def find_file(
    name: str,
    root_dir: Union[str, Path] = '.',
) -> Optional[Path]:
    """
    Find file by name in directory tree.

    Args:
        name: file name
        root_dir: root directory to search from

    Returns:
        path: file path or None if not found
    """
    root_dir = Path(root_dir)

    for file_path in root_dir.rglob(name):
        return file_path

    return None


def get_unique_filename(
    base_name: str,
    directory: Union[str, Path] = '.',
) -> Path:
    """
    Get unique filename (add timestamp if needed).

    Args:
        base_name: base file name
        directory: directory to check

    Returns:
        path: unique file path
    """
    directory = Path(directory)
    base_path = directory / base_name

    if not base_path.exists():
        return base_path

    # Add timestamp
    stem = base_path.stem
    suffix = base_path.suffix
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_name = f"{stem}_{timestamp}{suffix}"
    unique_path = directory / unique_name

    return unique_path


# ─────────────────────────────────────────────────────────────────────────────
# File Reading/Writing
# ─────────────────────────────────────────────────────────────────────────────

def read_json(file_path: Union[str, Path]) -> dict:
    """
    Read JSON file.

    Args:
        file_path: path to JSON file

    Returns:
        data: parsed JSON data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(
    data: dict,
    file_path: Union[str, Path],
    indent: int = 2,
):
    """
    Write JSON file.

    Args:
        data: data to write
        file_path: output file path
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def read_pickle(file_path: Union[str, Path]):
    """
    Read pickle file.

    Args:
        file_path: path to pickle file

    Returns:
        data: unpickled data
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(
    data,
    file_path: Union[str, Path],
):
    """
    Write pickle file.

    Args:
        data: data to pickle
        file_path: output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def read_text(file_path: Union[str, Path]) -> str:
    """
    Read text file.

    Args:
        file_path: path to text file

    Returns:
        content: file content as string
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return content


def write_text(
    content: str,
    file_path: Union[str, Path],
):
    """
    Write text file.

    Args:
        content: text content
        file_path: output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        f.write(content)


def append_text(
    content: str,
    file_path: Union[str, Path],
):
    """
    Append to text file.

    Args:
        content: text content
        file_path: output file path
    """
    file_path = Path(file_path)

    with open(file_path, 'a') as f:
        f.write(content)


# ─────────────────────────────────────────────────────────────────────────────
# Directory Operations
# ─────────────────────────────────────────────────────────────────────────────

def copy_tree(src: Union[str, Path], dst: Union[str, Path]):
    """
    Copy directory tree.

    Args:
        src: source directory
        dst: destination directory
    """
    src = Path(src)
    dst = Path(dst)

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)


def remove_tree(path: Union[str, Path]):
    """
    Remove directory tree.

    Args:
        path: directory path
    """
    path = Path(path)

    if path.exists():
        shutil.rmtree(path)


def list_directories(path: Union[str, Path]) -> List[Path]:
    """
    List subdirectories.

    Args:
        path: parent directory

    Returns:
        directories: list of directory paths
    """
    path = Path(path)

    directories = [d for d in path.iterdir() if d.is_dir()]

    return sorted(directories)


def get_directory_size(path: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes.

    Args:
        path: directory path

    Returns:
        size: total size in bytes
    """
    path = Path(path)
    total_size = 0

    for file_path in path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    return total_size


def format_size(size: int) -> str:
    """
    Format file size as human-readable string.

    Args:
        size: size in bytes

    Returns:
        formatted: formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} TB"


# ─────────────────────────────────────────────────────────────────────────────
# Batch Operations
# ─────────────────────────────────────────────────────────────────────────────

def copy_files(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    pattern: str = '*',
) -> int:
    """
    Copy matching files to destination.

    Args:
        src_dir: source directory
        dst_dir: destination directory
        pattern: glob pattern

    Returns:
        count: number of files copied
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    ensure_directory(dst_dir)

    files = get_files(src_dir, pattern=pattern)

    for src_file in files:
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)

    return len(files)


def remove_files(
    directory: Union[str, Path],
    pattern: str = '*',
) -> int:
    """
    Remove matching files.

    Args:
        directory: directory
        pattern: glob pattern

    Returns:
        count: number of files removed
    """
    directory = Path(directory)
    files = get_files(directory, pattern=pattern)

    for file_path in files:
        file_path.unlink()

    return len(files)


if __name__ == "__main__":
    # Test file utilities
    print("File utilities loaded successfully")

    # Test path utilities
    test_dir = Path('/tmp/test_files')
    ensure_directory(test_dir)

    # Test file operations
    test_file = test_dir / 'test.json'
    write_json({'key': 'value'}, test_file)
    data = read_json(test_file)
    print(f"JSON read/write: {data}")

    # Test text operations
    text_file = test_dir / 'test.txt'
    write_text('Hello, World!', text_file)
    content = read_text(text_file)
    print(f"Text read/write: {content}")

    # Cleanup
    remove_tree(test_dir)

    print("✓ File utilities working!")
