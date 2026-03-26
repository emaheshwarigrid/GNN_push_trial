import os
from pathlib import Path

# --- Path Safety Block ---
if Path.cwd().name in ['datascripts']:
    os.chdir(Path.cwd().parent)

def check_project_files_sorted(directory="."):
    file_list = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # We still skip .venv and .git because you should NEVER push those to GitHub
        if '.venv' in root or '.git' in root:
            continue
            
        for name in files:
            filepath = os.path.join(root, name)
            try:
                # Get size in bytes
                file_size_bytes = os.path.getsize(filepath)
                rel_path = os.path.relpath(filepath, directory)
                file_list.append((rel_path, file_size_bytes))
            except OSError:
                continue

    # Sort by size (index 1) in descending order
    file_list.sort(key=lambda x: x[1], reverse=True)

    print(f"{'File Name':<45} | {'Size (MB)':<10}")
    print("-" * 60)
    
    for path, size in file_list:
        # Display in MB for better readability of large files
        size_mb = size / (1024 * 1024)
        print(f"{path:<45} | {size_mb:>10.2f} MB")

# Run the check
check_project_files_sorted()