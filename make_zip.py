import zipfile
import os

base_dir = r"C:\Users\Devrsh\.gemini\antigravity\scratch\warehouse-brain"
out_zip_1 = r"C:\Users\Devrsh\.gemini\antigravity\scratch\warehouse-brain.zip"
out_zip_2 = r"C:\Users\Devrsh\.gemini\antigravity\brain\8dc15ddd-c4f0-4ee7-b213-6adad646f8d4\warehouse-brain.zip"

exclude_files = {'test.py', 'check_env.py', 'powershell.cmd', 'make_zip.py'}

def make_clean_zip(target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for f in files:
                if f in exclude_files or f.endswith('.pyc') or f.endswith('.tmp'):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_dir)
                archive_name = os.path.join('warehouse-brain', rel_path)
                zf.write(full_path, archive_name)
    print(f"Created: {target_path} (size: {os.path.getsize(target_path):,} bytes)")

make_clean_zip(out_zip_1)
make_clean_zip(out_zip_2)
