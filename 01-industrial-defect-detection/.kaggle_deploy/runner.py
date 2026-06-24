# Auto-generated Kaggle Project Runner
import os
import sys
import base64
import zipfile
import io
import subprocess

print("=" * 60)
print("🚀 UNPACKING PROJECT BUNDLE ON KAGGLE WORKER")
print("=" * 60)

BUNDLE_B64 = """UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA=="""

raw_zip = base64.b64decode(BUNDLE_B64)
with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
    zf.extractall(".")

print("✅ Files unpacked successfully:")
for item in os.listdir("."):
    print("  -", item)

if os.path.exists("requirements.txt"):
    print("\n📦 Installing requirements.txt...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])



print("\n🚀 Executing entrypoint: train.py --epochs 2 --classifier-epochs 2 --img-size 128 --batch-size 16 --phase both")
print("=" * 60)
cmd = [sys.executable, "train.py"]
if "--epochs 2 --classifier-epochs 2 --img-size 128 --batch-size 16 --phase both":
    cmd.extend("--epochs 2 --classifier-epochs 2 --img-size 128 --batch-size 16 --phase both".split())

res = subprocess.run(cmd)
print(f"🏁 Execution finished with returncode {res.returncode}")
sys.exit(res.returncode)
