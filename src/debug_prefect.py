"""
Create a file called debug_prefect.py to find the issue
"""
import os
import sys

print("=== DEBUGGING PREFECT CONFIGURATION ===\n")

# Check environment variables
print("1. Environment Variables:")
print(f"   PREFECT_API_URL from env: {os.environ.get('PREFECT_API_URL', 'NOT SET')}")
print(f"   PREFECT_API_KEY from env: {os.environ.get('PREFECT_API_KEY', 'NOT SET')[:10]}..." if os.environ.get('PREFECT_API_KEY') else "   PREFECT_API_KEY from env: NOT SET")

# Import Prefect and check settings
print("\n2. Before importing Prefect:")
print(f"   sys.path[0]: {sys.path[0]}")

import prefect
from prefect.settings import PREFECT_API_URL, PREFECT_API_KEY

print("\n3. After importing Prefect:")
print(f"   PREFECT_API_URL.value(): {PREFECT_API_URL.value()}")
print(f"   prefect.__version__: {prefect.__version__}")

# Check if there's a prefect.yaml or .prefectignore file
print("\n4. Checking for config files:")
import os
for file in ['prefect.yaml', '.prefectignore', '.env', 'pyproject.toml']:
    if os.path.exists(file):
        print(f"   Found: {file}")
        if file == '.env':
            with open(file, 'r') as f:
                for line in f:
                    if 'PREFECT' in line:
                        print(f"      -> {line.strip()}")

# Now import your app modules and check again
print("\n5. After importing sculpt modules:")
try:
    from sculpt.flows.umap_flow import umap_analysis_flow
    print(f"   PREFECT_API_URL after import: {PREFECT_API_URL.value()}")
except Exception as e:
    print(f"   Error importing: {e}")

# Check if any module is overriding
print("\n6. Checking sys.modules for overrides:")
for module_name, module in sys.modules.items():
    if module and hasattr(module, '__file__') and module.__file__:
        if 'sculpt' in module_name or 'app' in module_name:
            try:
                with open(module.__file__, 'r') as f:
                    content = f.read()
                    if 'PREFECT_API_URL' in content and '127.0.0.1:4200' in content:
                        print(f"   ⚠️ Found override in: {module.__file__}")
                        # Show the line
                        for i, line in enumerate(content.split('\n'), 1):
                            if 'PREFECT_API_URL' in line and '127.0.0.1:4200' in line:
                                print(f"      Line {i}: {line.strip()}")
            except:
                pass

print("\n7. Final check:")
print(f"   Current PREFECT_API_URL: {PREFECT_API_URL.value()}")
if "cloud.prefect.io" in str(PREFECT_API_URL.value()):
    print("   ✅ Cloud configuration is active")
else:
    print("   ❌ Local configuration is active")