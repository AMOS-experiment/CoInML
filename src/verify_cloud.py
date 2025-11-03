"""
Save as verify_cloud.py and run it
"""
# Force cloud settings FIRST
import os
os.environ['PREFECT_API_KEY'] = 'pnu_iJvRtT7boZ39gedtlcz8qgcoPDVmRP0r7D2P'
os.environ['PREFECT_API_URL'] = 'https://api.prefect.cloud/api/accounts/50347e91-ac53-44ce-a917-457c8fd18911/workspaces/3cf33d5e-bea0-4672-a917-ff80af7f037f'

print("Environment variables set:")
print(f"PREFECT_API_URL: {os.environ.get('PREFECT_API_URL')}")
print(f"PREFECT_API_KEY: {os.environ.get('PREFECT_API_KEY')[:20]}...")

# Now import and run a flow
from sculpt.flows.umap_flow import test_umap_flow

print("\nRunning test flow...")
result = test_umap_flow(num_neighbors=15, min_dist=0.1)
print(f"Flow completed: {result}")
print("\n✅ Check https://app.prefect.cloud - the flow should be there!")