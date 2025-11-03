#!/bin/bash

# Script to initialize Prefect work pools and register flows

echo "🔧 Initializing Prefect..."

# Wait for Prefect server to be ready
echo "Waiting for Prefect server to be ready..."
sleep 10

# Create a work pool if it doesn't exist
echo "Creating work pool..."
docker exec src-prefect-server-1 prefect work-pool create \
    --type docker \
    default-agent-pool \
    2>/dev/null || echo "Work pool already exists"

# Register the flows
echo "Registering flows..."
docker exec src-sculpt-app-1 python -c "
import sys
sys.path.insert(0, '/app')
import asyncio
from prefect import flow
from prefect.deployments import Deployment
import os

os.environ['PREFECT_API_URL'] = 'http://prefect-server:4200/api'

# Import flows
from sculpt.flows.umap_flow import umap_analysis_flow

async def register():
    # Register UMAP flow
    deployment = await Deployment.build_from_flow(
        flow=umap_analysis_flow,
        name='umap-docker',
        work_pool_name='default-agent-pool',
        tags=['sculpt', 'umap']
    )
    id = await deployment.apply()
    print(f'✓ UMAP flow deployed: {id}')

asyncio.run(register())
"

echo "✅ Prefect initialization complete!"
echo "📊 Check deployments at: http://localhost:4200/deployments"