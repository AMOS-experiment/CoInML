#!/usr/bin/env python
"""
Test script to verify Prefect is working in Docker.
Run this from your host machine.
"""

import asyncio
import os
from prefect import flow, task
from prefect.client.orchestration import PrefectClient
import pandas as pd
import numpy as np

# Point to Docker Prefect server
PREFECT_API_URL = "http://localhost:4200/api"
os.environ["PREFECT_API_URL"] = PREFECT_API_URL


@task(name="generate-data")
def generate_test_data():
    """Generate test data."""
    print("Generating test data...")
    return pd.DataFrame(
        np.random.randn(100, 5),
        columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
    )


@task(name="process-data")
def process_data(df):
    """Process the data."""
    print(f"Processing {len(df)} rows...")
    return df.mean()


@flow(name="docker-test-flow")
def test_flow():
    """Simple test flow."""
    data = generate_test_data()
    result = process_data(data)
    print(f"Result: {result}")
    return result


async def test_prefect_connection():
    """Test connection to Prefect server."""
    try:
        # Use explicit URL
        client = PrefectClient(api=PREFECT_API_URL)
        
        # Check health
        health = await client.api_healthcheck()
        if health:
            print("✅ Prefect server is healthy!")
        else:
            print("⚠️ Prefect server health check returned None")
        
        # List flows
        flows = await client.read_flows()
        print(f"\n📋 Found {len(flows)} flows:")
        for f in flows:
            print(f"  - {f.name}")
        
        # List deployments
        deployments = await client.read_deployments()
        print(f"\n🚀 Found {len(deployments)} deployments:")
        for d in deployments:
            print(f"  - {d.name}")
            
        # List work pools
        work_pools = await client.read_work_pools()
        print(f"\n🏊 Found {len(work_pools)} work pools:")
        for wp in work_pools:
            print(f"  - {wp.name} ({wp.type}) - Status: {wp.status}")
                
    except Exception as e:
        print(f"❌ Error connecting to Prefect: {e}")
        print(f"   URL being used: {PREFECT_API_URL}")
        print("\nMake sure Docker containers are running:")
        print("  docker-compose ps")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔍 Testing Prefect Docker setup...\n")
    
    # Test connection
    asyncio.run(test_prefect_connection())
    
    # Run a test flow
    print("\n🏃 Running test flow...")
    try:
        result = test_flow()
        print("\n✅ Test complete! Check http://localhost:4200 to see the flow run.")
    except Exception as e:
        print(f"❌ Error running flow: {e}")