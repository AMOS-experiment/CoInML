"""
Test if flows are reaching Prefect Cloud
Save as: test_cloud.py
"""

import asyncio
import time

from prefect.client.orchestration import get_client

from prefect import flow, get_run_logger, task


@task
def simple_task():
    logger = get_run_logger()
    logger.info("Task running in Prefect Cloud!")
    time.sleep(2)
    return "Success from cloud"


@flow(name="Test Cloud Flow", log_prints=True)
def test_cloud_flow():
    logger = get_run_logger()
    logger.info("Testing Prefect Cloud connection...")
    result = simple_task()
    logger.info(f"Result: {result}")
    return {"status": "success", "message": result}


async def check_cloud_connection():
    """Check if we're connected to cloud"""
    try:
        async with get_client() as client:
            # This should show cloud URL
            print(f"API URL: {client.api_url}")

            # Check health
            health = await client.api_healthcheck()
            print(f"API Health: {health}")

            # List recent flows (if any)
            flows = await client.read_flows(limit=5)
            print(f"Number of flows in cloud: {len(flows)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("1. Checking cloud connection...")
    asyncio.run(check_cloud_connection())

    print("\n2. Running test flow...")
    # This should appear in Prefect Cloud
    state = test_cloud_flow()

    print(f"\n3. Flow completed with state: {state}")
    print("\n✅ Check https://app.prefect.cloud to see if the flow appeared!")
    print("   Look in the 'Flow Runs' section")
