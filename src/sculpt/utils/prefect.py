import asyncio
from typing import Optional

from prefect import get_client


async def _schedule(
    deployment_name: str,
    flow_run_name: str,
    parameters: Optional[dict] = None,
    tags: Optional[list] = [],
):
    async with get_client() as client:
        deployment = await client.read_deployment_by_name(deployment_name)
        assert (
            deployment
        ), f"No deployment found in config for deployment_name {deployment_name}"
        flow_run = await client.create_flow_run_from_deployment(
            deployment.id,
            parameters=parameters,
            name=flow_run_name,
            tags=tags,
        )
    return flow_run.id


def schedule_prefect_flow(
    deployment_name: str,
    parameters: Optional[dict] = None,
    flow_run_name: Optional[str] = None,
    tags: Optional[list] = [],
):
    if not flow_run_name:
        model_name = parameters["model_name"]
        flow_run_name = f"{deployment_name}: {model_name}"
    flow_run_id = asyncio.run(
        _schedule(deployment_name, flow_run_name, parameters, tags)
    )
    return flow_run_id


async def _get_flow_run_state(flow_run_id):
    async with get_client() as client:
        flow_run = await client.read_flow_run(flow_run_id)
        return flow_run.state


def get_flow_run_state(flow_run_id):
    flow_run_state = asyncio.run(_get_flow_run_state(flow_run_id))
    return flow_run_state.type


async def get_flow_run_result_async(flow_run_id: str):
    """Get the result from a completed flow run"""
    async with get_client() as client:
        flow_run = await client.read_flow_run(flow_run_id)

        if flow_run.state.is_completed():
            # TODO: Change this to read the dataframe that the job generates
            return await flow_run.state.result()
        elif flow_run.state.is_failed():
            raise Exception(f"Flow run failed: {flow_run.state.message}")
        elif flow_run.state.is_cancelled():
            raise Exception("Flow run was cancelled")
        else:
            raise Exception(f"Flow run in unexpected state: {flow_run.state.type}")


def get_flow_run_result(flow_run_id: str):
    """Synchronous wrapper"""
    return asyncio.run(get_flow_run_result_async(flow_run_id))
