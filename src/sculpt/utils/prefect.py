import asyncio
import logging
import os
from typing import Optional

from prefect import get_client
from prefect.filesystems import LocalFileSystem
from prefect.results import ResultStore

PREFECT_LOCAL_STORAGE_PATH = os.getenv(
    "PREFECT_LOCAL_STORAGE_PATH", "./data/prefect-storage"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            metadata = flow_run.state.data
            if metadata and metadata.storage_key:
                try:
                    storage_key = metadata.storage_key
                    if "/" in storage_key:
                        filename = storage_key.split("/")[-1]
                    else:
                        filename = storage_key
                    storage = LocalFileSystem(basepath=PREFECT_LOCAL_STORAGE_PATH)
                    result_store = ResultStore(result_storage=storage)
                    data = await result_store.aread(filename)
                    return data
                except (ValueError, FileNotFoundError, OSError) as e:
                    logger.warning(
                        f"Failed to read result from custom storage: {e}. Falling back to default method."
                    )
                    return await flow_run.state.result(
                        raise_on_failure=True, fetch=True
                    )
            else:
                return await flow_run.state.result(raise_on_failure=True, fetch=True)
        elif flow_run.state.is_failed():
            error_msg = flow_run.state.message or "No error message available"
            raise Exception(f"Flow run {flow_run_id} failed: {error_msg}")
        elif flow_run.state.is_cancelled():
            raise Exception(f"Flow run {flow_run_id} was cancelled")
        elif flow_run.state.is_crashed():
            raise Exception(f"Flow run {flow_run_id} crashed")
        elif (
            flow_run.state.is_running()
            or flow_run.state.is_pending()
            or flow_run.state.is_scheduled()
        ):
            raise Exception(
                f"Flow run {flow_run_id} is not yet complete (state: {flow_run.state.type})"
            )
        else:
            raise Exception(
                f"Flow run {flow_run_id} in unexpected state: {flow_run.state.type}"
            )


def get_flow_run_result(flow_run_id: str):
    """Synchronous wrapper"""
    return asyncio.run(get_flow_run_result_async(flow_run_id))
