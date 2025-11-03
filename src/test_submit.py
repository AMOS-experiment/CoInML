"""
Test if .submit() works with your Prefect setup
Save as: test_submit.py
"""
import asyncio
from prefect import flow, task, get_run_logger
from prefect.client.orchestration import get_client
import time

@task
def simple_task():
    logger = get_run_logger()
    logger.info("Running simple task")
    time.sleep(2)
    return "Task complete"

@flow
def test_submit_flow():
    logger = get_run_logger()
    logger.info("Test flow starting")
    result = simple_task()
    return {"success": True, "result": result}

async def test_async_submit():
    """Test if async submit works"""
    print("Testing async submit...")
    
    # Method 1: Using submit (preferred)
    print("\n1. Testing .submit()...")
    try:
        flow_run = test_submit_flow.submit()
        print(f"✅ Submit worked! Flow run ID: {flow_run.id}")
        print(f"Check http://127.0.0.1:4200/flow-runs/flow-run/{flow_run.id}")
        
        # Wait for it
        result = flow_run.wait()
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Submit failed: {e}")
    
    # Method 2: Using apply_async
    print("\n2. Testing .apply_async()...")
    try:
        future = test_submit_flow.apply_async()
        print(f"✅ Apply async worked!")
        result = future.wait()
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Apply async failed: {e}")
    
    # Method 3: Create deployment and run
    print("\n3. Testing deployment method...")
    try:
        from prefect.deployments import Deployment
        
        deployment = Deployment.build_from_flow(
            flow=test_submit_flow,
            name="test-deployment",
        )
        deployment_id = deployment.apply()
        
        async with get_client() as client:
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id=deployment_id
            )
            print(f"✅ Deployment worked! Flow run: {flow_run.id}")
            return True
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
    
    return False

if __name__ == "__main__":
    # First test sync execution (always works)
    print("Testing direct execution (should work)...")
    result = test_submit_flow()
    print(f"Direct execution result: {result}")
    
    # Now test async submission
    print("\n" + "="*50)
    success = asyncio.run(test_async_submit())
    
    if not success:
        print("\n" + "="*50)
        print("SOLUTION: Since .submit() doesn't work with your setup,")
        print("we need to use a background task runner.")
        print("See the next file for a working solution.")


"""
Working solution using background tasks
Save as: sculpt/callbacks/umap_prefect_callbacks.py
"""
from dash import callback, Input, Output, State, html, dcc, ALL, no_update
import plotly.express as px
from prefect import Flow
from prefect.task_runners import SequentialTaskRunner
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from sculpt.flows.umap_flow import umap_analysis_flow

# Store running flows
RUNNING_FLOWS = {}
executor = ThreadPoolExecutor(max_workers=2)

def run_flow_in_background(flow_id, stored_files, selected_ids, num_neighbors, 
                           min_dist, sample_frac, selected_features_list):
    """Run flow in background thread and store results"""
    try:
        # This creates a Prefect flow run that shows in UI
        with Flow("UMAP-Analysis", task_runner=SequentialTaskRunner()) as flow:
            result = umap_analysis_flow(
                stored_files=stored_files,
                selected_ids=selected_ids,
                num_neighbors=num_neighbors,
                min_dist=min_dist,
                sample_frac=sample_frac,
                selected_features_list=selected_features_list,
                dbscan_eps=0.5,
                dbscan_min_samples=5
            )
        
        # Run the flow - this should show in Prefect UI
        flow_state = flow.run()
        RUNNING_FLOWS[flow_id] = {"status": "completed", "result": result}
    except Exception as e:
        RUNNING_FLOWS[flow_id] = {"status": "failed", "error": str(e)}


@callback(
    [Output("umap-graph-container", "children"),
     Output("debug-output", "children", allow_duplicate=True),
     Output("umap-flow-status", "children")],
    [Input("run-umap", "n_clicks"),
     Input("umap-check-interval", "n_intervals")],
    [State("stored-files", "data"),
     State("umap-file-selector", "value"),
     State("num-neighbors", "value"),
     State("min-dist", "value"),
     State("sample-frac", "value"),
     State({"type": "feature-selector-graph1", "category": ALL}, "value"),
     State("umap-flow-id-store", "data")],
    prevent_initial_call=True,
)
def run_umap_with_monitoring(n_clicks, n_intervals, stored_files, selected_ids,
                            num_neighbors, min_dist, sample_frac, 
                            selected_features_list, flow_id_data):
    """Run UMAP and monitor progress"""
    
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Start new flow
    if triggered == "run-umap" and stored_files and selected_ids:
        flow_id = str(uuid.uuid4())
        
        # Submit to executor
        RUNNING_FLOWS[flow_id] = {"status": "running"}
        executor.submit(
            run_flow_in_background,
            flow_id, stored_files, selected_ids,
            num_neighbors, min_dist, sample_frac, selected_features_list
        )
        
        return (
            html.Div("Starting UMAP analysis..."),
            f"Flow ID: {flow_id}",
            dcc.Store(id="umap-flow-id-store", data=flow_id)
        )
    
    # Check progress
    elif triggered == "umap-check-interval" and flow_id_data:
        if flow_id_data in RUNNING_FLOWS:
            flow_info = RUNNING_FLOWS[flow_id_data]
            
            if flow_info["status"] == "running":
                return (
                    html.Div("UMAP running... Check Prefect UI"),
                    "Still running",
                    no_update
                )
            elif flow_info["status"] == "completed":
                result = flow_info["result"]
                if result and result.get('success'):
                    umap_df = result['umap_df']
                    fig = px.scatter(
                        umap_df, x='UMAP1', y='UMAP2', color='Cluster',
                        title=f"UMAP ({len(umap_df)} points)"
                    )
                    
                    # Clear the flow from memory
                    del RUNNING_FLOWS[flow_id_data]
                    
                    return (
                        dcc.Graph(figure=fig),
                        "Completed!",
                        html.Div("✅ Analysis complete", style={"color": "green"})
                    )
            elif flow_info["status"] == "failed":
                return (
                    html.Div(f"Error: {flow_info['error']}", style={"color": "red"}),
                    flow_info['error'],
                    html.Div("❌ Analysis failed", style={"color": "red"})
                )
    
    return no_update, no_update, no_update


# Add these components to your layout:
# dcc.Store(id="umap-flow-id-store"),
# dcc.Interval(id="umap-check-interval", interval=2000),
# html.Div(id="umap-flow-status")