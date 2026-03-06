"""
Prefect-enabled UMAP callbacks for SCULPT
Non-blocking UI with real-time progress tracking
(Version without dash-bootstrap-components)
"""
from dash import callback, Input, Output, State, html, dcc, ALL, no_update
import plotly.express as px
import plotly.graph_objects as go
from prefect import get_client
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

from sculpt.flows.umap_flow import umap_analysis_flow
from sculpt.utils.ui import create_smart_confidence_ui


@callback(
    Output("umap-flow-run-store", "data"),
    Output("umap-status-message", "children"),
    Output("umap-progress-card", "style"),
    Output("umap-progress-interval", "disabled"),
    Input("run-umap", "n_clicks"),
    State("stored-files", "data"),
    State("umap-file-selector", "value"),
    State("num-neighbors", "value"),
    State("min-dist", "value"),
    State("sample-frac", "value"),
    State({"type": "feature-selector-graph1", "category": ALL}, "value"),
    prevent_initial_call=True,
)
def start_umap_flow(
    n_clicks,
    stored_files,
    selected_ids,
    num_neighbors,
    min_dist,
    sample_frac,
    selected_features_list,
):
    """
    Start the UMAP analysis flow using Prefect (non-blocking!)
    """
    if not stored_files:
        return {}, html.Div("❌ No files uploaded.", style={"color": "red"}), {"display": "none"}, True
    
    if not selected_ids:
        return {}, html.Div("❌ No files selected for UMAP.", style={"color": "red"}), {"display": "none"}, True
    
    # Submit the Prefect flow
    try:
        # Check Prefect connection
        from prefect import get_client
        import asyncio
        
        async def check_prefect():
            async with get_client() as client:
                health = await client.api_healthcheck()
                print(f"Prefect health: {health}")
        
        asyncio.run(check_prefect())
        print("Prefect server connection OK")
        flow_run = umap_analysis_flow.with_options(
            name=f"umap-n{num_neighbors}-d{min_dist}-{len(selected_ids)}files"
        ).submit(
            stored_files=stored_files,
            selected_ids=selected_ids,
            num_neighbors=num_neighbors,
            min_dist=min_dist,
            sample_frac=sample_frac,
            selected_features_list=selected_features_list,
            dbscan_eps=0.5,
            dbscan_min_samples=5
        )

          
        return (
            {"flow_run_id": str(flow_run.id), "flow_name": flow_run.name},
            html.Div([
                html.Span("🔄 ", style={"color": "blue"}),
                html.Span(f"UMAP analysis started: {flow_run.name}", style={"color": "blue"}),
                html.Br(),
                html.Small("Monitor progress below. The UI remains responsive!", style={"color": "gray"})
            ]),
            {"display": "block"},  # Show progress card
            False  # Enable interval for monitoring
        )
        
    except Exception as e:
        return (
            {},
            html.Div([
                "❌ Error starting UMAP flow: ",
                html.Code(str(e))
            ], style={"color": "red"}),
            {"display": "none"},
            True
        )


@callback(
    Output("umap-progress-bar", "style"),
    Output("umap-progress-bar", "children"),
    Output("umap-status-details", "children"),
    Output("umap-graph-container", "children"),
    Output("debug-output", "children", allow_duplicate=True),
    Output("combined-data-store", "data", allow_duplicate=True),
    Output("umap-quality-metrics", "children", allow_duplicate=True),
    Output("umap-progress-card", "style", allow_duplicate=True),
    Output("umap-progress-interval", "disabled", allow_duplicate=True),
    Input("umap-progress-interval", "n_intervals"),
    State("umap-flow-run-store", "data"),
    prevent_initial_call=True,
)
def monitor_umap_flow_progress(n_intervals, flow_store):
    """
    Monitor the progress of the UMAP Prefect flow
    """
    if not flow_store or "flow_run_id" not in flow_store:
        return (
            {"width": "0%", "height": "20px", "backgroundColor": "#007bff", "borderRadius": "4px", "transition": "width 0.3s ease", "textAlign": "center", "color": "white", "lineHeight": "20px"},
            "0%",
            "",
            no_update, no_update, no_update, no_update,
            {"display": "none"}, True
        )
    
    try:
        # Get flow run status asynchronously
        async def get_flow_status():
            async with get_client() as client:
                flow_run = await client.read_flow_run(flow_store['flow_run_id'])
                
                # Get task runs for detailed progress
                task_runs = await client.read_task_runs(
                    flow_run_filter={"id": {"any_": [flow_store['flow_run_id']]}}
                )
                
                return flow_run, task_runs
        
        flow_run, task_runs = asyncio.run(get_flow_status())
        
        # Determine progress based on state
        if flow_run.state.is_pending():
            return (
                {"width": "0%", "height": "20px", "backgroundColor": "#6c757d", "borderRadius": "4px", "transition": "width 0.3s ease", "textAlign": "center", "color": "white", "lineHeight": "20px"},
                "Queued",
                "Flow is queued, waiting to start...",
                no_update, no_update, no_update, no_update,
                {"display": "block", "marginBottom": "15px"}, False
            )
        
        elif flow_run.state.is_running():
            # Calculate progress from task states
            total_tasks = len(task_runs)
            completed_tasks = sum(1 for t in task_runs if t.state.is_completed())
            progress = int((completed_tasks / max(total_tasks, 1)) * 100)
            
            # Find current running task
            current_task = None
            for task in task_runs:
                if task.state.is_running():
                    current_task = task.name
                    break
            
            task_status = f"Running: {current_task or 'Initializing...'}"
            
            # Create detailed status
            status_details = html.Div([
                html.Div(f"Tasks: {completed_tasks}/{total_tasks} completed"),
                html.Div(f"Current: {current_task or 'Setting up...'}"),
                html.Small([
                    "View details in ",
                    html.A("Prefect UI", 
                          href=f"http://127.0.0.1:4200/flow-runs/flow-run/{flow_store['flow_run_id']}", 
                          target="_blank")
                ])
            ])
            
            return (
                {"width": f"{progress}%", "height": "20px", "backgroundColor": "#007bff", "borderRadius": "4px", "transition": "width 0.3s ease", "textAlign": "center", "color": "white", "lineHeight": "20px"},
                f"{progress}%",
                status_details,
                no_update, no_update, no_update, no_update,
                {"display": "block", "marginBottom": "15px"}, False
            )
        
        elif flow_run.state.is_completed():
            # Get the result
            result = flow_run.state.result()
            
            if result and result.get('success'):
                # Create UMAP visualization
                umap_df = result['umap_df']
                
                # Determine color column
                if 'cluster' in umap_df.columns:
                    color_col = 'cluster'
                    umap_df[color_col] = umap_df[color_col].astype(str)
                    title = f"UMAP Embedding (DBSCAN clusters: {result['clustering_info']['n_clusters']})"
                else:
                    color_col = 'file_label'
                    title = "UMAP Embedding"
                
                # Create the plot
                fig = px.scatter(
                    umap_df,
                    x='UMAP1',
                    y='UMAP2',
                    color=color_col,
                    title=title,
                    hover_data=['file_label'] if 'file_label' in umap_df.columns else None
                )
                
                fig.update_layout(height=600)
                
                # Create quality metrics
                metrics = create_smart_confidence_ui(
                    n_samples=result['metadata']['n_samples'],
                    n_features=result['metadata']['n_features'],
                    num_neighbors=15,  # You might want to get this from parameters
                    min_dist=0.1,
                    umap_random_state=42
                )
                
                # Debug output
                debug_text = f"""
                UMAP computation complete!
                - Samples: {result['metadata']['n_samples']}
                - Features: {result['metadata']['n_features']}
                - Clusters found: {result['clustering_info']['n_clusters']}
                - Computation time: {result['metadata']['computation_time']:.2f}s
                """
                
                return (
                    {"width": "100%", "height": "20px", "backgroundColor": "#28a745", "borderRadius": "4px", "textAlign": "center", "color": "white", "lineHeight": "20px"},
                    "Complete!",
                    html.Div("✅ UMAP analysis completed successfully!", style={"color": "green"}),
                    dcc.Graph(figure=fig),
                    debug_text,
                    result['combined_df'].to_dict('records'),
                    metrics,
                    {"display": "none"},  # Hide progress card
                    True  # Disable interval
                )
            else:
                return (
                    {"width": "0%", "height": "20px", "backgroundColor": "#dc3545", "borderRadius": "4px", "textAlign": "center", "color": "white", "lineHeight": "20px"},
                    "Failed",
                    html.Div("❌ Flow completed but no results returned", style={"color": "red"}),
                    no_update, no_update, no_update, no_update,
                    {"display": "none"}, True
                )
        
        elif flow_run.state.is_failed():
            error_msg = str(flow_run.state.message) if flow_run.state.message else "Unknown error"
            return (
                {"width": "0%", "height": "20px", "backgroundColor": "#dc3545", "borderRadius": "4px", "textAlign": "center", "color": "white", "lineHeight": "20px"},
                "Failed",
                html.Div([
                    "❌ Flow failed: ",
                    html.Code(error_msg),
                    html.Br(),
                    html.A("View details in Prefect UI", 
                          href=f"http://127.0.0.1:4200/flow-runs/flow-run/{flow_store['flow_run_id']}", 
                          target="_blank")
                ], style={"color": "red"}),
                no_update, no_update, no_update, no_update,
                {"display": "none"}, True
            )
        
        else:
            # Unknown state
            return (
                {"width": "0%", "height": "20px", "backgroundColor": "#ffc107", "borderRadius": "4px", "textAlign": "center", "color": "black", "lineHeight": "20px"},
                "Unknown",
                f"Unknown state: {flow_run.state.type}",
                no_update, no_update, no_update, no_update,
                {"display": "block", "marginBottom": "15px"}, False
            )
            
    except Exception as e:
        return (
            {"width": "0%", "height": "20px", "backgroundColor": "#dc3545", "borderRadius": "4px", "textAlign": "center", "color": "white", "lineHeight": "20px"},
            "Error",
            html.Div([
                "❌ Error checking flow status: ",
                html.Code(str(e))
            ], style={"color": "red"}),
            no_update, no_update, no_update, no_update,
            {"display": "none"}, True
        )