"""
Docker-compatible UMAP Prefect callbacks using Prefect 3.x patterns.
Save this as: sculpt/callbacks/umap_prefect_callbacks_docker.py
"""

import os
import json
import threading
import uuid
from dash import callback, Input, Output, State, html, ALL, no_update, ctx
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd

# Docker-aware Prefect API URL
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")

# Store for running flows
FLOW_RESULTS = {}

def run_flow_in_docker(**kwargs):
    """Run flow in Docker environment using Prefect 3.x"""
    try:
        # Import inside function to avoid circular imports
        from sculpt.flows.umap_flow import umap_analysis_flow
        
        # Generate unique flow ID
        flow_id = str(uuid.uuid4())
        
        print(f"🚀 Starting UMAP flow {flow_id} in Docker environment")
        print(f"Using Prefect API: {PREFECT_API_URL}")
        
        # Run the flow directly (Prefect 3.x handles orchestration)
        # This will submit to the Prefect server in Docker
        result = umap_analysis_flow(**kwargs)
        
        print(f"✅ Flow {flow_id} completed successfully!")
        
        FLOW_RESULTS[flow_id] = {
            "status": "completed",
            "result": result
        }
        
        return flow_id
        
    except Exception as e:
        print(f"❌ Error in flow: {e}")
        flow_id = str(uuid.uuid4())
        FLOW_RESULTS[flow_id] = {
            "status": "failed",
            "error": str(e)
        }
        return flow_id


@callback(
    [Output('umap-graph', 'figure'),
     Output('debug-output', 'children'),
     Output('combined-data-store', 'data'),
     Output('umap-quality-metrics', 'children'),
     Output('umap-flow-status', 'data')],
    Input('run-umap', 'n_clicks'),
    [State('stored-files', 'data'),
     State('umap-file-selector', 'value'),
     State('num-neighbors', 'value'),
     State('min-dist', 'value'),
     State('sample-frac', 'value'),
     State({'type': 'feature-selector-graph1', 'category': ALL}, 'value')],
    prevent_initial_call=True
)
def run_umap_docker(n_clicks, stored_files, selected_ids, n_neighbors, 
                    min_dist, sample_frac, selected_features_list):
    """Docker-compatible UMAP callback - starts the flow"""
    
    if not n_clicks or not stored_files or not selected_ids:
        raise PreventUpdate
    
    print(f"🔵 UMAP button clicked in Docker environment!")
    print(f"Files available: {len(stored_files)}")
    print(f"Selected files: {selected_ids}")
    
    # Flatten selected features
    features = []
    for feature_list in selected_features_list:
        if feature_list:
            features.extend(feature_list)
    
    if not features:
        return (
            {},
            html.Div("⚠️ Please select at least one feature", style={"color": "orange"}),
            no_update,
            html.Div("No features selected"),
            {"status": "error"}
        )
    
    # Prepare parameters
    params = {
        "stored_files": stored_files,
        "selected_files": selected_ids,
        "selected_features": features,
        "n_neighbors": n_neighbors or 15,
        "min_dist": min_dist or 0.1,
        "sample_frac": sample_frac or 1.0,
        "enable_clustering": True
    }
    
    print(f"Running with parameters: {params}")
    
    # Generate flow ID for tracking
    flow_id = str(uuid.uuid4())
    
    # Run flow in background thread
    thread = threading.Thread(
        target=lambda: FLOW_RESULTS.update({
            flow_id: {"status": "running", "result": None}
        }) or run_flow_in_docker(**params)
    )
    thread.daemon = True
    thread.start()
    
    # Store flow ID in the status
    return (
        {},  # Empty figure initially
        html.Div("🔄 UMAP analysis started in Docker environment...", style={"color": "blue"}),
        no_update,
        html.Div("Processing..."),
        {"status": "running", "flow_id": flow_id}
    )


@callback(
    [Output('umap-graph', 'figure', allow_duplicate=True),
     Output('debug-output', 'children', allow_duplicate=True),
     Output('combined-data-store', 'data', allow_duplicate=True),
     Output('umap-quality-metrics', 'children', allow_duplicate=True),
     Output('umap-flow-status', 'data', allow_duplicate=True)],
    Input('umap-progress-interval', 'n_intervals'),
    State('umap-flow-status', 'data'),
    prevent_initial_call=True
)
def check_docker_flow_status(n_intervals, flow_status):
    """Check status of flows running in Docker"""
    
    if not flow_status or flow_status.get("status") != "running":
        raise PreventUpdate
    
    flow_id = flow_status.get("flow_id")
    if not flow_id or flow_id not in FLOW_RESULTS:
        raise PreventUpdate
    
    result = FLOW_RESULTS[flow_id]
    
    if result["status"] == "running":
        # Still running, don't update
        raise PreventUpdate
    
    elif result["status"] == "completed":
        # Process successful result
        flow_result = result["result"]
        
        if not flow_result or not flow_result.get("success"):
            return (
                {},
                html.Div("❌ Flow completed but no valid results", style={"color": "red"}),
                no_update,
                html.Div("No results"),
                {"status": "failed"}
            )
        
        # Create UMAP visualization
        umap_df = flow_result.get("umap_df")
        
        if umap_df is not None and not umap_df.empty:
            # Create the plot
            fig = px.scatter(
                umap_df,
                x="UMAP1",
                y="UMAP2",
                color="cluster" if "cluster" in umap_df.columns else None,
                title="UMAP Analysis Results (Docker)",
                hover_data=umap_df.columns.tolist()
            )
            
            fig.update_layout(
                height=600,
                hovermode='closest',
                showlegend=True
            )
            
            # Prepare metrics
            metadata = flow_result.get("metadata", {})
            metrics_content = html.Div([
                html.H6("Analysis Metrics"),
                html.P(f"Samples processed: {metadata.get('n_samples', 'N/A')}"),
                html.P(f"Features used: {metadata.get('n_features', 'N/A')}"),
                html.P(f"Clusters found: {metadata.get('n_clusters', 'N/A')}"),
                html.P(f"Computation time: {metadata.get('computation_time', 'N/A'):.2f}s" 
                       if metadata.get('computation_time') else "Time: N/A")
            ])
            
            # Clear the result from memory
            del FLOW_RESULTS[flow_id]
            
            return (
                fig,
                html.Div("✅ UMAP analysis completed successfully!", style={"color": "green"}),
                flow_result.get("combined_df", {}).to_dict('records') if hasattr(flow_result.get("combined_df", {}), 'to_dict') else {},
                metrics_content,
                {"status": "completed", "flow_id": None}
            )
        else:
            return (
                {},
                html.Div("⚠️ Flow completed but UMAP dataframe is empty", style={"color": "orange"}),
                no_update,
                html.Div("No data to display"),
                {"status": "warning"}
            )
    
    elif result["status"] == "failed":
        error = result.get("error", "Unknown error")
        
        # Clear the result
        del FLOW_RESULTS[flow_id]
        
        return (
            {},
            html.Div(f"❌ Error: {error}", style={"color": "red"}),
            no_update,
            html.Div("Analysis failed - check logs"),
            {"status": "failed", "error": str(error)}
        )
    
    # Default: don't update
    raise PreventUpdate