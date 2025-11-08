"""
UMAP Prefect callbacks for SCULPT
Save as: sculpt/callbacks/umap_prefect_callbacks.py
"""

import os
import threading
import time
import uuid

import plotly.express as px
from dash import ALL, Input, Output, State, callback, ctx, html, no_update

from sculpt.utils.prefect import (
    get_flow_run_result,
    get_flow_run_state,
    schedule_prefect_flow,
)

# Detect Docker environment
IS_DOCKER = os.environ.get("DOCKER_CONTAINER") == "true"

# Store for running flows
FLOW_RESULTS = {}


def run_flow_async(flow_id, **kwargs):
    """Run the Prefect flow asynchronously"""
    try:
        import os

        import prefect

        # Force Prefect to use the server API
        os.environ["PREFECT_API_URL"] = "http://prefect-server:4200/api"

        print("=" * 60)
        print(f"🚀 Starting UMAP flow {flow_id}")
        print(f"Environment: {'Docker' if IS_DOCKER else 'Local'}")
        print(f"Prefect version: {prefect.__version__}")
        print(f"API URL: {os.environ['PREFECT_API_URL']}")
        print("=" * 60)

        flow_run_id = schedule_prefect_flow(
            deployment_name="SCULPT UMAP Analysis/umap_analysis_flow",
            parameters=kwargs,
            flow_run_name=f"UMAP Flow {flow_id}",
            tags=["sculpt", "umap"],
        )

        flow_ended = False
        while not flow_ended:
            state = get_flow_run_state(flow_run_id)
            if state.name in ["COMPLETED", "FAILED", "CANCELLED", "CRASHED"]:
                flow_ended = True
            print(f"⏳ Flow {flow_run_id} is still running with {state}...")
            time.sleep(1)

        if state.name == "COMPLETED":
            result = get_flow_run_result(flow_run_id)
            print(f"✅ Flow completed with result: {result}")
        elif state.name == "FAILED":
            print("❌ Flow failed")
        elif state.name == "CANCELLED":
            print("🚫 Flow was cancelled")
        elif state.name == "CRASHED":
            print("💥 Flow crashed unexpectedly")
        # Run the flow - this will now show in Prefect UI
        # result = tracked_umap_flow(**kwargs)

        print("=" * 60)
        print(f"✅ Flow {flow_id} completed successfully!")
        print("Check http://localhost:4200 for details")
        print("=" * 60)

        FLOW_RESULTS[flow_id] = {"status": "completed", "result": result}

    except Exception as e:
        print(f"❌ Flow {flow_id} failed with error: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        FLOW_RESULTS[flow_id] = {"status": "failed", "error": str(e)}


@callback(
    Output("umap-graph", "figure", allow_duplicate=True),
    Output("debug-output", "children", allow_duplicate=True),
    Output("combined-data-store", "data", allow_duplicate=True),
    Output("umap-quality-metrics", "children", allow_duplicate=True),
    Output("umap-flow-run-store", "data", allow_duplicate=True),
    Input("run-umap", "n_clicks"),
    Input("umap-progress-interval", "n_intervals"),
    State("stored-files", "data"),
    State("umap-file-selector", "value"),
    State("num-neighbors", "value"),
    State("min-dist", "value"),
    State("sample-frac", "value"),
    State({"type": "feature-selector-graph1", "category": ALL}, "value"),
    State("umap-flow-run-store", "data"),
    prevent_initial_call=True,
)
def run_umap_analysis(
    n_clicks,
    n_intervals,
    stored_files,
    selected_ids,
    num_neighbors,
    min_dist,
    sample_frac,
    selected_features_list,
    flow_status,
):
    """Run UMAP analysis with Prefect"""

    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update

    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    # Start new flow when button clicked
    if triggered == "run-umap":
        env_label = "🐳 Docker" if IS_DOCKER else "💻 Local"
        print(f"{env_label} UMAP button clicked!")
        print(f"Files available: {len(stored_files) if stored_files else 0}")
        print(f"Selected IDs: {selected_ids}")
        print(f"Selected IDs type: {type(selected_ids)}, content: {selected_ids}")

        if not stored_files or not selected_ids:
            return (
                {},
                html.Div(
                    "Please upload and select files first", style={"color": "orange"}
                ),
                no_update,
                no_update,
                no_update,
            )

        # FIX: The flow expects List[Dict], not a dictionary
        # Only include the selected files, maintaining the list structure
        processed_files = []
        for file_info in stored_files:
            file_id = file_info.get("id")
            # Convert selected_ids to integers if they're strings
            selected_ids_int = (
                [int(sid) for sid in selected_ids] if selected_ids else []
            )

            if file_id in selected_ids_int:
                # Keep the data in the format the flow expects
                processed_file = {
                    "id": file_id,
                    "filename": file_info.get("filename", "Unknown"),
                    "data": file_info.get("data"),  # Keep original data format
                    "is_selection": file_info.get("is_selection", False),
                }
                # Add any other fields from the original file_info that might be needed
                for key in file_info:
                    if key not in processed_file:
                        processed_file[key] = file_info[key]

                processed_files.append(processed_file)

        print(
            f"Processing {len(processed_files)} selected files from {len(stored_files)} total files"
        )

        # Flatten selected features
        features = []
        for feature_list in selected_features_list:
            if feature_list:
                features.extend(feature_list)

        print(
            f"Selected features: {features[:5]}..."
            if len(features) > 5
            else f"Selected features: {features}"
        )

        # Generate unique flow ID
        flow_id = str(uuid.uuid4())[:8]

        # FIX: Pass the list of files, not a dictionary
        # Also ensure selected_ids are integers
        params = {
            "stored_files": processed_files,  # Pass the list of selected files
            "selected_ids": selected_ids_int,  # Pass as list of integers
            "num_neighbors": num_neighbors or 15,
            "min_dist": min_dist or 0.1,
            "sample_frac": sample_frac or 1.0,
            "selected_features_list": [features] if features else [[]],
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 5,
        }

        print("Running with parameters:")
        print(f"  - stored_files: {len(processed_files)} files (as list)")
        print(f"  - selected_ids: {selected_ids_int}")
        print(f"  - num_neighbors: {params['num_neighbors']}")
        print(f"  - min_dist: {params['min_dist']}")
        print(f"  - sample_frac: {params['sample_frac']}")
        print(f"  - selected_features_list: {len(features)} features")

        # Initialize status
        FLOW_RESULTS[flow_id] = {"status": "running"}

        # Run flow in background thread
        thread = threading.Thread(target=run_flow_async, args=(flow_id,), kwargs=params)
        thread.daemon = True
        thread.start()

        # Return initial status
        empty_fig = {
            "data": [],
            "layout": {
                "title": "UMAP Analysis Running...",
                "annotations": [
                    {
                        "text": f"Processing with Prefect...<br>Flow ID: {flow_id}<br>Check http://localhost:4200",
                        "showarrow": False,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                    }
                ],
            },
        }

        return (
            empty_fig,
            html.Div(
                f"UMAP analysis started ({env_label})...", style={"color": "blue"}
            ),
            no_update,
            html.Div("Processing..."),
            {"flow_id": flow_id, "status": "running"},
        )

    # Check progress on interval
    elif triggered == "umap-progress-interval" and flow_status:
        flow_id = flow_status.get("flow_id")
        if not flow_id or flow_id not in FLOW_RESULTS:
            return no_update, no_update, no_update, no_update, no_update

        result = FLOW_RESULTS[flow_id]

        if result["status"] == "completed":
            flow_result = result["result"]

            # Clean up
            del FLOW_RESULTS[flow_id]

            if not flow_result or not flow_result.get("success"):
                error_msg = (
                    flow_result.get("error", "Unknown error")
                    if flow_result
                    else "No results"
                )
                return (
                    {},
                    html.Div(f"Analysis failed: {error_msg}", style={"color": "red"}),
                    no_update,
                    html.Div("Failed"),
                    {"flow_id": None, "status": "failed"},
                )

            # Extract results
            umap_df = flow_result["umap_df"]
            metadata = flow_result.get("metadata", {})

            # Create visualization
            fig = px.scatter(
                umap_df,
                x="UMAP1",
                y="UMAP2",
                color="cluster",
                hover_data=["file_label"],
                title=f"UMAP Visualization ({metadata.get('n_samples', 0)} points, {metadata.get('n_clusters', 0)} clusters)",
            )

            fig.update_layout(height=600, template="plotly_dark", hovermode="closest")

            # Prepare metrics
            metrics_html = html.Div(
                [
                    html.H6("UMAP Analysis Metrics"),
                    html.P(f"Total Samples: {metadata.get('n_samples', 0)}"),
                    html.P(f"Features Used: {metadata.get('n_features', 0)}"),
                    html.P(f"Clusters Found: {metadata.get('n_clusters', 0)}"),
                    html.P(
                        f"Computation Time: {metadata.get('computation_time', 0):.2f}s"
                    ),
                ]
            )

            # Success message
            success_msg = html.Div(
                [
                    html.Span("✅ UMAP analysis completed! ", style={"color": "green"}),
                    html.Span(f"({metadata.get('computation_time', 0):.2f}s)"),
                ]
            )

            return (
                fig,
                success_msg,
                umap_df.to_dict("records"),
                metrics_html,
                {"flow_id": None, "status": "completed"},
            )

        elif result["status"] == "failed":
            # Clean up
            del FLOW_RESULTS[flow_id]

            error_msg = result.get("error", "Unknown error")
            return (
                {},
                html.Div(f"Analysis failed: {error_msg}", style={"color": "red"}),
                no_update,
                html.Div("Failed"),
                {"flow_id": None, "status": "failed"},
            )

        # Still running
        return no_update, no_update, no_update, no_update, no_update

    return no_update, no_update, no_update, no_update, no_update


# Create an alias for monitor_umap_flow_progress since app.py imports it
monitor_umap_flow_progress = run_umap_analysis
