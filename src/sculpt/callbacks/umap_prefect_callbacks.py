"""
UMAP Prefect callbacks for SCULPT
"""

import os
import uuid

import plotly.express as px
from dash import ALL, Input, Output, State, callback, ctx, html, no_update

from sculpt.utils.prefect import (
    get_flow_run_result,
    get_flow_run_state,
    schedule_prefect_flow,
)

IS_DOCKER = os.environ.get("DOCKER_CONTAINER") == "true"


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
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update

    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    env_label = "🐳 Docker" if IS_DOCKER else "💻 Local"

    # ------------------------------------------------------------------ #
    # Button clicked — schedule the flow and return immediately            #
    # ------------------------------------------------------------------ #
    if triggered == "run-umap":
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

        selected_ids_int = [int(sid) for sid in selected_ids] if selected_ids else []
        processed_files = []
        for file_info in stored_files:
            if file_info.get("id") in selected_ids_int:
                processed_files.append(
                    {
                        "id": file_info.get("id"),
                        "filename": file_info.get("filename", "Unknown"),
                        "data": file_info.get("data"),
                        "is_selection": file_info.get("is_selection", False),
                        **{
                            k: v
                            for k, v in file_info.items()
                            if k not in ("id", "filename", "data", "is_selection")
                        },
                    }
                )

        features = [f for fl in selected_features_list if fl for f in fl]

        params = {
            # TODO: Remove data from prefect parameters and instead calculate the features in the prefect flow
            "stored_files": processed_files,
            "selected_ids": selected_ids_int,
            "num_neighbors": num_neighbors or 15,
            "min_dist": min_dist or 0.1,
            "sample_frac": sample_frac or 1.0,
            "selected_features_list": [features] if features else [[]],
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 5,
        }

        flow_run_id = schedule_prefect_flow(
            deployment_name="SCULPT UMAP Analysis/umap_analysis_flow",
            parameters=params,
            flow_run_name=f"UMAP Flow {str(uuid.uuid4())[:8]}",
            tags=["sculpt", "umap"],
        )

        print(f"🚀 Scheduled flow run: {flow_run_id}")

        empty_fig = {
            "data": [],
            "layout": {
                "title": "UMAP Analysis Running...",
                "annotations": [
                    {
                        "text": f"Processing with Prefect...<br>Flow run: {flow_run_id}",
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
            {"flow_run_id": flow_run_id, "status": "running"},
        )

    # ------------------------------------------------------------------ #
    # Interval tick — check state directly from Prefect                   #
    # ------------------------------------------------------------------ #
    elif triggered == "umap-progress-interval" and flow_status:
        flow_run_id = flow_status.get("flow_run_id")
        if not flow_run_id or flow_status.get("status") not in ("running", "pending"):
            return no_update, no_update, no_update, no_update, no_update

        state = get_flow_run_state(flow_run_id)
        print(f"⏳ Flow {flow_run_id} state: {state.name}")

        if state.name not in ("COMPLETED", "FAILED", "CANCELLED", "CRASHED"):
            # Still running — update status label but nothing else
            return (
                no_update,
                html.Div(f"Running... (state: {state.name})", style={"color": "blue"}),
                no_update,
                no_update,
                {"flow_run_id": flow_run_id, "status": "running"},
            )

        if state.name == "COMPLETED":
            try:
                result = get_flow_run_result(flow_run_id)
                print(f"Result type: {type(result)}")
                print(
                    f"Result keys: {result.keys() if isinstance(result, dict) else 'NOT A DICT'}"
                )
            except Exception as e:
                return (
                    {},
                    html.Div(f"Failed to retrieve result: {e}", style={"color": "red"}),
                    no_update,
                    html.Div("Failed"),
                    {"flow_run_id": None, "status": "failed"},
                )

            if not isinstance(result, dict) or not result.get("success"):
                error_msg = (
                    result.get("error", "Unknown error")
                    if isinstance(result, dict)
                    else str(result)
                )
                return (
                    {},
                    html.Div(f"Analysis failed: {error_msg}", style={"color": "red"}),
                    no_update,
                    html.Div("Failed"),
                    {"flow_run_id": None, "status": "failed"},
                )

            umap_df = result["umap_df"]
            metadata = result.get("metadata", {})

            fig = px.scatter(
                umap_df,
                x="UMAP1",
                y="UMAP2",
                color="cluster",
                hover_data=["file_label"],
                title=f"UMAP ({metadata.get('n_samples', 0)} points, {metadata.get('n_clusters', 0)} clusters)",
            )
            fig.update_layout(height=600, template="plotly_dark", hovermode="closest")

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

            return (
                fig,
                html.Div(
                    [
                        html.Span(
                            "✅ UMAP analysis completed! ", style={"color": "green"}
                        ),
                        html.Span(f"({metadata.get('computation_time', 0):.2f}s)"),
                    ]
                ),
                umap_df.to_dict("records"),
                metrics_html,
                {"flow_run_id": None, "status": "completed"},
            )

        # FAILED / CANCELLED / CRASHED
        return (
            {},
            html.Div(f"Flow {state.name.lower()}", style={"color": "red"}),
            no_update,
            html.Div(state.name.capitalize()),
            {"flow_run_id": None, "status": "failed"},
        )

    return no_update, no_update, no_update, no_update, no_update


monitor_umap_flow_progress = run_umap_analysis
