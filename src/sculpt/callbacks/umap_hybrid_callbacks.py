"""
Complete Hybrid UMAP Callback with FIXED duplicate output handling
Save this as: sculpt/callbacks/umap_hybrid_callbacks.py
"""

import os
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, html, no_update
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Set Prefect API URL
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# Import Prefect flow
from sculpt.flows.umap_flow import umap_analysis_flow

# Import utilities
from sculpt.utils.metrics.clustering_quality import (
    cluster_stability,
    hopkins_statistic,
    physics_cluster_consistency,
)
from sculpt.utils.metrics.confidence_assessment import (
    calculate_adaptive_confidence_score,
)
from sculpt.utils.metrics.physics_features import (
    calculate_physics_features,
    calculate_physics_features_flexible,
    calculate_physics_features_with_profile,
    has_physics_features,
)
from sculpt.utils.ui import create_smart_confidence_ui


@callback(
    Output("umap-graph", "figure", allow_duplicate=True),  # FIXED: Added allow_duplicate here too!
    Output("debug-output", "children", allow_duplicate=True),  # FIXED: Added allow_duplicate
    Output("combined-data-store", "data", allow_duplicate=True),  # FIXED: Added allow_duplicate
    Output("umap-quality-metrics", "children", allow_duplicate=True),  # FIXED: Added allow_duplicate
    # Progress UI outputs (if they exist)
    Output("umap-status-message", "children", allow_duplicate=True),  # For progress message
    Output("umap-progress-card", "style", allow_duplicate=True),  # For progress card visibility
    Input("run-umap", "n_clicks"),
    State("stored-files", "data"),
    State("umap-file-selector", "value"),
    State("num-neighbors", "value"),
    State("min-dist", "value"),
    State("sample-frac", "value"),
    State({"type": "feature-selector-graph1", "category": ALL}, "value"),
    State("metric-selector", "value"),
    State("point-opacity", "value"),
    State("color-mode", "value"),
    State("visualization-type", "value"),
    State("heatmap-bandwidth", "value"),
    State("heatmap-colorscale", "value"),
    State("show-points-overlay", "value"),
    State("file-config-assignments-store", "data"),
    State("configuration-profiles-store", "data"),
    prevent_initial_call=True,
)
def update_umap_hybrid(
    n_clicks,
    stored_files,
    selected_ids,
    num_neighbors,
    min_dist,
    sample_frac,
    selected_features_list,
    selected_metrics,
    point_opacity,
    color_mode,
    visualization_type,
    heatmap_bandwidth,
    heatmap_colorscale,
    show_points_overlay,
    assignments_store,
    profiles_store,
):
    """
    Complete Hybrid UMAP: Includes ALL original features + Prefect integration
    This will show up in Prefect UI while maintaining all functionality
    FIXED: Added allow_duplicate=True to prevent conflicts
    """
    
    # Default returns for progress UI
    empty_progress_msg = ""
    hidden_progress_card = {"display": "none"}
    
    if not stored_files:
        return {}, "No files uploaded.", {}, [html.Div("No files uploaded.")], empty_progress_msg, hidden_progress_card

    if not selected_ids:
        return {}, "No files selected for UMAP.", {}, [html.Div("No files selected.")], empty_progress_msg, hidden_progress_card

    try:
        # Show processing message
        processing_msg = html.Div([
            html.Span("🔄 ", style={"color": "blue"}),
            html.Span("Processing UMAP with Prefect...", style={"color": "blue"})
        ])
        show_progress_card = {"display": "block", "marginBottom": "15px"}
        
        print("=" * 60)
        print(f"Running UMAP with Prefect (will show in Prefect UI)")
        print(f"Configuration: neighbors={num_neighbors}, min_dist={min_dist}")
        print(f"Visualization: type={visualization_type}, color={color_mode}")
        print("=" * 60)
        
        # Step 1: Pre-process data with physics features (IMPORTANT!)
        # This happens BEFORE sending to Prefect because the flow expects preprocessed data
        processed_files = {}
        debug_messages = []
        
        for file_id in selected_ids:
            file_data = stored_files[file_id]
            df = pd.DataFrame(file_data["data"])
            df["file_label"] = file_data["filename"]
            
            # Check if this is a selection file
            is_selection = file_data.get("is_selection", False)
            
            if is_selection:
                # Selection files already have UMAP coordinates
                debug_messages.append(f"{file_data['filename']}: Selection file with {len(df)} events")
                processed_files[file_id] = {
                    "filename": file_data["filename"],
                    "data": df.to_dict('records'),
                    "is_selection": True
                }
            else:
                # Regular COLTRIMS file - apply physics features if configured
                if not has_physics_features(df):
                    # Get profile assignment for this file
                    profile_name = assignments_store.get(file_data["filename"]) if assignments_store else None
                    
                    if profile_name and profile_name != "none" and profiles_store and profile_name in profiles_store:
                        # Calculate with assigned profile
                        profile_config = profiles_store[profile_name]
                        try:
                            df = calculate_physics_features_with_profile(df, profile_config)
                            debug_messages.append(f"Applied profile '{profile_name}' to {file_data['filename']}")
                        except Exception as e:
                            debug_messages.append(f"Error applying profile to {file_data['filename']}: {str(e)}")
                            # Fallback to flexible calculation
                            try:
                                df = calculate_physics_features_flexible(df, None)
                            except:
                                debug_messages.append(f"Could not calculate physics features for {file_data['filename']}")
                    else:
                        # No profile assigned, use flexible calculation
                        try:
                            df = calculate_physics_features_flexible(df, None)
                            if assignments_store:
                                debug_messages.append(f"No profile for {file_data['filename']}, using default")
                        except:
                            debug_messages.append(f"Could not calculate physics features for {file_data['filename']}")
                
                # Store processed data
                processed_files[file_id] = {
                    "filename": file_data["filename"],
                    "data": df.to_dict('records'),
                    "is_selection": False
                }
        
        # Step 2: Run the Prefect flow with processed data
        print("Submitting to Prefect flow...")
        result = umap_analysis_flow.with_options(
            name=f"umap-dash-n{num_neighbors}-d{min_dist}-{visualization_type}"
        ).run(
            stored_files=processed_files,
            selected_ids=selected_ids,
            num_neighbors=num_neighbors,
            min_dist=min_dist,
            sample_frac=sample_frac,
            selected_features_list=selected_features_list,
            dbscan_eps=0.5,
            dbscan_min_samples=5
        )
        
        if not result or not result.get('success'):
            error_msg = html.Div([
                html.Span("❌ ", style={"color": "red"}),
                html.Span("UMAP flow failed", style={"color": "red"})
            ])
            return {}, "UMAP flow failed", {}, [html.Div("Flow execution failed")], error_msg, hidden_progress_card
        
        # Step 3: Extract results from Prefect flow
        umap_df = result['umap_df']
        combined_df = result['combined_df']
        cluster_labels = result['cluster_labels']
        clustering_info = result['clustering_info']
        flow_debug = result.get('debug_messages', [])
        metadata = result.get('metadata', {})
        
        # Combine debug messages
        all_debug = debug_messages + flow_debug
        all_debug.append(f"✅ Prefect flow completed in {metadata.get('computation_time', 0):.2f}s")
        all_debug.append(f"📊 Check Prefect UI at http://127.0.0.1:4200")
        debug_str = "<br>".join(all_debug)
        
        # Step 4: Create visualization based on type and color mode
        
        # Handle cluster coloring
        if color_mode == "cluster":
            # Convert cluster labels to strings for better legend
            umap_df["cluster"] = [f"Cluster {x}" if x != -1 else "Noise" for x in cluster_labels]
            color_column = "cluster"
        else:
            color_column = "file_label"
        
        # Create figure based on visualization type
        if visualization_type == "heatmap":
            # Create density heatmap
            umap_data = umap_df[["UMAP1", "UMAP2"]].to_numpy()
            
            # Create the grid for the heatmap
            x_min, x_max = umap_data[:, 0].min() - 0.5, umap_data[:, 0].max() + 0.5
            y_min, y_max = umap_data[:, 1].min() - 0.5, umap_data[:, 1].max() + 0.5
            
            # Create a meshgrid
            grid_size = 200
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            xx, yy = np.meshgrid(x_grid, y_grid)
            grid_points = np.column_stack([xx.flatten(), yy.flatten()])
            
            # Compute KDE
            kde = gaussian_kde(umap_data.T, bw_method=heatmap_bandwidth)
            densities = kde(grid_points.T).reshape(grid_size, grid_size)
            
            fig = go.Figure()
            
            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    x=x_grid,
                    y=y_grid,
                    z=densities,
                    colorscale=heatmap_colorscale,
                    showscale=True,
                    colorbar=dict(title="Density"),
                    hoverinfo="none",
                )
            )
            
            # Optionally overlay scatter points
            if show_points_overlay == "yes":
                # Create color map for files
                unique_files = umap_df["file_label"].unique()
                colors = px.colors.qualitative.Plotly
                color_map = {file: colors[i % len(colors)] for i, file in enumerate(unique_files)}
                
                for label in unique_files:
                    mask = umap_df["file_label"] == label
                    df_subset = umap_df[mask]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_subset["UMAP1"],
                            y=df_subset["UMAP2"],
                            mode="markers",
                            name=label,
                            marker=dict(
                                size=4,
                                color=color_map[label],
                                opacity=0.5,
                                line=dict(width=0)
                            ),
                            hovertemplate="UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>%{text}",
                            text=[label] * len(df_subset),
                            showlegend=True
                        )
                    )
            
            title = f"UMAP Density Heatmap (n={len(umap_df)}, neighbors={num_neighbors}, min_dist={min_dist})"
            fig.update_layout(title=title)
            
        else:  # scatter plot
            fig = px.scatter(
                umap_df,
                x="UMAP1",
                y="UMAP2",
                color=color_column,
                title=f"UMAP via Prefect (n={len(umap_df)}, neighbors={num_neighbors}, min_dist={min_dist})",
                labels={color_column: "File" if color_mode == "file" else "Cluster"},
                opacity=point_opacity if point_opacity else 0.7,
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            hovermode='closest',
            dragmode='lasso'  # Enable lasso selection
        )
        
        # Step 5: Store data for other callbacks
        combined_data_json = {
            "combined_df": combined_df.to_json(date_format="iso", orient="split"),
            "umap_coords": umap_df.to_json(date_format="iso", orient="split"),
            "cluster_labels": cluster_labels,
            "feature_cols": result['feature_cols'],
            "clustering_info": clustering_info,
        }
        
        # Step 6: Calculate quality metrics if requested
        metrics_children = []
        if selected_metrics and len(selected_metrics) > 0:
            try:
                metrics = {}
                n_clusters = clustering_info['n_clusters']
                n_noise = clustering_info['n_noise']
                
                # Get UMAP embedding for metrics
                X_umap = umap_df[["UMAP1", "UMAP2"]].to_numpy()
                X_umap_scaled = StandardScaler().fit_transform(X_umap)
                
                # Only calculate metrics for valid clusters
                if n_clusters > 1:
                    mask = np.array(cluster_labels) != -1
                    if np.sum(mask) > 1:
                        if "silhouette" in selected_metrics:
                            metrics["silhouette"] = silhouette_score(
                                X_umap_scaled[mask], np.array(cluster_labels)[mask]
                            )
                        
                        if "davies_bouldin" in selected_metrics:
                            metrics["davies_bouldin"] = davies_bouldin_score(
                                X_umap_scaled[mask], np.array(cluster_labels)[mask]
                            )
                        
                        if "calinski_harabasz" in selected_metrics:
                            metrics["calinski_harabasz"] = calinski_harabasz_score(
                                X_umap_scaled[mask], np.array(cluster_labels)[mask]
                            )
                
                if "hopkins" in selected_metrics:
                    metrics["hopkins"] = hopkins_statistic(X_umap_scaled)
                
                if "stability" in selected_metrics:
                    metrics["stability"] = cluster_stability(
                        X_umap_scaled, 0.5, 5, n_iterations=3
                    )
                
                if "physics_consistency" in selected_metrics and combined_df is not None:
                    physics_metrics = physics_cluster_consistency(
                        combined_df, cluster_labels
                    )
                    metrics.update(physics_metrics)
                
                # Add cluster info to metrics
                metrics["n_clusters"] = n_clusters
                metrics["n_noise"] = n_noise
                metrics["noise_ratio"] = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
                
                # Calculate confidence score
                confidence_data = calculate_adaptive_confidence_score(
                    metrics, clustering_method="dbscan"
                )
                metrics_children = [create_smart_confidence_ui(confidence_data)]
                
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                traceback.print_exc()
                metrics_children = [html.Div(f"Metrics error: {str(e)}")]
        else:
            metrics_children = [html.Div("Enable metrics in settings to see quality assessment")]
        
        # Success message
        success_msg = html.Div([
            html.Span("✅ ", style={"color": "green"}),
            html.Span("UMAP analysis completed!", style={"color": "green"})
        ])
        
        return fig, debug_str, combined_data_json, metrics_children, success_msg, hidden_progress_card
    
    except Exception as e:
        print(f"Error in hybrid UMAP: {e}")
        traceback.print_exc()
        error_msg = f"Error: {str(e)}"
        error_div = html.Div([
            html.Span("❌ ", style={"color": "red"}),
            html.Span(error_msg, style={"color": "red"})
        ])
        return (
            {},
            error_msg,
            {},
            [html.Div(error_msg)],
            error_div,
            hidden_progress_card
        )