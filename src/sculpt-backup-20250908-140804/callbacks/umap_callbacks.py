import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap.umap_ as umap
from dash import ALL, Input, Output, State, callback, html
from matplotlib.path import Path
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

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


# Callback for Graph 1: Original UMAP Embedding with selected features
@callback(
    Output("umap-graph", "figure"),
    Output("debug-output", "children"),
    Output("combined-data-store", "data"),
    Output("umap-quality-metrics", "children"),
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
def update_umap(
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
    """Compute UMAP embedding on selected files using selected features."""
    if not stored_files:
        return {}, "No files uploaded.", {}, [html.Div("No files uploaded.")]

    if not selected_ids:
        return {}, "No files selected for UMAP.", {}, [html.Div("No files selected.")]

    try:
        # Collect all selected features
        all_selected_features = []
        for features in selected_features_list:
            if features:  # Only add non-empty lists
                all_selected_features.extend(features)

        sampled_dfs = []
        debug_str = ""
        selection_dfs = []  # Separate list for selection files to handle differently

        # Process each selected file
        for f in stored_files:
            if f["id"] in selected_ids:
                try:
                    df = pd.read_json(f["data"], orient="split")
                    is_selection = f.get("is_selection", False)

                    if is_selection:
                        # This is a saved selection file
                        debug_str += f"{f['filename']}: Selection file with {len(df)} events.<br>"

                        # Make sure it has required columns for visualization
                        if (
                            "UMAP1" in df.columns
                            and "UMAP2" in df.columns
                            and "file_label" in df.columns
                        ):
                            selection_dfs.append(df)
                        else:
                            debug_str += f"Warning: Selection file {f['filename']} is missing required columns.<br>"
                    else:
                        # Regular COLTRIMS file - apply configuration-aware physics features
                        df["file_label"] = f["filename"]  # Add file name as a label

                        # Check if physics features already exist, if not calculate them
                        if not has_physics_features(df):
                            # Get profile assignment for this file
                            profile_name = (
                                assignments_store.get(f["filename"])
                                if assignments_store
                                else None
                            )

                            if (
                                profile_name
                                and profile_name != "none"
                                and profiles_store
                                and profile_name in profiles_store
                            ):
                                # Calculate with assigned profile
                                profile_config = profiles_store[profile_name]
                                try:
                                    df = calculate_physics_features_with_profile(
                                        df, profile_config
                                    )
                                    debug_str += f"Applied profile '{profile_name}' to {f['filename']}<br>"
                                except Exception as e:
                                    debug_str += f"Error applying profile to {f['filename']}: {str(e)}<br>"
                                    # Fallback to flexible calculation
                                    df = calculate_physics_features_flexible(df, None)
                            else:
                                # No profile assigned, use flexible calculation
                                df = calculate_physics_features_flexible(df, None)
                                if assignments_store:
                                    debug_str += f"No profile assigned for {f['filename']}, using default calculation<br>"

                        # Sample the data
                        sample_size = int(len(df) * sample_frac)
                        if sample_size > 0 and sample_size < len(df):
                            df = df.sample(n=sample_size, random_state=42).reset_index(
                                drop=True
                            )

                        sampled_dfs.append(df)
                        debug_str += (
                            f"{f['filename']}: {len(df)} events after sampling.<br>"
                        )
                except Exception as e:
                    debug_str += f"Error processing {f['filename']}: {str(e)}.<br>"

        # Process regular COLTRIMS files (if any)
        combined_df = None
        umap_df = pd.DataFrame(columns=["UMAP1", "UMAP2", "file_label"])

        if len(sampled_dfs) > 0:
            # Combine all selected datasets
            combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(
                drop=True
            )
            debug_str += f"Combined data shape: {combined_df.shape}.<br>"

            # Use selected features for UMAP
            if all_selected_features and len(all_selected_features) > 0:
                feature_cols = [
                    col for col in combined_df.columns if col in all_selected_features
                ]
                if feature_cols:
                    debug_str += f"Using selected features for UMAP: {', '.join(feature_cols)}<br>"
                    X = combined_df[feature_cols].to_numpy()
                else:
                    # Fallback to original momentum columns
                    original_cols = [
                        col
                        for col in combined_df.columns
                        if col.startswith("particle_")
                    ]
                    X = combined_df[original_cols].to_numpy()
                    debug_str += "No valid features selected, using original momentum components.<br>"
            else:
                # Use original momentum columns
                original_cols = [
                    col for col in combined_df.columns if col.startswith("particle_")
                ]
                X = combined_df[original_cols].to_numpy()
                debug_str += (
                    "No features selected, using original momentum components.<br>"
                )

            # Run UMAP
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=int(num_neighbors),
                min_dist=float(min_dist),
                metric="euclidean",
                random_state=42,
            )

            # Handle NaN/inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Fit UMAP
            umap_data = reducer.fit_transform(X)

            # Create DataFrame for visualization
            umap_df = pd.DataFrame(umap_data, columns=["UMAP1", "UMAP2"])
            umap_df["file_label"] = combined_df["file_label"]
        else:
            combined_df = pd.DataFrame()

        # Add any selection files directly to the visualization
        for sel_df in selection_dfs:
            # Add the selection data to the UMAP visualization data
            # Use only the necessary columns for visualization
            selection_viz_df = sel_df[["UMAP1", "UMAP2", "file_label"]].copy()

            # Append to the UMAP dataframe
            umap_df = pd.concat([umap_df, selection_viz_df], ignore_index=True)

        # Calculate clustering for DBSCAN coloring (do this regardless of color mode)

        # Get UMAP coordinates for clustering
        X_umap = umap_df[["UMAP1", "UMAP2"]].to_numpy()

        # Scale the data for better DBSCAN performance
        scaler = StandardScaler()
        X_umap_scaled = scaler.fit_transform(X_umap)

        # Find a reasonable epsilon
        eps_candidates = np.linspace(0.1, 1.0, 10)
        best_eps = 0.5  # Default
        max_clusters = 0

        # Try different eps values and pick the one that gives a reasonable number of clusters
        for eps in eps_candidates:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_umap_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # We want to maximize the number of clusters but avoid too many noise points
            noise_count = np.sum(labels == -1)
            noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0

            # Good balance: enough clusters but not too many noise points
            if n_clusters >= 2 and noise_ratio < 0.5 and n_clusters > max_clusters:
                max_clusters = n_clusters
                best_eps = eps

        # Run DBSCAN with the best eps
        dbscan = DBSCAN(eps=best_eps, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_umap_scaled)

        # Create color mappings for both file and cluster modes
        # Color mapping for file labels
        unique_labels = umap_df["file_label"].unique()
        colorscale = px.colors.qualitative.Plotly  # Use Plotly's default colorscale
        color_map = {
            label: colorscale[i % len(colorscale)]
            for i, label in enumerate(unique_labels)
        }

        # Color mapping for clusters
        # Special handling for noise points (-1 label)
        unique_clusters = sorted(set(cluster_labels))
        if -1 in unique_clusters:
            # Move noise to the end
            unique_clusters.remove(-1)
            unique_clusters.append(-1)

        # Use a colorscale that works well for clusters
        if len(unique_clusters) <= 10:
            cluster_colorscale = px.colors.qualitative.D3  # Good for distinct clusters
        else:
            # For many clusters, use a continuous colorscale
            cluster_colorscale = px.colors.sequential.Viridis

        # Create color mapping for clusters
        cluster_colors = {}
        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:  # Noise points
                cluster_colors[cluster] = "rgba(150,150,150,0.5)"  # Gray for noise
            else:
                # Regular clusters
                if len(unique_clusters) - (1 if -1 in unique_clusters else 0) <= 10:
                    colorscale_idx = i % len(cluster_colorscale)
                    cluster_colors[cluster] = cluster_colorscale[colorscale_idx]
                else:
                    # For many clusters, distribute colors evenly
                    n_real_clusters = len(unique_clusters) - (
                        1 if -1 in unique_clusters else 0
                    )
                    idx = i / (n_real_clusters - 1) if n_real_clusters > 1 else 0
                    idx = min(0.99, max(0, idx))  # Ensure it's between 0 and 1
                    color_idx = int(idx * (len(cluster_colorscale) - 1))
                    cluster_colors[cluster] = cluster_colorscale[color_idx]

        # Initialize our figure

        # Heatmap visualization
        if visualization_type == "heatmap":

            fig = go.Figure()

            # Get UMAP coordinates
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

            # Compute KDE (Kernel Density Estimation)
            kde = gaussian_kde(umap_data.T, bw_method=heatmap_bandwidth)
            densities = kde(grid_points.T).reshape(grid_size, grid_size)

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

            # Optionally, overlay scatter points with reduced opacity for context
            if show_points_overlay == "yes":
                if color_mode == "file":
                    # Color by file source with reduced opacity
                    for label in umap_df["file_label"].unique():
                        mask = umap_df["file_label"] == label
                        df_subset = umap_df[mask]

                        fig.add_trace(
                            go.Scatter(
                                x=df_subset["UMAP1"],
                                y=df_subset["UMAP2"],
                                mode="markers",
                                marker=dict(
                                    size=4,  # Smaller points
                                    color=color_map[label],
                                    opacity=0.3,  # Reduced opacity
                                    line=dict(width=0),
                                ),
                                name=f"{label} ({len(df_subset)} pts)",
                            )
                        )
                elif color_mode == "cluster":
                    # Color by DBSCAN cluster with reduced opacity
                    for cluster in unique_clusters:
                        mask = cluster_labels == cluster
                        cluster_points = umap_df.iloc[mask]

                        # For noise points, make them smaller and more transparent
                        marker_size = 3 if cluster == -1 else 4
                        marker_opacity = 0.2 if cluster == -1 else 0.3

                        fig.add_trace(
                            go.Scatter(
                                x=cluster_points["UMAP1"],
                                y=cluster_points["UMAP2"],
                                mode="markers",
                                marker=dict(
                                    size=marker_size,
                                    color=cluster_colors[cluster],
                                    opacity=marker_opacity,
                                    line=dict(width=0),
                                ),
                                name=f"Cluster {cluster if cluster != -1 else 'Noise'} ({len(cluster_points)} pts)",
                            )
                        )
        else:
            # Original scatter plot visualization
            fig = go.Figure()

            # Visualization based on color mode
            if color_mode == "file":
                # Color by file source (original coloring)
                # Add traces for each file label
                for label in unique_labels:
                    mask = umap_df["file_label"] == label
                    df_subset = umap_df[mask]

                    fig.add_trace(
                        go.Scatter(
                            x=df_subset["UMAP1"],
                            y=df_subset["UMAP2"],
                            mode="markers",
                            marker=dict(
                                size=7,
                                color=color_map[label],
                                opacity=point_opacity,
                                line=dict(width=0),
                            ),
                            name=f"{label} ({len(df_subset)} pts)",
                        )
                    )

            elif color_mode == "cluster":
                # Color by DBSCAN cluster
                # Add points for each cluster
                for cluster in unique_clusters:
                    mask = cluster_labels == cluster

                    # Get points for this cluster
                    cluster_points = umap_df.iloc[mask]

                    # For noise points, make them smaller and more transparent
                    marker_size = 5 if cluster == -1 else 7
                    marker_opacity = (
                        point_opacity * 0.7 if cluster == -1 else point_opacity
                    )

                    # Add trace for this cluster
                    fig.add_trace(
                        go.Scatter(
                            x=cluster_points["UMAP1"],
                            y=cluster_points["UMAP2"],
                            mode="markers",
                            marker=dict(
                                size=marker_size,
                                color=cluster_colors[cluster],
                                opacity=marker_opacity,
                                line=dict(width=0),
                            ),
                            name=f"Cluster {cluster if cluster != -1 else 'Noise'} ({len(cluster_points)} pts)",
                        )
                    )

        # Update figure properties
        title_suffix = ""
        if color_mode == "cluster":
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = np.sum(cluster_labels == -1)
            title_suffix = f" - {n_clusters} clusters detected"

        # Adjust legend position based on number of traces
        legend_y_position = (
            -0.5 if len(fig.data) > 12 else -0.4 if len(fig.data) > 8 else -0.3
        )

        # Create legend configuration
        legend_config = dict(
            orientation="h",
            yanchor="top",  # Anchor to the top of the legend box
            y=legend_y_position,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",  # More opaque background for readability
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10),  # Smaller font for many items
            itemwidth=30,  # Smaller item width for more compact, richer color symbols
            itemsizing="constant",
            tracegroupgap=5,  # Reduced gap between legend groups
        )

        # Adjust figure height based on legend size - more space for many clusters
        figure_height = 600
        if len(fig.data) > 15:
            figure_height = 750
        elif len(fig.data) > 10:
            figure_height = 700
        elif len(fig.data) > 6:
            figure_height = 650

        # Apply the layout settings
        if visualization_type == "heatmap":
            title = f"UMAP Density Heatmap (n_neighbors={num_neighbors}, min_dist={min_dist}){title_suffix}"
        else:
            title = f"UMAP Embedding (n_neighbors={num_neighbors}, min_dist={min_dist}){title_suffix}"

        fig.update_layout(
            height=figure_height,
            title=title,
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title=f"{'Clusters' if color_mode == 'cluster' else 'Data File'}",
            dragmode="lasso",
            legend=legend_config,
            modebar=dict(
                add=["lasso2d", "select2d"],
                remove=["pan2d", "autoScale2d"],
                orientation="h",
                bgcolor="rgba(255,255,255,0.9)",
                color="rgba(68,68,68,1)",
                activecolor="rgba(254,95,85,1)",
            ),
            margin=dict(l=50, r=50, t=50, b=100),  # Increased bottom margin for legend
        )

        # Store data for other callbacks
        combined_data_json = {
            "combined_df": (
                combined_df.to_json(date_format="iso", orient="split")
                if not combined_df.empty
                else "{}"
            ),
            "umap_coords": umap_df.to_json(date_format="iso", orient="split"),
            "selected_features_graph1": all_selected_features,
            "cluster_labels": (
                cluster_labels.tolist() if len(cluster_labels) > 0 else []
            ),
        }

        # Calculate clustering metrics WITH SMART CONFIDENCE
        metrics_children = []
        if (
            not umap_df.empty
            and "UMAP1" in umap_df.columns
            and "UMAP2" in umap_df.columns
        ):
            try:
                # Get UMAP coordinates for clustering
                X_umap = umap_df[["UMAP1", "UMAP2"]].to_numpy()

                # Scale the data for better DBSCAN performance
                scaler = StandardScaler()
                X_umap_scaled = scaler.fit_transform(X_umap)

                # Find optimal DBSCAN parameters
                eps_candidates = np.linspace(0.1, 1.0, 10)
                best_eps = 0.5
                max_clusters = 0

                for eps in eps_candidates:
                    dbscan = DBSCAN(eps=eps, min_samples=5)
                    labels = dbscan.fit_predict(X_umap_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_count = np.sum(labels == -1)
                    noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0

                    if (
                        n_clusters >= 2
                        and noise_ratio < 0.5
                        and n_clusters > max_clusters
                    ):
                        max_clusters = n_clusters
                        best_eps = eps

                # Run DBSCAN with best parameters
                dbscan = DBSCAN(eps=best_eps, min_samples=5)
                cluster_labels = dbscan.fit_predict(X_umap_scaled)

                # Collect metrics for confidence calculation
                metrics = {}
                unique_clusters = set(cluster_labels)
                n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                n_noise = np.sum(cluster_labels == -1)
                noise_ratio = (
                    n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
                )

                metrics["noise_ratio"] = noise_ratio

                # Calculate selected metrics
                if n_clusters >= 2:
                    mask = cluster_labels != -1
                    non_noise_points = np.sum(mask)
                    non_noise_clusters = len(set(cluster_labels[mask]))

                    if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                        if "silhouette" in selected_metrics:
                            metrics["silhouette"] = silhouette_score(
                                X_umap_scaled[mask], cluster_labels[mask]
                            )

                        if "davies_bouldin" in selected_metrics:
                            metrics["davies_bouldin"] = davies_bouldin_score(
                                X_umap_scaled[mask], cluster_labels[mask]
                            )

                        if "calinski_harabasz" in selected_metrics:
                            metrics["calinski_harabasz"] = calinski_harabasz_score(
                                X_umap_scaled[mask], cluster_labels[mask]
                            )

                        if "hopkins" in selected_metrics:
                            metrics["hopkins"] = hopkins_statistic(X_umap_scaled)

                        if "stability" in selected_metrics:
                            metrics["stability"] = cluster_stability(
                                X_umap_scaled, best_eps, 5, n_iterations=3
                            )

                        if (
                            "physics_consistency" in selected_metrics
                            and combined_df is not None
                            and not combined_df.empty
                        ):
                            physics_metrics = physics_cluster_consistency(
                                combined_df, cluster_labels
                            )
                            metrics.update(physics_metrics)

                # SMART CONFIDENCE CALCULATION
                confidence_data = calculate_adaptive_confidence_score(
                    metrics, clustering_method="dbscan"
                )

                # Create the smart confidence UI
                metrics_children = [create_smart_confidence_ui(confidence_data)]

            except Exception as e:
                traceback.format_exc()
                metrics_children = [html.Div(f"Error calculating confidence: {str(e)}")]
        else:
            metrics_children = [html.Div("Run UMAP to see reliability assessment")]

        return fig, debug_str, combined_data_json, metrics_children

    except Exception as e:
        print(f"Error in update_umap: {e}")

        traceback.print_exc()
        return (
            {},
            f"Error computing UMAP: {str(e)}",
            {},
            [html.Div(f"Error computing UMAP: {str(e)}")],
        )


# Callback for Graph 2: Selected Points (Stored Coordinates)
@callback(
    Output("umap-graph-selected-only", "figure"),
    Output("debug-output-selected-only", "children"),
    Input("show-selected", "n_clicks"),
    State("selected-points-store", "data"),  # Use stored selection data
    State("combined-data-store", "data"),
    State("umap-graph", "figure"),  # Add original figure as input
    prevent_initial_call=True,
)
def update_umap_selected_only(
    n_clicks, selectedData, combined_data_json, original_figure
):
    """Display the selected points from Graph 1."""
    try:
        if not combined_data_json:
            return {}, "No UMAP data available. Run UMAP first."

        # Load the UMAP coordinates
        umap_df_all = pd.read_json(
            combined_data_json["umap_coords"], orient="split"
        ).reset_index(drop=True)

        if not selectedData:
            return (
                {},
                "No points selected. Use the lasso or box select tool on Graph 1.",
            )

        # Initialize variables to hold indices
        indices = []
        debug_text = ""

        # Handle box selection
        if "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]
            debug_text = f"Box selection: x: [{x_range[0]:.2f}, {x_range[1]:.2f}], y: [{y_range[0]:.2f}, {y_range[1]:.2f}]<br>"

            selected_mask = (
                (umap_df_all["UMAP1"] >= x_range[0])
                & (umap_df_all["UMAP1"] <= x_range[1])
                & (umap_df_all["UMAP2"] >= y_range[0])
                & (umap_df_all["UMAP2"] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            debug_text = "Lasso selection<br>"
            # Instead of trying to extract indices from points, we'll use the coordinates
            # from the lassoPoints and find the points within the polygon

            # Extract lasso polygon coordinates
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

            # Check which points are within the lasso path
            points_array = np.column_stack([umap_df_all["UMAP1"], umap_df_all["UMAP2"]])
            inside_lasso = lasso_path.contains_points(points_array)

            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()

        # Handle direct point selection (fallback)
        elif "points" in selectedData:
            indices = [pt["pointIndex"] for pt in selectedData["points"]]
            debug_text = f"Direct point selection: {len(indices)} points selected<br>"

        if not indices:
            return {}, "No valid selection or no points found in selection area."

        # Extract the selected points
        selected_umap_df = umap_df_all.iloc[indices].reset_index(drop=True)

        # Extract color information from original figure

        # Create a color map from the original figure
        color_map = {}
        if original_figure and "data" in original_figure:
            for trace in original_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    color_map[trace["name"]] = trace["marker"]["color"]

        # Create figure with consistent colors
        fig = go.Figure()

        # Create a temporary plotly express figure to get the layout setting

        temp_fig = px.scatter(
            selected_umap_df,
            x="UMAP1",
            y="UMAP2",
            color="file_label",
            title=f"Selected Points ({len(indices)} events)",
            labels={"file_label": "Data File"},
            opacity=0.7,
        )

        # Use the layout from temp_fig
        fig.update_layout(temp_fig.layout)

        # Add traces for each file label
        for label in selected_umap_df["file_label"].unique():
            mask = selected_umap_df["file_label"] == label
            df_subset = selected_umap_df[mask]

            marker_color = color_map.get(label, None)  # Get color from map or None

            if marker_color:
                # Use color from original figure
                fig.add_trace(
                    go.Scatter(
                        x=df_subset["UMAP1"],
                        y=df_subset["UMAP2"],
                        mode="markers",
                        marker=dict(size=8, color=marker_color, opacity=0.7),
                        name=label,
                    )
                )
            else:
                # Fallback to auto-assigned color
                fig.add_trace(
                    go.Scatter(
                        x=df_subset["UMAP1"],
                        y=df_subset["UMAP2"],
                        mode="markers",
                        marker=dict(size=8, opacity=0.7),
                        name=label,
                    )
                )

        fig.update_layout(height=600)

        # Count points by file
        file_counts = selected_umap_df["file_label"].value_counts().to_dict()
        count_str = "<br>".join(
            [f"{file}: {count} events" for file, count in file_counts.items()]
        )
        debug_text += (
            f"<br>Total selected: {len(selected_umap_df)} events<br>{count_str}"
        )

        return fig, debug_text

    except Exception as e:
        print(f"Error in Graph 2 callback: {e}")

        traceback.print_exc()
        return {}, f"Error processing selection: {str(e)}"


# Modified to store the subset data used for Graph 3
@callback(
    Output("umap-graph-selected-run", "figure"),
    Output("debug-output-selected-run", "children"),
    Output("combined-data-store", "data", allow_duplicate=True),
    Output("umap-quality-metrics-graph3", "children"),
    Input("run-umap-selected-run", "n_clicks"),
    State("selected-points-store", "data"),
    State("combined-data-store", "data"),
    State("num-neighbors-selected-run", "value"),
    State("min-dist-selected-run", "value"),
    State({"type": "feature-selector-graph3", "category": ALL}, "value"),
    State("umap-graph", "figure"),
    State("metric-selector-graph3", "value"),
    prevent_initial_call=True,
)
def update_umap_selected_run(
    n_clicks,
    selectedData,
    combined_data_json,
    num_neighbors_sel,
    min_dist_sel,
    selected_features_list_graph3,
    original_figure,
    selected_metrics,
):
    """Re-run UMAP on only the selected points from Graph 1 using Graph 3's feature selection."""
    try:
        # Initialize default return values
        empty_fig = {}
        default_data = combined_data_json or {}
        empty_metrics = []

        if not combined_data_json:
            return (
                empty_fig,
                "No UMAP data available. Run UMAP first.",
                default_data,
                empty_metrics,
            )

        # Load the combined dataframe
        combined_df = pd.read_json(
            combined_data_json["combined_df"], orient="split"
        ).reset_index(drop=True)

        # Get features selected for Graph 3
        all_selected_features_graph3 = []
        for features in selected_features_list_graph3:
            if features:  # Only add non-empty lists
                all_selected_features_graph3.extend(features)

        if not selectedData:
            return (
                empty_fig,
                "No points selected. Use the lasso or box select tool on Graph 1.",
                default_data,
                empty_metrics,
            )

        # Initialize variables to hold indices
        indices = []
        debug_text = ""

        # Load the UMAP coordinates for reference when finding points
        umap_coords = pd.read_json(
            combined_data_json["umap_coords"], orient="split"
        ).reset_index(drop=True)

        # Handle box selection
        if "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]
            debug_text = f"Box selection for re-run: x range: [{x_range[0]:.2f}, {x_range[1]:.2f}], y range: "
            "[{y_range[0]:.2f}, {y_range[1]:.2f}]<br>"

            selected_mask = (
                (umap_coords["UMAP1"] >= x_range[0])
                & (umap_coords["UMAP1"] <= x_range[1])
                & (umap_coords["UMAP2"] >= y_range[0])
                & (umap_coords["UMAP2"] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            debug_text = "Lasso selection for re-run<br>"

            # Extract lasso polygon coordinates
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

            # Check which points are within the lasso path
            points_array = np.column_stack([umap_coords["UMAP1"], umap_coords["UMAP2"]])
            inside_lasso = lasso_path.contains_points(points_array)

            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()

        # Handle direct point selection (fallback)
        elif "points" in selectedData:
            indices = [pt["pointIndex"] for pt in selectedData["points"]]
            debug_text = (
                f"Direct point selection for re-run: {len(indices)} points selected<br>"
            )

        if not indices:
            return (
                empty_fig,
                "No valid selection or no points found in selection area.",
                default_data,
                empty_metrics,
            )

        # Extract selected points from the combined dataframe
        selected_df = combined_df.iloc[indices].reset_index(drop=True)
        debug_text += f"Selected data shape: {selected_df.shape}<br>"

        # IMPORTANT: Store the selected subset for Graph 3 in the combined_data_json
        # This is the key change to make the Graph 3 re-run work properly
        combined_data_json["graph3_subset"] = selected_df.to_json(
            date_format="iso", orient="split"
        )
        combined_data_json["graph3_indices"] = (
            indices  # Store the original indices as well
        )

        # Use selected features for UMAP if available (from Graph 3 selection)
        if all_selected_features_graph3 and len(all_selected_features_graph3) > 0:
            feature_cols = [
                col
                for col in selected_df.columns
                if col in all_selected_features_graph3
            ]
            if feature_cols:
                debug_text += f"Using selected features for UMAP re-run: {', '.join(feature_cols)}<br>"
                X_selected = selected_df[feature_cols].to_numpy()
            else:
                # Fallback to original momentum columns
                original_cols = [
                    col for col in selected_df.columns if col.startswith("particle_")
                ]
                X_selected = selected_df[original_cols].to_numpy()
                debug_text += "No valid features selected for re-run, using original momentum components.<br>"
        else:
            # Use original momentum columns
            original_cols = [
                col for col in selected_df.columns if col.startswith("particle_")
            ]
            X_selected = selected_df[original_cols].to_numpy()
            debug_text += "No features selected for re-run, using original momentum components.<br>"

        # Count points by file
        file_counts = selected_df["file_label"].value_counts().to_dict()
        count_str = "<br>".join(
            [f"{file}: {count} events" for file, count in file_counts.items()]
        )
        debug_text += f"<br>{count_str}<br>"

        # Re-run UMAP on the selected subset
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(num_neighbors_sel),
            min_dist=float(min_dist_sel),
            metric="euclidean",
            random_state=42,
        )

        # Handle NaN/inf values
        X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit UMAP
        umap_data_sel = reducer.fit_transform(X_selected)

        # Create DataFrame for visualization
        umap_df_sel = pd.DataFrame(umap_data_sel, columns=["UMAP1", "UMAP2"])
        umap_df_sel["file_label"] = selected_df["file_label"]

        # Also store the Graph 3 UMAP coordinates
        combined_data_json["graph3_umap_coords"] = umap_df_sel.to_json(
            date_format="iso", orient="split"
        )

        # Create the new figure with original colors
        # Extract color information from original figure

        # Create a color map from the original figure
        color_map = {}
        if original_figure and "data" in original_figure:
            for trace in original_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    color_map[trace["name"]] = trace["marker"]["color"]

        # Create figure with consistent colors
        fig = go.Figure()

        # Create a temporary plotly express figure to get the layout settings

        temp_fig = px.scatter(
            umap_df_sel,
            x="UMAP1",
            y="UMAP2",
            color="file_label",
            title=f"Re-run UMAP on Selected Points (n_neighbors={num_neighbors_sel}, min_dist={min_dist_sel})",
            labels={"file_label": "Data File"},
            opacity=0.7,
        )

        # Use the layout from temp_fig
        fig.update_layout(temp_fig.layout)

        # Add traces for each file label
        for label in umap_df_sel["file_label"].unique():
            mask = umap_df_sel["file_label"] == label
            df_subset = umap_df_sel[mask]

            marker_color = color_map.get(label, None)  # Get color from map or None

            if marker_color:
                # Use color from original figure
                fig.add_trace(
                    go.Scatter(
                        x=df_subset["UMAP1"],
                        y=df_subset["UMAP2"],
                        mode="markers",
                        marker=dict(size=8, color=marker_color, opacity=0.7),
                        name=label,
                    )
                )
            else:
                # Fallback to auto-assigned color
                fig.add_trace(
                    go.Scatter(
                        x=df_subset["UMAP1"],
                        y=df_subset["UMAP2"],
                        mode="markers",
                        marker=dict(size=8, opacity=0.7),
                        name=label,
                    )
                )

        fig.update_layout(height=600)

        debug_text += f"Re-run UMAP completed on {len(umap_df_sel)} events with colors preserved from original."

        # Calculate clustering metrics WITH SMART CONFIDENCE
        metrics_children = []
        try:

            # Get UMAP coordinates for clustering
            X_umap = umap_df_sel[["UMAP1", "UMAP2"]].to_numpy()

            # Scale the data for better DBSCAN performance
            scaler = StandardScaler()
            X_umap_scaled = scaler.fit_transform(X_umap)

            # Find a reasonable epsilon for DBSCAN
            eps_candidates = np.linspace(0.1, 1.0, 10)
            best_eps = 0.5  # Default
            max_clusters = 0

            # Try different eps values and pick the one that gives a reasonable number of clusters
            for eps in eps_candidates:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(X_umap_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                noise_count = np.sum(labels == -1)
                noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0

                if n_clusters >= 2 and noise_ratio < 0.5 and n_clusters > max_clusters:
                    max_clusters = n_clusters
                    best_eps = eps

            # Run DBSCAN with the best eps
            dbscan = DBSCAN(eps=best_eps, min_samples=5)
            cluster_labels = dbscan.fit_predict(X_umap_scaled)

            # Collect metrics for confidence calculation
            metrics = {}
            unique_clusters = set(cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = np.sum(cluster_labels == -1)
            noise_ratio = (
                n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            )

            metrics["noise_ratio"] = noise_ratio

            # Only calculate metrics if we have at least 2 clusters
            if n_clusters >= 2:
                # For metrics, we need to exclude noise points (-1)
                mask = cluster_labels != -1
                non_noise_points = np.sum(mask)
                non_noise_clusters = len(set(cluster_labels[mask]))

                if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                    if "silhouette" in selected_metrics:
                        metrics["silhouette"] = silhouette_score(
                            X_umap_scaled[mask], cluster_labels[mask]
                        )

                    if "davies_bouldin" in selected_metrics:
                        metrics["davies_bouldin"] = davies_bouldin_score(
                            X_umap_scaled[mask], cluster_labels[mask]
                        )

                    if "calinski_harabasz" in selected_metrics:
                        metrics["calinski_harabasz"] = calinski_harabasz_score(
                            X_umap_scaled[mask], cluster_labels[mask]
                        )

                    # Add new metrics based on selection
                    if "hopkins" in selected_metrics:
                        h_stat = hopkins_statistic(X_umap_scaled)
                        metrics["hopkins"] = h_stat

                    if "stability" in selected_metrics:
                        stability = cluster_stability(
                            X_umap_scaled, best_eps, 5, n_iterations=3
                        )
                        metrics["stability"] = stability

                    # Add physics consistency if selected
                    if "physics_consistency" in selected_metrics:
                        # Match the cluster labels to the original dataset
                        physics_metrics = physics_cluster_consistency(
                            selected_df, cluster_labels
                        )
                        metrics.update(physics_metrics)

            # SMART CONFIDENCE CALCULATION
            confidence_data = calculate_adaptive_confidence_score(
                metrics, clustering_method="dbscan"
            )

            # Create the smart confidence UI
            metrics_children = [create_smart_confidence_ui(confidence_data)]

        except Exception as e:

            trace = traceback.format_exc()
            metrics_children = [html.Div(f"Error calculating confidence: {str(e)}")]

        return fig, debug_text, combined_data_json, metrics_children

    except Exception as e:
        print(f"Error in Graph 3 callback: {e}")

        traceback.print_exc()
        error_msg = f"Error re-running UMAP: {str(e)}"
        return {}, error_msg, combined_data_json or {}, []


# Complete callback for "New Graph: UMAP Re-run on Graph 3 Selected Points"
@callback(
    Output("umap-graph-graph3-selection", "figure"),
    Output("debug-output-graph3-selection", "children"),
    Output("graph3-selection-umap-store", "data"),
    Output("umap-quality-metrics-graph3-selection", "children"),
    Input("run-umap-graph3-selection", "n_clicks"),
    State("selected-points-run-store", "data"),  # Selected points from Graph 3
    State("umap-graph-selected-run", "figure"),  # Graph 3 figure data
    State("num-neighbors-graph3-selection", "value"),
    State("min-dist-graph3-selection", "value"),
    State({"type": "feature-selector-graph3-selection", "category": ALL}, "value"),
    State("combined-data-store", "data"),
    State("metric-selector-graph3-selection", "value"),
    prevent_initial_call=True,
)
def update_umap_graph3_selection(
    n_clicks,
    graph3_selection,
    graph3_figure,
    num_neighbors,
    min_dist,
    selected_features_list,
    combined_data_json,
    selected_metrics,
):
    """Run UMAP on the selected points from Graph 3, using the same approach as Graph 1."""
    try:
        # Initialize default return values
        empty_fig = {}
        empty_store = []
        empty_metrics = []
        debug_text = ""

        # Validate inputs
        if not graph3_selection:
            return (
                empty_fig,
                "No selection data found for Graph 3. Use the lasso or box select tool.",
                empty_store,
                empty_metrics,
            )

        debug_text += "Processing selection from Graph 3.<br>"

        # Check if graph3_subset and graph3_umap_coords are available
        if (
            "graph3_subset" not in combined_data_json
            or combined_data_json["graph3_subset"] == "{}"
        ):
            debug_text += "Graph 3 subset not found in data store.<br>"
            return (
                empty_fig,
                "Graph 3 subset data not found. Please re-run Graph 3 first.",
                empty_store,
                empty_metrics,
            )

        if (
            "graph3_umap_coords" not in combined_data_json
            or combined_data_json["graph3_umap_coords"] == "{}"
        ):
            debug_text += "Graph 3 UMAP coordinates not found.<br>"
            return (
                empty_fig,
                "Graph 3 UMAP coordinates not found. Please re-run Graph 3 first.",
                empty_store,
                empty_metrics,
            )

        # Load the Graph 3 subset data and UMAP coordinates
        graph3_subset_df = pd.read_json(
            combined_data_json["graph3_subset"], orient="split"
        )
        graph3_umap_coords = pd.read_json(
            combined_data_json["graph3_umap_coords"], orient="split"
        )

        debug_text += f"Found Graph 3 subset with {len(graph3_subset_df)} rows.<br>"
        debug_text += (
            f"Found Graph 3 UMAP coordinates with {len(graph3_umap_coords)} points.<br>"
        )

        # Process the selection using geometric operations - similar to Graph 1
        indices = []

        # Handle box selection
        if "range" in graph3_selection:
            x_range = graph3_selection["range"]["x"]
            y_range = graph3_selection["range"]["y"]
            debug_text += f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]<br>"

            # Find points inside the box
            selected_mask = (
                (graph3_umap_coords["UMAP1"] >= x_range[0])
                & (graph3_umap_coords["UMAP1"] <= x_range[1])
                & (graph3_umap_coords["UMAP2"] >= y_range[0])
                & (graph3_umap_coords["UMAP2"] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()
            debug_text += f"Found {len(indices)} points in box selection.<br>"

        # Handle lasso selection
        elif "lassoPoints" in graph3_selection:
            debug_text += "Lasso selection detected.<br>"

            # Extract lasso polygon coordinates
            lasso_x = graph3_selection["lassoPoints"]["x"]
            lasso_y = graph3_selection["lassoPoints"]["y"]

            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

            # Check which points are within the lasso path
            points_array = np.column_stack(
                [graph3_umap_coords["UMAP1"], graph3_umap_coords["UMAP2"]]
            )
            inside_lasso = lasso_path.contains_points(points_array)

            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()
            debug_text += f"Found {len(indices)} points in lasso selection.<br>"

        # As a fallback, try to use points directly if available
        elif "points" in graph3_selection and graph3_selection["points"]:
            debug_text += "Direct point selection detected.<br>"

            # Try to map points to indices more intelligently
            all_points = graph3_selection["points"]
            debug_text += f"Selection contains {len(all_points)} points.<br>"

            # Here we'll use curve-based indexing with a bit of validation
            for point in all_points:
                curve_num = point.get("curveNumber", -1)
                point_idx = point.get("pointIndex", -1)

                # Only proceed if we have valid values
                if curve_num >= 0 and point_idx >= 0:
                    # Find the corresponding index in our coords dataframe
                    # This is tricky because pointIndex is relative to each curve
                    # We'll need to reconstruct the mapping

                    # First get the label for this curve
                    curve_label = None
                    if (
                        graph3_figure
                        and "data" in graph3_figure
                        and curve_num < len(graph3_figure["data"])
                    ):
                        trace = graph3_figure["data"][curve_num]
                        if "name" in trace:
                            curve_label = trace["name"]
                            if " (" in curve_label:
                                curve_label = curve_label.split(" (")[0]

                    if curve_label is not None:
                        # Find matching rows in our coords dataframe
                        matching_rows = graph3_umap_coords[
                            graph3_umap_coords["file_label"] == curve_label
                        ]

                        # Verify point_idx is valid for this subset
                        if 0 <= point_idx < len(matching_rows):
                            # Get the actual index in the full dataframe
                            actual_idx = matching_rows.iloc[point_idx].name
                            indices.append(actual_idx)
                        else:
                            debug_text += f"Warning: pointIndex {point_idx} out of range for curve {curve_num} with label "
                            "{curve_label}.<br>"

            debug_text += f"Mapped {len(indices)} points from direct selection.<br>"

        if not indices:
            return (
                empty_fig,
                "No valid points found in the selection region.",
                empty_store,
                empty_metrics,
            )

        # Get counts by label before extraction
        label_counts_before = (
            graph3_umap_coords.iloc[indices]["file_label"].value_counts().to_dict()
        )
        debug_text += "Label distribution in selection:<br>"
        for label, count in sorted(label_counts_before.items()):
            debug_text += f"- {label}: {count} points<br>"

        # Verify indices are valid
        valid_indices = [i for i in indices if 0 <= i < len(graph3_subset_df)]
        if len(valid_indices) != len(indices):
            debug_text += f"Warning: {len(indices) - len(valid_indices)} invalid indices were removed.<br>"
            indices = valid_indices

        if not indices:
            return (
                empty_fig,
                "No valid indices found in the selection.",
                empty_store,
                empty_metrics,
            )

        # Extract the subset of data for selected points
        selected_df = graph3_subset_df.iloc[indices].copy()
        debug_text += f"Created dataframe with {len(selected_df)} rows.<br>"

        # Store the original labels to ensure consistency
        original_labels = selected_df["file_label"].values

        # Collect selected features for UMAP
        all_selected_features = []
        for features in selected_features_list:
            if features:  # Only add non-empty lists
                all_selected_features.extend(features)

        # Use selected features for UMAP if available
        if all_selected_features:
            feature_cols = [
                col
                for col in selected_df.columns
                if col in all_selected_features and col != "file_label"
            ]
            if feature_cols:
                debug_text += (
                    f"Using {len(feature_cols)} selected features for UMAP.<br>"
                )
                X = selected_df[feature_cols].to_numpy()
            else:
                # Fallback to momentum columns
                momentum_cols = [
                    col
                    for col in selected_df.columns
                    if col.startswith("particle_") and col != "file_label"
                ]
                debug_text += f"No valid selected features. Using {len(momentum_cols)} momentum columns.<br>"
                X = selected_df[momentum_cols].to_numpy()
        else:
            # Use momentum columns
            momentum_cols = [
                col
                for col in selected_df.columns
                if col.startswith("particle_") and col != "file_label"
            ]
            debug_text += f"No features selected. Using {len(momentum_cols)} momentum columns.<br>"
            X = selected_df[momentum_cols].to_numpy()

        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Run UMAP
        try:
            debug_text += "Running UMAP...<br>"
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(
                    int(num_neighbors), len(X) - 1
                ),  # Ensure n_neighbors is valid
                min_dist=float(min_dist),
                metric="euclidean",
                random_state=42,
            )

            # Fit UMAP
            umap_result = reducer.fit_transform(X)
            debug_text += "UMAP transformation completed successfully.<br>"
        except Exception as e:
            debug_text += f"Error running UMAP: {str(e)}<br>"
            return (
                empty_fig,
                f"Error running UMAP: {str(e)}<br>Debug info: {debug_text}",
                empty_store,
                empty_metrics,
            )

        # Create DataFrame for visualization with original labels
        result_df = pd.DataFrame(
            {
                "UMAP1": umap_result[:, 0],
                "UMAP2": umap_result[:, 1],
                "file_label": original_labels,  # Use original labels to maintain correspondence
                "original_index": indices,
            }
        )

        # Extract color information from Graph 3 figure
        color_map = {}
        if graph3_figure and "data" in graph3_figure:
            for trace in graph3_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    # Clean the label if it contains point count
                    clean_name = trace["name"]
                    if " (" in clean_name:
                        clean_name = clean_name.split(" (")[0]
                    color_map[clean_name] = trace["marker"]["color"]

        # Create figure with the same colors as Graph 3

        fig = go.Figure()

        # Create one trace per label for clean visualization
        for label in result_df["file_label"].unique():
            mask = result_df["file_label"] == label
            df_subset = result_df[mask]

            # Get color from Graph 3 or use default
            color = color_map.get(label, None)

            # Add trace for this label
            fig.add_trace(
                go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode="markers",
                    marker=dict(size=8, color=color, opacity=0.7),
                    name=f"{label} ({len(df_subset)} pts)",
                )
            )

        # Update figure properties
        fig.update_layout(
            height=600,
            title=f"UMAP on Graph 3 Selected Points (n_neighbors={num_neighbors}, min_dist={min_dist})",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data File",
        )

        # Store the result for potential future use
        graph3_selection_umap_store = {
            "umap_coords": result_df.to_json(date_format="iso", orient="split"),
            "feature_data": selected_df.to_json(date_format="iso", orient="split"),
        }

        # Calculate clustering metrics WITH SMART CONFIDENCE
        metrics_children = []
        try:

            # Get UMAP coordinates for clustering
            X_umap = result_df[["UMAP1", "UMAP2"]].to_numpy()

            # Scale the data for better DBSCAN performance
            scaler = StandardScaler()
            X_umap_scaled = scaler.fit_transform(X_umap)

            # Find a reasonable epsilon for DBSCAN
            eps_candidates = np.linspace(0.1, 1.0, 10)
            best_eps = 0.5  # Default
            max_clusters = 0

            # Try different eps values and pick the one that gives a reasonable number of clusters
            for eps in eps_candidates:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(X_umap_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                noise_count = np.sum(labels == -1)
                noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0

                if n_clusters >= 2 and noise_ratio < 0.5 and n_clusters > max_clusters:
                    max_clusters = n_clusters
                    best_eps = eps

            # Run DBSCAN with the best eps
            dbscan = DBSCAN(eps=best_eps, min_samples=5)
            cluster_labels = dbscan.fit_predict(X_umap_scaled)

            # Collect metrics for confidence calculation
            metrics = {}
            unique_clusters = set(cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = np.sum(cluster_labels == -1)
            noise_ratio = (
                n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            )

            metrics["noise_ratio"] = noise_ratio

            # Only calculate metrics if we have at least 2 clusters
            if n_clusters >= 2:
                # For metrics, we need to exclude noise points (-1)
                mask = cluster_labels != -1
                non_noise_points = np.sum(mask)
                non_noise_clusters = len(set(cluster_labels[mask]))

                if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                    if "silhouette" in selected_metrics:
                        metrics["silhouette"] = silhouette_score(
                            X_umap_scaled[mask], cluster_labels[mask]
                        )

                    if "davies_bouldin" in selected_metrics:
                        metrics["davies_bouldin"] = davies_bouldin_score(
                            X_umap_scaled[mask], cluster_labels[mask]
                        )

                    if "calinski_harabasz" in selected_metrics:
                        metrics["calinski_harabasz"] = calinski_harabasz_score(
                            X_umap_scaled[mask], cluster_labels[mask]
                        )

                    # Add new metrics based on selection
                    if "hopkins" in selected_metrics:
                        h_stat = hopkins_statistic(X_umap_scaled)
                        metrics["hopkins"] = h_stat

                    if "stability" in selected_metrics:
                        stability = cluster_stability(
                            X_umap_scaled, best_eps, 5, n_iterations=3
                        )
                        metrics["stability"] = stability

                    # Add physics consistency if selected
                    if (
                        "physics_consistency" in selected_metrics
                        and selected_df is not None
                        and not selected_df.empty
                    ):
                        physics_metrics = physics_cluster_consistency(
                            selected_df, cluster_labels
                        )
                        metrics.update(physics_metrics)

            # SMART CONFIDENCE CALCULATION
            confidence_data = calculate_adaptive_confidence_score(
                metrics, clustering_method="dbscan"
            )

            # Create the smart confidence UI
            metrics_children = [create_smart_confidence_ui(confidence_data)]

        except Exception as e:

            trace = traceback.format_exc()
            metrics_children = [html.Div(f"Error calculating confidence: {str(e)}")]

        return fig, debug_text, graph3_selection_umap_store, metrics_children

    except Exception as e:
        print(f"Error in Graph 3 selection UMAP: {e}")

        traceback.print_exc()
        error_msg = f"Error running UMAP on Graph 3 selection: {str(e)}"
        return {}, error_msg, [], []
