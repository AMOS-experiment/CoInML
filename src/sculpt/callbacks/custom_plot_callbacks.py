import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, callback_context, html, no_update

from sculpt.utils.metrics.physics_features import (
    calculate_physics_features_with_profile,
    has_physics_features,
)


@callback(
    Output("custom-feature-plot", "figure"),
    Output("debug-output-custom-plot", "children"),
    Input("plot-custom-features", "n_clicks"),
    State("x-axis-feature", "value"),
    State("y-axis-feature", "value"),
    State("selection-source", "value"),
    State("selected-points-store", "data"),  # Graph 1 selection
    State("selected-points-run-store", "data"),  # Graph 3 selection
    State("combined-data-store", "data"),
    State("umap-graph", "figure"),
    State("umap-graph-selected-run", "figure"),
    prevent_initial_call=True,
)
def update_custom_feature_plot(
    n_clicks,
    x_feature,
    y_feature,
    selection_source,
    graph1_selection,
    graph3_selection,
    combined_data_json,
    graph1_figure,
    graph3_figure,
):
    """Create a custom scatter plot with proper selection handling for both Graph 1 and Graph 3."""
    try:
        # Initialize debug info
        debug_text = []
        debug_text.append(f"X-axis feature: {x_feature}")
        debug_text.append(f"Y-axis feature: {y_feature}")
        debug_text.append(f"Selection source: {selection_source}")

        if not x_feature or not y_feature:
            return {}, "Please select both X and Y axis features."

        # Step 1: Load datasets
        # ------------------------------------------

        # Load combined dataset (original data)
        combined_df = None
        if (
            combined_data_json
            and "combined_df" in combined_data_json
            and combined_data_json["combined_df"] != "{}"
        ):
            combined_df = pd.read_json(
                combined_data_json["combined_df"], orient="split"
            )
            debug_text.append(f"Loaded combined dataset with {len(combined_df)} rows")
        else:
            return {}, "No combined dataset available. Please run UMAP first."

        # Load Graph 3 subset
        graph3_subset_df = None
        if (
            combined_data_json
            and "graph3_subset" in combined_data_json
            and combined_data_json["graph3_subset"] != "{}"
        ):
            graph3_subset_df = pd.read_json(
                combined_data_json["graph3_subset"], orient="split"
            )
            debug_text.append(
                f"Loaded Graph 3 subset with {len(graph3_subset_df)} rows"
            )

        # Load UMAP coordinates
        umap_coords = None
        if (
            "umap_coords" in combined_data_json
            and combined_data_json["umap_coords"] != "{}"
        ):
            umap_coords = pd.read_json(
                combined_data_json["umap_coords"], orient="split"
            )

        # Load Graph 3 UMAP coordinates
        graph3_umap_coords = None
        if (
            "graph3_umap_coords" in combined_data_json
            and combined_data_json["graph3_umap_coords"] != "{}"
        ):
            graph3_umap_coords = pd.read_json(
                combined_data_json["graph3_umap_coords"], orient="split"
            )

        # Verify features exist in datasets
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return {}, f"Features {x_feature} or {y_feature} not found in dataset."

        # Step 2: Process Graph 1 Selection
        # ------------------------------------------
        df_to_plot_graph1 = pd.DataFrame(columns=["x", "y", "label", "source"])

        if (
            selection_source in ["graph2", "both"]
            and graph1_selection
            and umap_coords is not None
        ):
            debug_text.append("\nProcessing Graph 1 selection...")

            # Extract indices
            indices = []

            # Handle different selection types (box, lasso, direct)
            if "range" in graph1_selection:
                # Box selection
                x_range = graph1_selection["range"]["x"]
                y_range = graph1_selection["range"]["y"]

                # Find points inside the box
                selected_mask = (
                    (umap_coords["UMAP1"] >= x_range[0])
                    & (umap_coords["UMAP1"] <= x_range[1])
                    & (umap_coords["UMAP2"] >= y_range[0])
                    & (umap_coords["UMAP2"] <= y_range[1])
                )
                indices = np.where(selected_mask)[0].tolist()

                debug_text.append(f"Box selection with {len(indices)} points")

            elif "lassoPoints" in graph1_selection:
                # Lasso selection
                from matplotlib.path import Path

                # Create path from lasso points
                lasso_x = graph1_selection["lassoPoints"]["x"]
                lasso_y = graph1_selection["lassoPoints"]["y"]
                lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

                # Check which points are inside the lasso
                points_array = np.column_stack(
                    [umap_coords["UMAP1"], umap_coords["UMAP2"]]
                )
                inside_lasso = lasso_path.contains_points(points_array)
                indices = np.where(inside_lasso)[0].tolist()

                debug_text.append(f"Lasso selection with {len(indices)} points")

            elif "points" in graph1_selection:
                # Direct point selection
                indices = [p.get("pointIndex", -1) for p in graph1_selection["points"]]
                indices = [i for i in indices if i >= 0]
                debug_text.append(f"Direct selection with {len(indices)} points")

            # Extract selected rows from combined dataset
            if indices and len(indices) > 0:
                # Ensure indices are valid
                valid_indices = [i for i in indices if 0 <= i < len(combined_df)]

                if valid_indices:
                    # Extract feature values
                    selected_rows = combined_df.iloc[valid_indices]

                    # Get label distribution for debugging
                    label_counts = selected_rows["file_label"].value_counts().to_dict()
                    debug_text.append("Graph 1 selection label distribution:")
                    for label, count in sorted(label_counts.items()):
                        debug_text.append(f"- {label}: {count} points")

                    # Create dataframe for plotting
                    df_to_plot_graph1 = pd.DataFrame(
                        {
                            "x": selected_rows[x_feature],
                            "y": selected_rows[y_feature],
                            "label": selected_rows["file_label"],
                            "source": "Graph 1",
                        }
                    )

                    debug_text.append(
                        f"Extracted {len(df_to_plot_graph1)} points from Graph 1 selection"
                    )
                else:
                    debug_text.append("No valid indices found in Graph 1 selection")

        # Step 3: Process Graph 3 Selection
        # ------------------------------------------
        df_to_plot_graph3 = pd.DataFrame(columns=["x", "y", "label", "source"])

        if (
            selection_source in ["graph3", "both"]
            and graph3_selection
            and graph3_subset_df is not None
            and graph3_umap_coords is not None
        ):
            debug_text.append("\nProcessing Graph 3 selection...")

            # Extract indices using the same approach as in update_umap_graph3_selection
            indices = []

            # Handle box selection
            if "range" in graph3_selection:
                x_range = graph3_selection["range"]["x"]
                y_range = graph3_selection["range"]["y"]
                debug_text.append(
                    f"Box selection in Graph 3: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"
                )

                # Find points inside the box
                selected_mask = (
                    (graph3_umap_coords["UMAP1"] >= x_range[0])
                    & (graph3_umap_coords["UMAP1"] <= x_range[1])
                    & (graph3_umap_coords["UMAP2"] >= y_range[0])
                    & (graph3_umap_coords["UMAP2"] <= y_range[1])
                )
                indices = np.where(selected_mask)[0].tolist()
                debug_text.append(
                    f"Found {len(indices)} points in box selection from Graph 3"
                )

            # Handle lasso selection
            elif "lassoPoints" in graph3_selection:
                debug_text.append("Lasso selection in Graph 3")

                # Extract lasso polygon coordinates
                lasso_x = graph3_selection["lassoPoints"]["x"]
                lasso_y = graph3_selection["lassoPoints"]["y"]

                from matplotlib.path import Path

                # Create a Path object from the lasso points
                lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

                # Check which points are within the lasso path
                points_array = np.column_stack(
                    [graph3_umap_coords["UMAP1"], graph3_umap_coords["UMAP2"]]
                )
                inside_lasso = lasso_path.contains_points(points_array)

                # Get indices of points inside the lasso
                indices = np.where(inside_lasso)[0].tolist()
                debug_text.append(
                    f"Found {len(indices)} points in lasso selection from Graph 3"
                )

            # Handle direct point selection
            elif "points" in graph3_selection and graph3_selection["points"]:
                points = graph3_selection["points"]
                debug_text.append(f"Found {len(points)} points in Graph 3 selection")

                # For direct point selection, extract indices directly
                for point in points:
                    idx = point.get("pointIndex", -1)
                    if 0 <= idx < len(graph3_subset_df):
                        indices.append(idx)

                debug_text.append(
                    f"Extracted {len(indices)} valid indices from Graph 3 points"
                )

            if indices and len(indices) > 0:
                # Ensure indices are valid
                valid_indices = [i for i in indices if 0 <= i < len(graph3_subset_df)]

                if valid_indices:
                    # Extract feature values
                    selected_rows = graph3_subset_df.iloc[valid_indices]

                    # Get label distribution for debugging
                    label_counts = selected_rows["file_label"].value_counts().to_dict()
                    debug_text.append("Graph 3 selection label distribution:")
                    for label, count in sorted(label_counts.items()):
                        debug_text.append(f"- {label}: {count} points")

                    # Create dataframe for plotting
                    df_to_plot_graph3 = pd.DataFrame(
                        {
                            "x": selected_rows[x_feature],
                            "y": selected_rows[y_feature],
                            "label": selected_rows["file_label"],
                            "source": "Graph 3",
                        }
                    )

                    debug_text.append(
                        f"Extracted {len(df_to_plot_graph3)} points from Graph 3 selection"
                    )
                else:
                    debug_text.append("No valid indices found in Graph 3 selection")

        # Step 4: Combine data sources based on selection_source
        # ------------------------------------------------------------
        df_to_plot = pd.DataFrame(columns=["x", "y", "label", "source"])

        if selection_source == "graph2":
            df_to_plot = df_to_plot_graph1
        elif selection_source == "graph3":
            df_to_plot = df_to_plot_graph3
        elif selection_source == "both":
            df_to_plot = pd.concat(
                [df_to_plot_graph1, df_to_plot_graph3], ignore_index=True
            )

        if df_to_plot.empty:
            return {}, "No points to plot with the selected criteria."

        # Step 5: Create color map from both figures for consistency
        # ------------------------------------------------------------
        color_map = {}

        # Get colors from Graph 1 figure
        if graph1_figure and "data" in graph1_figure:
            for trace in graph1_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    name = trace["name"]
                    if " (" in name:  # Clean label if it contains count
                        name = name.split(" (")[0]
                    color_map[name] = trace["marker"]["color"]

        # Get additional colors from Graph 3 figure
        if graph3_figure and "data" in graph3_figure:
            for trace in graph3_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    name = trace["name"]
                    if " (" in name:  # Clean label if it contains count
                        name = name.split(" (")[0]
                    if name not in color_map:  # Don't overwrite existing colors
                        color_map[name] = trace["marker"]["color"]

        # Step 6: Create the plot
        # ------------------------------------------------------------
        import plotly.graph_objects as go

        fig = go.Figure()

        # Use different symbols for different sources if plotting both
        if selection_source == "both":
            symbols = {"Graph 1": "circle", "Graph 3": "square"}

            # Add traces by source and label
            for source in df_to_plot["source"].unique():
                for label in df_to_plot[df_to_plot["source"] == source][
                    "label"
                ].unique():
                    mask = (df_to_plot["source"] == source) & (
                        df_to_plot["label"] == label
                    )
                    points = df_to_plot[mask]

                    fig.add_trace(
                        go.Scatter(
                            x=points["x"],
                            y=points["y"],
                            mode="markers",
                            marker=dict(
                                size=8,
                                color=color_map.get(label),
                                symbol=symbols.get(source, "circle"),
                                opacity=0.7,
                            ),
                            name=f"{label} ({source}, {len(points)} pts)",
                        )
                    )
        else:
            # Add one trace per label
            for label in df_to_plot["label"].unique():
                points = df_to_plot[df_to_plot["label"] == label]

                fig.add_trace(
                    go.Scatter(
                        x=points["x"],
                        y=points["y"],
                        mode="markers",
                        marker=dict(size=8, color=color_map.get(label), opacity=0.7),
                        name=f"{label} ({len(points)} pts)",
                    )
                )

        # Update layout
        fig.update_layout(
            height=600,
            title=f"Custom Feature Plot: {x_feature} vs {y_feature}",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title="Data Source",
        )

        return fig, "<br>".join(debug_text)

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        return {}, f"Error: {str(e)}<br><pre>{trace}</pre>"


# Main callback for Graph 1.5 scatter plot
@callback(
    Output("scatter-graph15", "figure"),
    Output("debug-output-graph15", "children"),
    Output("quality-metrics-graph15", "children"),
    Input("generate-plot-graph15", "n_clicks"),
    State("stored-files", "data"),
    State("file-selector-graph15", "value"),
    State("x-axis-feature-graph15", "value"),
    State("y-axis-feature-graph15", "value"),
    State("sample-frac-graph15", "value"),
    State("visualization-type-graph15", "value"),
    State("point-opacity-graph15", "value"),
    State("heatmap-bandwidth-graph15", "value"),
    State("heatmap-colorscale-graph15", "value"),
    State("show-points-overlay-graph15", "value"),
    State("color-mode-graph15", "value"),
    State("file-config-assignments-store", "data"),  # ADD THIS
    State("configuration-profiles-store", "data"),  # ADD THIS
    prevent_initial_call=True,
)
def update_scatter_graph15(
    n_clicks,
    stored_files,
    selected_ids,
    x_feature,
    y_feature,
    sample_frac,
    visualization_type,
    point_opacity,
    heatmap_bandwidth,
    heatmap_colorscale,
    show_points_overlay,
    color_mode,
    assignments_store,
    profiles_store,
):  # ADD PARAMETERS
    """Generate custom scatter plot for Graph 1.5."""
    if not stored_files:
        return {}, "No files uploaded.", []

    if not selected_ids:
        return {}, "No files selected for plotting.", []

    if not x_feature or not y_feature:
        return {}, "Please select both X and Y axis features.", []

    try:
        debug_text = []
        debug_text.append(f"X-axis feature: {x_feature}")
        debug_text.append(f"Y-axis feature: {y_feature}")
        debug_text.append(f"Visualization type: {visualization_type}")

        # Process selected files
        sampled_dfs = []
        debug_str = ""

        for f in stored_files:
            if f["id"] in selected_ids:
                try:
                    df = pd.read_json(f["data"], orient="split")
                    is_selection = f.get("is_selection", False)

                    if is_selection:
                        # This is a saved selection file
                        debug_str += f"{f['filename']}: Selection file with {len(df)} events.<br>"
                        df["file_label"] = f[
                            "filename"
                        ]  # Make sure it has a file_label
                        sampled_dfs.append(df)
                    else:
                        # Regular COLTRIMS file
                        df["file_label"] = f["filename"]  # Add file name as a label

                        # Check if physics features already exist
                        if not has_physics_features(df):
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
                                profile_config = profiles_store[profile_name]
                                try:
                                    df = calculate_physics_features_with_profile(
                                        df, profile_config
                                    )
                                except Exception as e:
                                    debug_str += f"Error calculating features for {f['filename']}: {str(e)}.<br>"
                                    continue  # Skip this file
                            else:
                                debug_str += f"Skipping {f['filename']} - no valid profile assigned.<br>"
                                continue  # Skip files without proper profile assignment

                        # Sample the data to reduce processing time
                        sample_size = max(
                            int(len(df) * sample_frac), 100
                        )  # Ensure at least 100 points
                        if len(df) > sample_size:
                            sampled = df.sample(n=sample_size, random_state=42)
                        else:
                            sampled = df

                        debug_str += f"{f['filename']}: {len(df)} events, sampled {len(sampled)}.<br>"
                        sampled_dfs.append(sampled)
                except Exception as e:
                    debug_str += f"Error processing {f['filename']}: {str(e)}.<br>"

        # Combine all selected datasets
        if not sampled_dfs:
            return {}, "No valid data to plot.", []

        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        debug_str += f"Combined data shape: {combined_df.shape}.<br>"

        # Verify features exist in dataset
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return {}, f"Features {x_feature} or {y_feature} not found in dataset.", []

        # Apply DBSCAN for cluster coloring (if needed)
        cluster_labels = None
        best_eps = 0.5  # Default value

        if color_mode == "cluster":
            debug_text.append("Applying DBSCAN clustering for coloring")

            # Extract only the two features we're plotting
            X_features = combined_df[[x_feature, y_feature]].to_numpy()

            # Handle NaN/inf values
            X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Standardize the data
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_features_scaled = scaler.fit_transform(X_features)

            # Find a reasonable epsilon for DBSCAN
            eps_candidates = np.linspace(0.1, 1.0, 10)
            best_eps = 0.5  # Default
            max_clusters = 0

            # Try different eps values and pick the one that gives a reasonable number of clusters
            for eps in eps_candidates:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(X_features_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                noise_count = np.sum(labels == -1)
                noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0

                if n_clusters >= 2 and noise_ratio < 0.5 and n_clusters > max_clusters:
                    max_clusters = n_clusters
                    best_eps = eps

            # Run DBSCAN with the best eps
            dbscan = DBSCAN(eps=best_eps, min_samples=5)
            cluster_labels = dbscan.fit_predict(X_features_scaled)

            debug_text.append(
                f"DBSCAN identified {max_clusters} clusters with eps={best_eps}"
            )

        # Create color maps for file labels and clusters

        # Create color map based on file labels
        unique_labels = combined_df["file_label"].unique()
        colorscale = px.colors.qualitative.Plotly
        color_map = {
            label: colorscale[i % len(colorscale)]
            for i, label in enumerate(unique_labels)
        }

        # Create color map for clusters if needed
        cluster_colors = {}
        if color_mode == "cluster" and cluster_labels is not None:
            # Special handling for noise points (-1 label)
            unique_clusters = sorted(set(cluster_labels))
            if -1 in unique_clusters:
                # Move noise to the end
                unique_clusters.remove(-1)
                unique_clusters.append(-1)

            # Choose color scale for clusters
            if len(unique_clusters) <= 10:
                cluster_colorscale = (
                    px.colors.qualitative.D3
                )  # Good for distinct clusters
            else:
                # For many clusters, use a continuous colorscale
                cluster_colorscale = px.colors.sequential.Viridis

            # Create color mapping for clusters
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

        # Initialize figure
        fig = go.Figure()

        # Heatmap visualization
        if visualization_type == "heatmap":
            from scipy.stats import gaussian_kde

            # Get coordinates for heatmap
            x_data = combined_df[x_feature].values
            y_data = combined_df[y_feature].values

            # Create the grid for the heatmap
            x_min, x_max = np.min(x_data) - 0.05 * (
                np.max(x_data) - np.min(x_data)
            ), np.max(x_data) + 0.05 * (np.max(x_data) - np.min(x_data))
            y_min, y_max = np.min(y_data) - 0.05 * (
                np.max(y_data) - np.min(y_data)
            ), np.max(y_data) + 0.05 * (np.max(y_data) - np.min(y_data))

            # Create a meshgrid
            grid_size = 200
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            xx, yy = np.meshgrid(x_grid, y_grid)
            grid_points = np.column_stack([xx.flatten(), yy.flatten()])

            # Compute KDE (Kernel Density Estimation)
            data_points = np.column_stack([x_data, y_data])
            kde = gaussian_kde(data_points.T, bw_method=heatmap_bandwidth)
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
                    for label in unique_labels:
                        mask = combined_df["file_label"] == label
                        df_subset = combined_df[mask]

                        fig.add_trace(
                            go.Scatter(
                                x=df_subset[x_feature],
                                y=df_subset[y_feature],
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
                elif color_mode == "cluster" and cluster_labels is not None:
                    # Color by DBSCAN cluster with reduced opacity
                    for cluster in sorted(set(cluster_labels)):
                        mask = cluster_labels == cluster
                        cluster_points = combined_df.iloc[mask]

                        # For noise points, make them smaller and more transparent
                        marker_size = 3 if cluster == -1 else 4
                        marker_opacity = 0.2 if cluster == -1 else 0.3

                        fig.add_trace(
                            go.Scatter(
                                x=cluster_points[x_feature],
                                y=cluster_points[y_feature],
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
            # Scatter plot visualization
            if color_mode == "file":
                # Color by file source
                for label in unique_labels:
                    mask = combined_df["file_label"] == label
                    df_subset = combined_df[mask]

                    fig.add_trace(
                        go.Scatter(
                            x=df_subset[x_feature],
                            y=df_subset[y_feature],
                            mode="markers",
                            marker=dict(
                                size=8,
                                color=color_map[label],
                                opacity=point_opacity,
                                line=dict(width=0),
                            ),
                            name=f"{label} ({len(df_subset)} pts)",
                        )
                    )

            elif color_mode == "cluster" and cluster_labels is not None:
                # Color by DBSCAN cluster
                for cluster in sorted(set(cluster_labels)):
                    mask = cluster_labels == cluster
                    cluster_points = combined_df.iloc[mask]

                    # For noise points, make them smaller and more transparent
                    marker_size = 5 if cluster == -1 else 8
                    marker_opacity = (
                        point_opacity * 0.7 if cluster == -1 else point_opacity
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=cluster_points[x_feature],
                            y=cluster_points[y_feature],
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

        # Update figure layout
        title_suffix = ""
        if color_mode == "cluster" and cluster_labels is not None:
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

        # Adjust figure height based on legend size
        figure_height = 600
        if len(fig.data) > 15:
            figure_height = 750
        elif len(fig.data) > 10:
            figure_height = 700
        elif len(fig.data) > 6:
            figure_height = 650

        # Apply the layout settings
        if visualization_type == "heatmap":
            title = f"Custom Feature Heatmap: {x_feature} vs {y_feature}{title_suffix}"
        else:
            title = (
                f"Custom Feature Scatter Plot: {x_feature} vs {y_feature}{title_suffix}"
            )

        fig.update_layout(
            height=figure_height,
            title=title,
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title=f"{'Clusters' if color_mode == 'cluster' else 'Data File'}",
            dragmode="lasso",  # Explicitly set lasso as default selection mode
            legend=legend_config,
            modebar=dict(
                add=["lasso2d", "select2d"],
                remove=["pan2d", "autoScale2d"],
                orientation="h",
                bgcolor="rgba(255,255,255,0.9)",
                color="rgba(68,68,68,1)",
                activecolor="rgba(254,95,85,1)"
            ),  # Add these tools to the modebar
            margin=dict(l=50, r=50, t=50, b=100),  # Increased bottom margin for legend
        )

        # Simplified cluster information (no complex metrics)
        metrics_children = []

        if cluster_labels is not None and color_mode == "cluster":
            # Count clusters and noise points
            unique_clusters = set(cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = np.sum(cluster_labels == -1)
            noise_ratio = (
                n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            )

            # Create simple cluster info UI
            metrics_children = [
                html.H4(
                    "Clustering Information",
                    style={"fontSize": "14px", "marginBottom": "5px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Clusters Detected: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(f"{n_clusters}"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Noise Points: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(f"{n_noise} ({noise_ratio:.1%})"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "DBSCAN Epsilon: ", style={"fontWeight": "bold"}
                                ),
                                html.Span(f"{best_eps:.3f}"),
                            ]
                        ),
                    ]
                ),
            ]

        # Convert debug_text list to string for output
        debug_str += "<br>".join(debug_text)

        return fig, debug_str, metrics_children

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        return {}, f"Error: {str(e)}<br><pre>{trace}</pre>", []


@callback(
    Output("graph25", "figure"),
    Output("debug-output-graph25", "children"),
    Output("selected-points-info-graph25", "children"),
    Input("show-selected-graph15", "n_clicks"),
    State("selected-points-store-graph15", "data"),
    State("x-axis-feature-graph15", "value"),
    State("y-axis-feature-graph15", "value"),
    State("scatter-graph15", "figure"),
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
    State("file-config-assignments-store", "data"),  # ADD THIS
    State("configuration-profiles-store", "data"),  # ADD THIS
    prevent_initial_call=True,
)
def update_graph25(
    n_clicks,
    selectedData,
    x_feature,
    y_feature,
    original_figure,
    selected_ids,
    stored_files,
    assignments_store,
    profiles_store,
):  # ADD THESE PARAMETERS
    """Display the selected points from Graph 1.5."""
    if not n_clicks or not selectedData:
        return {}, "No points selected.", "No points selected."

    debug_text = []

    try:
        # Process selected files to get the combined dataframe
        sampled_dfs = []

        for f in stored_files:
            if f["id"] in selected_ids:
                try:
                    df = pd.read_json(f["data"], orient="split")
                    is_selection = f.get("is_selection", False)

                    df["file_label"] = f["filename"]  # Add file name as a label

                    if not is_selection:
                        # Check if physics features already exist
                        if not has_physics_features(df):
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
                                profile_config = profiles_store[profile_name]
                                try:
                                    df = calculate_physics_features_with_profile(
                                        df, profile_config
                                    )
                                except Exception as e:
                                    debug_text.append(
                                        f"Error calculating features for {f['filename']}: {str(e)}"
                                    )
                                    continue  # Skip this file
                            else:
                                debug_text.append(
                                    f"Skipping {f['filename']} - no valid profile assigned"
                                )
                                continue  # Skip files without proper profile assignment

                    sampled_dfs.append(df)

                except Exception as e:
                    debug_text.append(f"Error processing {f['filename']}: {str(e)}")

        # Combine datasets
        if not sampled_dfs:
            return {}, "No valid files selected.", "No valid files selected."

        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        debug_text.append(f"Combined dataframe shape: {combined_df.shape}")

        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return (
                {},
                f"Features {x_feature} or {y_feature} not found in data.",
                "Features not found.",
            )

        # Extract selected points
        selected_indices = []

        # Handle box selection
        if "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]

            debug_text.append(
                f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"
            )

            selected_mask = (
                (combined_df[x_feature] >= x_range[0])
                & (combined_df[x_feature] <= x_range[1])
                & (combined_df[y_feature] >= y_range[0])
                & (combined_df[y_feature] <= y_range[1])
            )
            selected_indices = np.where(selected_mask)[0].tolist()
            debug_text.append(f"Found {len(selected_indices)} points in box selection")

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            from matplotlib.path import Path

            # Extract lasso polygon coordinates
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            debug_text.append(f"Lasso selection with {len(lasso_x)} points")

            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

            # Check which points are within the lasso path
            points_array = np.column_stack(
                [combined_df[x_feature].values, combined_df[y_feature].values]
            )
            inside_lasso = lasso_path.contains_points(points_array)

            # Get indices of points inside the lasso
            selected_indices = np.where(inside_lasso)[0].tolist()
            debug_text.append(
                f"Found {len(selected_indices)} points in lasso selection"
            )

        # Handle direct point selection
        elif "points" in selectedData and selectedData["points"]:
            debug_text.append(
                f"Direct selection with {len(selectedData['points'])} points"
            )

            for point in selectedData["points"]:
                x_val = point.get("x")
                y_val = point.get("y")

                if x_val is not None and y_val is not None:
                    # Find the closest point in the dataset
                    distances = (combined_df[x_feature] - x_val) ** 2 + (
                        combined_df[y_feature] - y_val
                    ) ** 2
                    closest_idx = distances.idxmin()
                    selected_indices.append(closest_idx)

        if not selected_indices:
            return {}, "No points found in the selection area.", "No points found."

        # Extract the selected points
        selected_df = combined_df.iloc[selected_indices].reset_index(drop=True)
        debug_text.append(f"Selected {len(selected_df)} points")

        # Create visualization
        import plotly.graph_objects as go

        fig = go.Figure()

        # Extract color information from original figure
        color_map = {}
        if original_figure and "data" in original_figure:
            for trace in original_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    # Clean the label if it contains point count
                    clean_name = trace["name"]
                    if " (" in clean_name:
                        clean_name = clean_name.split(" (")[0]
                    color_map[clean_name] = trace["marker"]["color"]

        # Add traces for each file label
        for label in selected_df["file_label"].unique():
            mask = selected_df["file_label"] == label
            df_subset = selected_df[mask]

            # Get color from original figure if available
            color = color_map.get(label, None)

            fig.add_trace(
                go.Scatter(
                    x=df_subset[x_feature],
                    y=df_subset[y_feature],
                    mode="markers",
                    marker=dict(size=8, color=color, opacity=0.7),
                    name=f"{label} ({len(df_subset)} pts)",
                )
            )

        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"Selected Points from Graph 1.5: {x_feature} vs {y_feature}",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title="Data File",
        )

        # Count points by file for information panel
        file_counts = selected_df["file_label"].value_counts().to_dict()

        # Create info text
        info_text = [
            html.Div(f"Total selected points: {len(selected_df)}"),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div(
                [
                    html.Div(f"{file}: {count} events")
                    for file, count in file_counts.items()
                ],
                style={"marginLeft": "10px"},
            ),
        ]

        return fig, "<br>".join(debug_text), info_text

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        return {}, f"Error: {str(e)}<br><pre>{trace}</pre>", f"Error: {str(e)}"


def create_select_all_callback(id_prefix):
    """Create a callback for Select All/None functionality for a specific id_prefix."""

    @callback(
        Output({"type": f"feature-selector-{id_prefix}", "category": ALL}, "value"),
        Input({"type": f"select-all-btn-{id_prefix}", "index": ALL}, "n_clicks"),
        Input({"type": f"select-none-btn-{id_prefix}", "index": ALL}, "n_clicks"),
        State({"type": f"feature-selector-{id_prefix}", "category": ALL}, "options"),
        State({"type": f"feature-selector-{id_prefix}", "category": ALL}, "id"),
        prevent_initial_call=True,
    )
    def handle_select_buttons(all_clicks, none_clicks, all_options, all_ids):
        """Handle Select All and Select None button clicks."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        # Get which button was clicked
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        triggered_info = eval(triggered_id)  # Convert string to dict

        # Initialize result with current values
        result = [no_update] * len(all_options)

        if "select-all-btn" in triggered_info["type"]:
            clicked_index = triggered_info["index"]
            if clicked_index == "all":
                # Select all features in all categories
                result = [[opt["value"] for opt in options] for options in all_options]
            else:
                # Select all features in specific category
                for i, category_id in enumerate(all_ids):
                    if category_id["category"] == clicked_index:
                        result[i] = [opt["value"] for opt in all_options[i]]

        elif "select-none-btn" in triggered_info["type"]:
            clicked_index = triggered_info["index"]
            if clicked_index == "all":
                # Deselect all features in all categories
                result = [[]] * len(all_options)
            else:
                # Deselect all features in specific category
                for i, category_id in enumerate(all_ids):
                    if category_id["category"] == clicked_index:
                        result[i] = []

        return result

    return handle_select_buttons


# Register callbacks for each feature selection UI
create_select_all_callback("graph1")
create_select_all_callback("graph3")
create_select_all_callback("graph3-selection")
create_select_all_callback("autoencoder")
create_select_all_callback("genetic")
create_select_all_callback("mi")
