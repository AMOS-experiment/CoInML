import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html
from sculpt.utils.unit_converter import generate_feature_label



def _build_physics_filter_options(feature_columns):
    """
    Build dropdown options from available physics feature columns.
    Filters out raw particle momentum columns (particle_*) and metadata columns,
    keeping only calculated physics features.
    
    Returns a list of {label, value} dicts for the dropdown.
    """
    from sculpt.utils.unit_converter import generate_feature_label

    # Columns to exclude from the physics parameter filter
    exclude_patterns = ["particle_", "UMAP", "file_label", "density", "cluster"]

    physics_cols = []
    for col in feature_columns:
        if any(col.startswith(p) for p in exclude_patterns):
            continue
        physics_cols.append(col)

    # Build options with nice labels
    options = []
    for col in physics_cols:
        label = generate_feature_label(col)
        options.append({"label": label, "value": col})

    return options


# Callback to dynamically populate the physics-parameter-dropdown
# (Custom Feature Filtering section)
@callback(
    Output("physics-parameter-dropdown", "options"),
    Input("features-data-store", "data"),
    prevent_initial_call=True,
)
def update_physics_filter_dropdowns(features_data):
    """Dynamically populate the physics parameter filter dropdown
    based on the actual calculated features, so it works with any
    particle configuration (not just the default 2 ions/1 neutral/2 electrons)."""
    if not features_data or "column_names" not in features_data:
        return []

    return _build_physics_filter_options(features_data["column_names"])


# Callback to dynamically populate the umap-physics-parameter-dropdown
# (UMAP Filtering section)
@callback(
    Output("umap-physics-parameter-dropdown", "options"),
    Input("combined-data-store", "data"),
    prevent_initial_call=True,
)
def update_umap_physics_filter_dropdowns(combined_data_json):
    """Dynamically populate the UMAP physics parameter filter dropdown
    based on the actual columns in the combined data after UMAP has been run."""
    if not combined_data_json or "combined_df" not in combined_data_json:
        return []

    try:
        combined_df = pd.read_json(combined_data_json["combined_df"], orient="split")
        if combined_df.empty:
            return []
        return _build_physics_filter_options(combined_df.columns.tolist())
    except Exception as e:
        print(f"Error populating UMAP physics filter dropdown: {e}")
        return []



from sculpt.utils.metrics.physics_features import (
    calculate_physics_features,
    calculate_physics_features_with_profile,
    has_physics_features,
)


# Callback to apply density filter
@callback(
    Output("filtered-data-graph", "figure", allow_duplicate=True),
    Output("density-filter-status", "children"),
    Output("density-filter-info", "children"),
    Output("filtered-data-store", "data", allow_duplicate=True),
    Input("apply-density-filter", "n_clicks"),
    State("density-bandwidth-slider", "value"),
    State("density-threshold-slider", "value"),
    State("x-axis-feature-graph15", "value"),
    State("y-axis-feature-graph15", "value"),
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
    State("sample-frac-graph15", "value"),  # Add sampling control
    State("file-config-assignments-store", "data"),  # ADD THIS
    State("configuration-profiles-store", "data"),  # ADD THIS
    prevent_initial_call=True,
)
def apply_density_filter(
    n_clicks,
    bandwidth,
    threshold_percentile,
    x_feature,
    y_feature,
    selected_ids,
    stored_files,
    sample_frac,
    assignments_store,
    profiles_store,
):  # ADD PARAMETERS
    ctx = dash.callback_context
    if not ctx.triggered or "apply-density-filter" not in ctx.triggered[0]["prop_id"]:
        raise dash.exceptions.PreventUpdate

    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    print(
        f"Density filter called with bandwidth={bandwidth}, threshold={threshold_percentile}"
    )

    try:
        # Process selected files - WITH SAMPLING to reduce processing time
        sampled_dfs = []
        total_rows = 0
        sampled_rows = 0

        for f in stored_files:
            if f["id"] in selected_ids:
                try:
                    df = pd.read_json(f["data"], orient="split")
                    total_rows += len(df)

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
                                    print(
                                        f"Error calculating features for {f['filename']}: {e}"
                                    )
                                    continue  # Skip this file
                            else:
                                print(
                                    f"Skipping {f['filename']} - no valid profile assigned"
                                )
                                continue  # Skip files without proper profile assignment

                        # Apply sampling to reduce processing time
                        if len(df) > 1000:
                            # Use the sample_frac from the UI or a default value
                            actual_sample_frac = (
                                sample_frac if sample_frac is not None else 0.1
                            )
                            sample_size = max(
                                int(len(df) * actual_sample_frac), 500
                            )  # Ensure reasonable number
                            df = df.sample(n=sample_size, random_state=42)
                            sampled_rows += len(df)
                        else:
                            sampled_rows += len(df)
                    else:
                        sampled_rows += len(df)

                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']}: {str(e)}")

        # Combine datasets
        if not sampled_dfs:
            return {}, "No valid files selected.", "No data available.", {}

        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        print(
            f"Combined dataframe: {sampled_rows} sampled rows from {total_rows} total rows"
        )

        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return (
                {},
                f"Features {x_feature} or {y_feature} not found in data.",
                "Features not found.",
                {},
            )

        # Extract feature data for density calculation
        feature_data = combined_df[[x_feature, y_feature]].to_numpy()

        # Handle NaN/inf values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

        # OPTIMIZED METHOD: Use grid-based density estimation with fewer bins for large datasets
        # Adjust bins based on data size
        num_bins = min(
            100, max(20, int(np.sqrt(len(feature_data))))
        )  # Scale bins with data size
        print(
            f"Using {num_bins} bins for density estimation on {len(feature_data)} points"
        )

        x_min, x_max = np.min(feature_data[:, 0]), np.max(feature_data[:, 1])
        y_min, y_max = np.min(feature_data[:, 1]), np.max(feature_data[:, 1])

        # Add small padding to min/max
        padding = 0.01
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        # OPTIMIZATION: Use a smaller sigma for large datasets
        sigma_scale = max(0.5, min(10, 10.0 * np.sqrt(1000.0 / len(feature_data))))
        actual_sigma = bandwidth * sigma_scale
        print(f"Using sigma {actual_sigma} for smoothing (bandwidth={bandwidth})")

        # Create 2D histogram - faster approach
        H, xedges, yedges = np.histogram2d(
            feature_data[:, 0],
            feature_data[:, 1],
            bins=num_bins,
            range=[[x_min, x_max], [y_min, y_max]],
        )

        # Apply smoothing with optimized sigma
        from scipy.ndimage import gaussian_filter

        H_smooth = gaussian_filter(H, sigma=actual_sigma)

        # Assign density values to each point - vectorized approach
        x_indices = np.clip(
            np.floor((feature_data[:, 0] - x_min) / (x_max - x_min) * num_bins).astype(
                int
            ),
            0,
            num_bins - 1,
        )
        y_indices = np.clip(
            np.floor((feature_data[:, 1] - y_min) / (y_max - y_min) * num_bins).astype(
                int
            ),
            0,
            num_bins - 1,
        )

        densities = H_smooth[x_indices, y_indices]

        # Calculate density threshold
        threshold = np.percentile(densities, threshold_percentile)

        # Filter by density
        high_density_mask = densities >= threshold
        filtered_df = combined_df.iloc[high_density_mask].copy()
        filtered_df["density"] = densities[
            high_density_mask
        ]  # Store density for reference

        print(
            f"Kept {len(filtered_df)} high-density points ({len(filtered_df)/len(combined_df):.1%} of original)"
        )

        # Check if we have any points left
        if len(filtered_df) == 0:
            return (
                {},
                "No points remain after filtering.",
                "Try lowering the threshold.",
                {},
            )

        # Create visualization of filtered data - OPTIMIZED RENDERING
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()

        # Use a standard color palette
        color_map = px.colors.qualitative.Plotly

        # OPTIMIZATION: Limit points displayed for better performance
        # For background points, randomly sample to reduce visual clutter
        if len(combined_df) > 5000:
            background_idx = np.random.choice(
                len(combined_df), size=5000, replace=False
            )
            background_df = combined_df.iloc[background_idx]
        else:
            background_df = combined_df

        # Add original data as background with reduced opacity
        fig.add_trace(
            go.Scatter(
                x=background_df[x_feature],
                y=background_df[y_feature],
                mode="markers",
                marker=dict(
                    size=3,  # Smaller points for background
                    color="gray",
                    opacity=0.05,  # Less opacity
                ),
                name="Original data",
                showlegend=True,
            )
        )

        # Add traces for each file label
        for i, label in enumerate(filtered_df["file_label"].unique()):
            mask = filtered_df["file_label"] == label
            df_subset = filtered_df[mask]

            # If subset is very large, sample it for display
            if len(df_subset) > 2000:
                subset_idx = np.random.choice(len(df_subset), size=2000, replace=False)
                df_subset = df_subset.iloc[subset_idx]
                display_name = f"{label} ({len(filtered_df[mask])} pts, showing 2000)"
            else:
                display_name = f"{label} ({len(df_subset)} pts)"

            # Use standard color palette
            color = color_map[i % len(color_map)]

            fig.add_trace(
                go.Scatter(
                    x=df_subset[x_feature],
                    y=df_subset[y_feature],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.7),
                    name=display_name,
                    showlegend=True,
                )
            )

        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"Density-Filtered Data: {x_feature} vs {y_feature} (Kept {len(filtered_df)} points)",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title="Data Source",
            showlegend=True,
        )

        # Store filtered data
        filtered_data_store = {
            "filtered_df": filtered_df.to_json(date_format="iso", orient="split"),
            "x_feature": x_feature,
            "y_feature": y_feature,
            "filtering_method": "density",
            "params": {
                "bandwidth": bandwidth,
                "threshold_percentile": threshold_percentile,
                "threshold_value": float(threshold),
            },
        }

        # Create info text
        file_counts = filtered_df["file_label"].value_counts().to_dict()
        info_text = [
            html.Div(
                f"Total points after filtering: {len(filtered_df)} ({len(filtered_df)/len(combined_df):.1%} of original)"
            ),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div(
                [
                    html.Div(f"{file}: {count} events")
                    for file, count in file_counts.items()
                ],
                style={"marginLeft": "10px"},
            ),
        ]

        return (
            fig,
            "Density filtering applied successfully!",
            info_text,
            filtered_data_store,
        )

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        print(f"Density filter error: {str(e)}\n{trace}")
        return {}, f"Error applying density filter: {str(e)}", f"Error: {str(e)}", {}


# Callback to apply parameter filter
@callback(
    Output("filtered-data-graph", "figure", allow_duplicate=True),
    Output("parameter-filter-status", "children"),
    Output("filtered-data-store", "data", allow_duplicate=True),
    Output("filtered-data-info", "children", allow_duplicate=True),
    Input("apply-parameter-filter", "n_clicks"),
    State("physics-parameter-dropdown", "value"),
    State("parameter-range-slider", "value"),
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
    State("x-axis-feature-graph15", "value"),
    State("y-axis-feature-graph15", "value"),
    State("file-config-assignments-store", "data"),  # ADD THIS
    State("configuration-profiles-store", "data"),  # ADD THIS
    prevent_initial_call=True,
)
def apply_parameter_filter(
    n_clicks,
    parameter,
    parameter_range,
    selected_ids,
    stored_files,
    x_feature,
    y_feature,
    assignments_store,
    profiles_store,
):  # ADD PARAMETERS
    ctx = dash.callback_context
    if not ctx.triggered or "apply-parameter-filter" not in ctx.triggered[0]["prop_id"]:
        raise dash.exceptions.PreventUpdate

    print(
        f"Parameter filter called with {parameter}=[{parameter_range[0]}, {parameter_range[1]}]"
    )

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
                                    print(
                                        f"Error calculating features for {f['filename']}: {e}"
                                    )
                                    continue  # Skip this file
                            else:
                                print(
                                    f"Skipping {f['filename']} - no valid profile assigned"
                                )
                                continue  # Skip files without proper profile assignment

                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']}: {str(e)}")

        # Combine datasets
        if not sampled_dfs:
            return {}, "No valid files selected.", {}, "No data available."

        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)

        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return (
                {},
                f"Features {x_feature} or {y_feature} not found in data.",
                {},
                "Features not found.",
            )

        if parameter not in combined_df.columns:
            return (
                {},
                f"Parameter {parameter} not found in data.",
                {},
                "Parameter not found.",
            )

        # Filter by parameter range
        parameter_mask = (combined_df[parameter] >= parameter_range[0]) & (
            combined_df[parameter] <= parameter_range[1]
        )
        filtered_df = combined_df.loc[parameter_mask].copy()

        if len(filtered_df) == 0:
            return (
                {},
                "No points remain after filtering.",
                {},
                "Try adjusting the parameter range.",
            )

        # Create visualization of filtered data - simpler version
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()

        # Add filtered data traces - group by file label
        color_map = px.colors.qualitative.Plotly  # Use standard Plotly colors

        for i, label in enumerate(filtered_df["file_label"].unique()):
            mask = filtered_df["file_label"] == label
            df_subset = filtered_df[mask]

            # Use standard color palette
            color = color_map[i % len(color_map)]

            fig.add_trace(
                go.Scatter(
                    x=df_subset[x_feature],
                    y=df_subset[y_feature],
                    mode="markers",
                    marker=dict(size=8, color=color, opacity=0.7),
                    name=f"{label} ({len(df_subset)} pts)",
                    showlegend=True,
                )
            )

        # Add a separate trace for all original data points (no histogram inset)
        # Sample if too many points for performance
        if len(combined_df) > 5000:
            background_idx = np.random.choice(
                len(combined_df), size=5000, replace=False
            )
            background_df = combined_df.iloc[background_idx]
        else:
            background_df = combined_df

        fig.add_trace(
            go.Scatter(
                x=background_df[x_feature],
                y=background_df[y_feature],
                mode="markers",
                marker=dict(size=4, color="gray", opacity=0.1),
                name=f"All data points ({len(combined_df)} total)",
                showlegend=True,
            )
        )

        # Update figure layout - simpler version
        fig.update_layout(
            height=600,
            title=f"Parameter-Filtered Data: {parameter} in range [{parameter_range[0]:.2f}, {parameter_range[1]:.2f}]",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title="Data Source",
            showlegend=True,
        )

        # Store filtered data
        filtered_data_store = {
            "filtered_df": filtered_df.to_json(date_format="iso", orient="split"),
            "x_feature": x_feature,
            "y_feature": y_feature,
            "filtering_method": "parameter",
            "params": {"parameter": parameter, "range": parameter_range},
        }

        # Create info text
        file_counts = filtered_df["file_label"].value_counts().to_dict()
        info_text = [
            html.Div(
                f"Total points after filtering: {len(filtered_df)} ({len(filtered_df)/len(combined_df):.1%} of original)"
            ),
            html.Div(
                f"Filter: {parameter} in range [{parameter_range[0]:.2f}, {parameter_range[1]:.2f}]"
            ),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div(
                [
                    html.Div(f"{file}: {count} events")
                    for file, count in file_counts.items()
                ],
                style={"marginLeft": "10px"},
            ),
        ]

        return (
            fig,
            "Parameter filtering applied successfully!",
            filtered_data_store,
            info_text,
        )

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        print(f"Parameter filter error: {str(e)}\n{trace}")
        return (
            {},
            f"Error applying parameter filter: {str(e)}",
            {},
            [html.Div(f"Error: {str(e)}")],
        )


# Callback to apply UMAP density filter
@callback(
    Output("umap-filtered-data-graph", "figure", allow_duplicate=True),
    Output("umap-density-filter-status", "children"),
    Output("umap-density-filter-info", "children"),
    Output("umap-filtered-data-store", "data", allow_duplicate=True),
    Input("apply-umap-density-filter", "n_clicks"),
    State("umap-density-bandwidth-slider", "value"),
    State("umap-density-threshold-slider", "value"),
    State("combined-data-store", "data"),
    State("umap-graph", "figure"),
    prevent_initial_call=True,
)
def apply_umap_density_filter(
    n_clicks, bandwidth, threshold_percentile, combined_data_json, umap_figure
):
    ctx = dash.callback_context
    if (
        not ctx.triggered
        or "apply-umap-density-filter" not in ctx.triggered[0]["prop_id"]
    ):
        raise dash.exceptions.PreventUpdate

    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    print(
        f"UMAP density filter called with bandwidth={bandwidth}, threshold={threshold_percentile}"
    )

    try:
        # Get the UMAP coordinates
        if not combined_data_json or "umap_coords" not in combined_data_json:
            return (
                {},
                "No UMAP data available. Run UMAP first.",
                "No data available.",
                {},
            )

        # Load UMAP coordinates and combined dataframe
        umap_df = pd.read_json(combined_data_json["umap_coords"], orient="split")
        combined_df = None
        if combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(
                combined_data_json["combined_df"], orient="split"
            )

        if umap_df.empty:
            return {}, "No UMAP data available.", "No data available.", {}

        # Extract UMAP coordinates for density calculation
        umap_coords = umap_df[["UMAP1", "UMAP2"]].values

        # Fast grid-based density estimation
        # Create a 2D histogram (like a heatmap)
        num_bins = 100  # Adjust based on data size and desired granularity
        x_min, x_max = np.min(umap_coords[:, 0]), np.max(umap_coords[:, 0])
        y_min, y_max = np.min(umap_coords[:, 1]), np.max(umap_coords[:, 1])

        # Add small padding to min/max
        padding = 0.01
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(
            umap_coords[:, 0],
            umap_coords[:, 1],
            bins=num_bins,
            range=[[x_min, x_max], [y_min, y_max]],
        )

        # Apply smoothing - simple box blur
        from scipy.ndimage import gaussian_filter

        H_smooth = gaussian_filter(
            H, sigma=bandwidth * 10
        )  # Adjust multiplier as needed

        # Assign density values to each point
        x_indices = np.clip(
            np.floor((umap_coords[:, 0] - x_min) / (x_max - x_min) * num_bins).astype(
                int
            ),
            0,
            num_bins - 1,
        )
        y_indices = np.clip(
            np.floor((umap_coords[:, 1] - y_min) / (y_max - y_min) * num_bins).astype(
                int
            ),
            0,
            num_bins - 1,
        )

        densities = H_smooth[x_indices, y_indices]

        # Calculate density threshold
        threshold = np.percentile(densities, threshold_percentile)

        # Filter by density
        high_density_mask = densities >= threshold
        filtered_umap_df = umap_df.iloc[high_density_mask].copy()
        filtered_umap_df["density"] = densities[
            high_density_mask
        ]  # Store density for reference

        # If we have the original data, also filter that
        filtered_data_df = None
        if combined_df is not None:
            if len(combined_df) == len(umap_df):
                filtered_data_df = combined_df.iloc[high_density_mask].copy()

        print(
            f"Kept {len(filtered_umap_df)} high-density UMAP points ({len(filtered_umap_df)/len(umap_df):.1%} of original)"
        )

        # Check if we have any points left
        if len(filtered_umap_df) == 0:
            return (
                {},
                "No points remain after filtering.",
                "Try lowering the threshold.",
                {},
            )

        # Create visualization of filtered UMAP data
        fig = go.Figure()

        # Extract color information from original UMAP figure
        color_map = {}
        if umap_figure and "data" in umap_figure:
            for trace in umap_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    # Clean the label if it contains point count
                    clean_name = trace["name"]
                    if " (" in clean_name:
                        clean_name = clean_name.split(" (")[0]
                    color_map[clean_name] = trace["marker"]["color"]

        # Add original data as background with reduced opacity
        fig.add_trace(
            go.Scatter(
                x=umap_df["UMAP1"],
                y=umap_df["UMAP2"],
                mode="markers",
                marker=dict(size=4, color="gray", opacity=0.1),
                name="Original data (low density)",
                showlegend=True,
            )
        )

        # Add traces for each file label
        for label in filtered_umap_df["file_label"].unique():
            mask = filtered_umap_df["file_label"] == label
            df_subset = filtered_umap_df[mask]

            # Use color from original figure if available
            color = color_map.get(label, None)

            fig.add_trace(
                go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode="markers",
                    marker=dict(size=8, color=color, opacity=0.7),
                    name=f"{label} ({len(df_subset)} pts)",
                    showlegend=True,
                )
            )

        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"Density-Filtered UMAP (Kept {len(filtered_umap_df)} points, {len(filtered_umap_df)/len(umap_df):.1%} of "
            "original)",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data Source",
            showlegend=True,
        )

        # Store filtered data
        filtered_data_store = {
            "filtered_umap_df": filtered_umap_df.to_json(
                date_format="iso", orient="split"
            ),
            "filtered_data_df": (
                filtered_data_df.to_json(date_format="iso", orient="split")
                if filtered_data_df is not None
                else "{}"
            ),
            "filtering_method": "density",
            "params": {
                "bandwidth": bandwidth,
                "threshold_percentile": threshold_percentile,
                "threshold_value": float(threshold),
            },
        }

        # Create info text
        file_counts = filtered_umap_df["file_label"].value_counts().to_dict()
        info_text = [
            html.Div(
                f"Total points after filtering: {len(filtered_umap_df)} ({len(filtered_umap_df)/len(umap_df):.1%} of original)"
            ),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div(
                [
                    html.Div(f"{file}: {count} events")
                    for file, count in file_counts.items()
                ],
                style={"marginLeft": "10px"},
            ),
        ]

        return (
            fig,
            "UMAP density filtering applied successfully!",
            info_text,
            filtered_data_store,
        )

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        print(f"UMAP density filter error: {str(e)}\n{trace}")
        return (
            {},
            f"Error applying UMAP density filter: {str(e)}",
            f"Error: {str(e)}",
            {},
        )


# Callback to apply UMAP parameter filter
@callback(
    Output("umap-filtered-data-graph", "figure", allow_duplicate=True),
    Output("umap-parameter-filter-status", "children"),
    Output("umap-filtered-data-store", "data", allow_duplicate=True),
    Output("umap-filtered-data-info", "children", allow_duplicate=True),
    Input("apply-umap-parameter-filter", "n_clicks"),
    State("umap-physics-parameter-dropdown", "value"),
    State("umap-parameter-range-slider", "value"),
    State("combined-data-store", "data"),
    State("umap-graph", "figure"),
    prevent_initial_call=True,
)
def apply_umap_parameter_filter(
    n_clicks, parameter, parameter_range, combined_data_json, umap_figure
):
    ctx = dash.callback_context
    if (
        not ctx.triggered
        or "apply-umap-parameter-filter" not in ctx.triggered[0]["prop_id"]
    ):
        raise dash.exceptions.PreventUpdate

    print(
        f"UMAP parameter filter called with {parameter}=[{parameter_range[0]}, {parameter_range[1]}]"
    )

    try:
        # Get the UMAP coordinates and combined dataframe
        if (
            not combined_data_json
            or "umap_coords" not in combined_data_json
            or "combined_df" not in combined_data_json
        ):
            return (
                {},
                "No UMAP data available. Run UMAP first.",
                {},
                "No data available.",
            )

        # Load UMAP coordinates and combined dataframe
        umap_df = pd.read_json(combined_data_json["umap_coords"], orient="split")
        combined_df = None
        if combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(
                combined_data_json["combined_df"], orient="split"
            )
        else:
            return (
                {},
                "No original data available for filtering.",
                {},
                "No original data available.",
            )

        if umap_df.empty or combined_df is None or combined_df.empty:
            return (
                {},
                "No data available for UMAP parameter filtering.",
                {},
                "No data available.",
            )

        # Check if parameter exists in the data
        if parameter not in combined_df.columns:
            # Try to calculate physics features if needed
            try:
                combined_df = calculate_physics_features(combined_df)
            except:  # noqa: E722
                # TODO: Check why a bare except is needed here
                pass

            if parameter not in combined_df.columns:
                return (
                    {},
                    f"Parameter {parameter} not found in data.",
                    {},
                    "Parameter not found.",
                )

        # Filter by parameter range
        parameter_mask = (combined_df[parameter] >= parameter_range[0]) & (
            combined_df[parameter] <= parameter_range[1]
        )

        # Make sure we can apply the mask
        if len(parameter_mask) != len(umap_df):
            return (
                {},
                "UMAP and data dimensions don't match.",
                {},
                "Data mismatch error.",
            )

        # Apply filter
        filtered_data_df = combined_df.loc[parameter_mask].copy()
        filtered_umap_df = umap_df.loc[parameter_mask].copy()

        if len(filtered_umap_df) == 0:
            return (
                {},
                "No points remain after filtering.",
                {},
                "Try adjusting the parameter range.",
            )

        # Create visualization of filtered UMAP data
        fig = go.Figure()

        # Extract color information from original UMAP figure
        color_map = {}
        if umap_figure and "data" in umap_figure:
            for trace in umap_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    # Clean the label if it contains point count
                    clean_name = trace["name"]
                    if " (" in clean_name:
                        clean_name = clean_name.split(" (")[0]
                    color_map[clean_name] = trace["marker"]["color"]

        # Add original data as background with reduced opacity
        fig.add_trace(
            go.Scatter(
                x=umap_df["UMAP1"],
                y=umap_df["UMAP2"],
                mode="markers",
                marker=dict(size=4, color="gray", opacity=0.1),
                name="All UMAP points",
                showlegend=True,
            )
        )

        # Add traces for each file label in filtered data
        for label in filtered_umap_df["file_label"].unique():
            mask = filtered_umap_df["file_label"] == label
            df_subset = filtered_umap_df[mask]

            # Use color from original figure if available
            color = color_map.get(label, None)

            fig.add_trace(
                go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode="markers",
                    marker=dict(size=8, color=color, opacity=0.7),
                    name=f"{label} ({len(df_subset)} pts)",
                    showlegend=True,
                )
            )

        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"Parameter-Filtered UMAP: {parameter} in range [{parameter_range[0]:.2f}, {parameter_range[1]:.2f}]",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data Source",
            showlegend=True,
        )

        # Store filtered data
        filtered_data_store = {
            "filtered_umap_df": filtered_umap_df.to_json(
                date_format="iso", orient="split"
            ),
            "filtered_data_df": (
                filtered_data_df.to_json(date_format="iso", orient="split")
                if filtered_data_df is not None
                else "{}"
            ),
            "filtering_method": "parameter",
            "params": {"parameter": parameter, "range": parameter_range},
        }

        # Create info text
        file_counts = filtered_umap_df["file_label"].value_counts().to_dict()
        info_text = [
            html.Div(
                f"Total points after filtering: {len(filtered_umap_df)} ({len(filtered_umap_df)/len(umap_df):.1%} of original)"
            ),
            html.Div(
                f"Filter: {parameter} in range [{parameter_range[0]:.2f}, {parameter_range[1]:.2f}]"
            ),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div(
                [
                    html.Div(f"{file}: {count} events")
                    for file, count in file_counts.items()
                ],
                style={"marginLeft": "10px"},
            ),
        ]

        return (
            fig,
            "UMAP parameter filtering applied successfully!",
            filtered_data_store,
            info_text,
        )

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        print(f"UMAP parameter filter error: {str(e)}\n{trace}")
        return (
            {},
            f"Error applying UMAP parameter filter: {str(e)}",
            {},
            [html.Div(f"Error: {str(e)}")],
        )


# Callback to show/hide parameter filter controls
@callback(
    Output("parameter-filter-controls", "style"),
    Output("parameter-range-slider", "min"),
    Output("parameter-range-slider", "max"),
    Output("parameter-range-slider", "marks"),
    Output("parameter-range-slider", "value"),
    Input("physics-parameter-dropdown", "value"),
    State("scatter-graph15", "figure"),
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
    State("file-config-assignments-store", "data"),  # ADD THIS
    State("configuration-profiles-store", "data"),  # ADD THIS
)
def update_parameter_filter_controls(
    selected_parameter,
    figure,
    selected_ids,
    stored_files,
    assignments_store,
    profiles_store,
):  # ADD PARAMETERS
    """Update the parameter filter controls based on the selected parameter."""
    if not selected_parameter:
        return {"display": "none"}, 0, 100, {0: "0", 100: "100"}, [0, 100]

    # Get dataset parameter range
    try:
        # Load data to determine parameter range
        sampled_dfs = []
        for f in stored_files:
            if f["id"] in selected_ids:
                df = pd.read_json(f["data"], orient="split")
                is_selection = f.get("is_selection", False)

                # Only calculate features for non-selection files
                if not is_selection and selected_parameter not in df.columns:
                    # Check if physics features need to be calculated
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
                                print(
                                    f"Error calculating features for {f['filename']}: {e}"
                                )
                                continue  # Skip this file
                        else:
                            print(
                                f"Skipping {f['filename']} - no valid profile assigned"
                            )
                            continue  # Skip files without proper profile assignment

                # Only add to sampled_dfs if the parameter exists
                if selected_parameter in df.columns:
                    sampled_dfs.append(
                        df[[selected_parameter]]
                    )  # Only keep the needed column for efficiency

        if not sampled_dfs:
            return {"display": "block"}, 0, 100, {0: "0", 100: "100"}, [0, 100]

        # Combine only the parameter column from all dataframes
        combined_param = pd.concat(sampled_dfs, ignore_index=True)

        param_min = float(combined_param[selected_parameter].min())
        param_max = float(combined_param[selected_parameter].max())

        # Handle edge case where min equals max
        if param_min == param_max:
            param_min = param_min - 1
            param_max = param_max + 1

        # Round the values for better display
        param_min = np.floor(param_min)
        param_max = np.ceil(param_max)

        # Create marks dictionary
        num_steps = 5
        step_size = (param_max - param_min) / (num_steps - 1)
        marks = {}
        for i in range(num_steps):
            value = param_min + i * step_size
            marks[value] = f"{value:.1f}"

        return {"display": "block"}, param_min, param_max, marks, [param_min, param_max]

    except Exception as e:
        print(f"Error updating parameter filter controls: {e}")
        import traceback

        traceback.print_exc()
        return {"display": "block"}, 0, 100, {0: "0", 100: "100"}, [0, 100]


# Callback to show/hide UMAP parameter filter controls
@callback(
    Output("umap-parameter-filter-controls", "style"),
    Output("umap-parameter-range-slider", "min"),
    Output("umap-parameter-range-slider", "max"),
    Output("umap-parameter-range-slider", "marks"),
    Output("umap-parameter-range-slider", "value"),
    Input("umap-physics-parameter-dropdown", "value"),
    State("umap-file-selector", "value"),
    State("stored-files", "data"),
    State("combined-data-store", "data"),
    prevent_initial_call=True,
)
def update_umap_parameter_filter_controls(
    selected_parameter, selected_ids, stored_files, combined_data_json
):
    """Update the UMAP parameter filter controls based on the selected parameter."""
    if not selected_parameter:
        return {"display": "none"}, 0, 100, {0: "0", 100: "100"}, [0, 100]

    # Get dataset parameter range
    try:
        # We need to load the combined data from the UMAP visualization
        if (
            not combined_data_json
            or "combined_df" not in combined_data_json
            or combined_data_json["combined_df"] == "{}"
        ):
            return {"display": "block"}, 0, 100, {0: "0", 100: "100"}, [0, 100]

        combined_df = pd.read_json(combined_data_json["combined_df"], orient="split")

        if selected_parameter not in combined_df.columns:
            # Some physics parameters might need to be calculated
            try:
                combined_df = calculate_physics_features(combined_df)
            except:  # noqa: E722
                # TODO: Check why a bare except is needed here
                pass

        if selected_parameter not in combined_df.columns:
            return {"display": "block"}, 0, 100, {0: "0", 100: "100"}, [0, 100]

        param_min = float(combined_df[selected_parameter].min())
        param_max = float(combined_df[selected_parameter].max())

        # Round the values for better display
        param_min = np.floor(param_min)
        param_max = np.ceil(param_max)

        # Create marks dictionary - USING FEWER MARKS
        # Just show min, 25%, 50%, 75%, and max values
        steps = [0, 0.25, 0.5, 0.75, 1.0]
        marks = {}
        for step in steps:
            value = param_min + step * (param_max - param_min)
            marks[value] = f"{value:.1f}"

        return {"display": "block"}, param_min, param_max, marks, [param_min, param_max]

    except Exception as e:
        print(f"Error updating UMAP parameter filter controls: {e}")
        return {"display": "block"}, 0, 100, {0: "0", 100: "100"}, [0, 100]


# Parameter-filter-controls display callback
@callback(
    Output("parameter-filter-controls", "style", allow_duplicate=True),
    Input("physics-parameter-dropdown", "value"),
    prevent_initial_call=True,
)
def update_parameter_filter_controls_visibility(selected_parameter):
    """Show/hide parameter filter controls based on selection."""
    if not selected_parameter:
        return {"display": "none"}
    return {"display": "block"}


@callback(
    Output("parameter-range-slider", "min", allow_duplicate=True),
    Output("parameter-range-slider", "max", allow_duplicate=True),
    Output("parameter-range-slider", "marks", allow_duplicate=True),
    Output("parameter-range-slider", "value", allow_duplicate=True),
    Input("physics-parameter-dropdown", "value"),
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
    State("file-config-assignments-store", "data"),  # ADD THIS
    State("configuration-profiles-store", "data"),  # ADD THIS
    prevent_initial_call=True,
)
def update_parameter_filter_range(
    selected_parameter, selected_ids, stored_files, assignments_store, profiles_store
):  # ADD PARAMETERS
    """Update the parameter filter range based on the selected parameter."""
    if not selected_parameter:
        return 0, 100, {0: "0", 100: "100"}, [0, 100]

    # Load data to determine parameter range
    try:
        param_values = []  # Collect all parameter values

        for f in stored_files:
            if f["id"] in selected_ids:
                df = pd.read_json(f["data"], orient="split")
                is_selection = f.get("is_selection", False)

                # Only process non-selection files
                if not is_selection:
                    # Check if the parameter already exists
                    if selected_parameter not in df.columns:
                        # Check if physics features need to be calculated
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
                                    print(
                                        f"Error calculating features for {f['filename']}: {e}"
                                    )
                                    continue  # Skip this file
                            else:
                                print(
                                    f"Skipping {f['filename']} - no valid profile assigned"
                                )
                                continue  # Skip files without proper profile assignment

                    # Extract parameter values if they exist
                    if selected_parameter in df.columns:
                        param_values.extend(df[selected_parameter].dropna().tolist())

        if not param_values:
            return 0, 100, {0: "0", 100: "100"}, [0, 100]

        # Calculate min and max from all values
        param_min = float(min(param_values))
        param_max = float(max(param_values))

        # Handle edge case where min equals max
        if param_min == param_max:
            param_min = param_min - 1
            param_max = param_max + 1

        # Round the values for better display
        param_min = np.floor(param_min)
        param_max = np.ceil(param_max)

        # Create marks dictionary with 5 evenly spaced marks
        steps = [0, 0.25, 0.5, 0.75, 1.0]
        marks = {}
        for step in steps:
            value = param_min + step * (param_max - param_min)
            marks[value] = f"{value:.1f}"

        return param_min, param_max, marks, [param_min, param_max]

    except Exception as e:
        print(f"Error updating parameter range: {e}")
        import traceback

        traceback.print_exc()
        return 0, 100, {0: "0", 100: "100"}, [0, 100]
