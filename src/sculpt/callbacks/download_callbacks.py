import io

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc
from matplotlib.path import Path  # For lasso selection

from sculpt.utils.file_handlers import convert_to_original_format, get_particle_info
from sculpt.utils.metrics.physics_features import (
    calculate_physics_features_with_profile,
    has_physics_features,
)


# =============================================================================
# HELPER FUNCTION FOR FLEXIBLE PARTICLE EXPORT
# =============================================================================

def prepare_df_for_export(df, remove_umap=True, remove_file_label=True):
    """
    Prepare a dataframe for export by converting to original column format.
    
    Args:
        df: DataFrame with particle_X_Px/Py/Pz columns
        remove_umap: Whether to remove UMAP1/UMAP2 columns
        remove_file_label: Whether to remove file_label column
    
    Returns:
        DataFrame with original column names and physics features
    """
    export_df = df.copy()
    
    # Remove UMAP coordinates if requested
    if remove_umap:
        for col in ["UMAP1", "UMAP2"]:
            if col in export_df.columns:
                export_df = export_df.drop(columns=[col])
    
    # Remove file_label if requested
    if remove_file_label and "file_label" in export_df.columns:
        export_df = export_df.drop(columns=["file_label"])
    
    # Remove density column if present (added during filtering)
    if "density" in export_df.columns:
        export_df = export_df.drop(columns=["density"])
    
    # Get particle momentum columns
    momentum_columns = sorted([col for col in export_df.columns if col.startswith("particle_")])
    
    if not momentum_columns:
        return export_df  # No particle columns, return as-is
    
    # Get particle info if available
    particle_info = get_particle_info(df)
    
    # Convert to original format using the helper function
    original_format_df = convert_to_original_format(export_df, particle_info)
    
    # Add any physics features that aren't already included
    for col in export_df.columns:
        if not col.startswith("particle_") and col not in original_format_df.columns:
            if col not in ["UMAP1", "UMAP2", "file_label", "density"]:
                original_format_df[col] = export_df[col]
    
    return original_format_df


# =============================================================================
# GRAPH 3 SELECTION UMAP DOWNLOAD
# =============================================================================

@callback(
    Output("download-selection-graph3-selection", "data"),
    Input("save-selection-graph3-selection-btn", "n_clicks"),
    State("selection-graph3-selection-filename", "value"),
    State("graph3-selection-umap-store", "data"),
    prevent_initial_call=True,
)
def download_graph3_selection_points(n_clicks, filename, graph3_selection_umap_store):
    """Generate CSV file of points from the Graph 3 selection UMAP for download."""
    if not n_clicks or not filename or not graph3_selection_umap_store:
        raise dash.exceptions.PreventUpdate

    try:
        # Load the feature data from the store
        feature_data = pd.read_json(
            graph3_selection_umap_store["feature_data"], orient="split"
        )

        if feature_data.empty:
            raise dash.exceptions.PreventUpdate

        # Convert to original format for export
        original_format_df = prepare_df_for_export(feature_data)

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving Graph 3 selection UMAP data: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# =============================================================================
# GRAPH 3 (RE-RUN UMAP) DOWNLOAD
# =============================================================================

@callback(
    Output("download-selection-run", "data"),
    Input("save-selection-run-btn", "n_clicks"),
    State("selection-run-filename", "value"),
    State("selected-points-run-store", "data"),
    State("combined-data-store", "data"),
    prevent_initial_call=True,
)
def download_selected_points_run(n_clicks, filename, selectedData, combined_data_json):
    """Generate CSV file of selected points from Graph 3 for download in original format."""
    if not n_clicks or not filename or not selectedData or not combined_data_json:
        raise dash.exceptions.PreventUpdate

    try:
        # Load the combined dataframe
        combined_df = pd.DataFrame()
        if combined_data_json.get("combined_df", "{}") != "{}":
            combined_df = pd.read_json(
                combined_data_json["combined_df"], orient="split"
            )

        if combined_df.empty:
            raise dash.exceptions.PreventUpdate

        # Get indices of selected points
        indices = []

        # Handle box selection
        if "range" in selectedData:
            if "points" in selectedData:
                indices = [pt["pointIndex"] for pt in selectedData["points"]]

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            if "points" in selectedData:
                indices = [pt["pointIndex"] for pt in selectedData["points"]]

        # Handle direct point selection
        elif "points" in selectedData:
            indices = [pt["pointIndex"] for pt in selectedData["points"]]

        if not indices:
            raise dash.exceptions.PreventUpdate

        # Extract selected points - handle index bounds
        if len(indices) <= len(combined_df):
            selected_df = combined_df.iloc[indices].reset_index(drop=True)
        else:
            selected_df = combined_df.head(min(len(indices), len(combined_df)))

        # Convert to original format for export
        original_format_df = prepare_df_for_export(selected_df)

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving Graph 3 selection: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# =============================================================================
# LATENT FEATURES DOWNLOAD
# =============================================================================

@callback(
    Output("download-latent-features", "data"),
    Input("save-latent-features-btn", "n_clicks"),
    State("latent-features-filename", "value"),
    State("autoencoder-latent-store", "data"),
    prevent_initial_call=True,
)
def download_latent_features(n_clicks, filename, latent_store):
    """Generate CSV file of latent features for download."""
    if (
        not n_clicks
        or not filename
        or not latent_store
        or "latent_features" not in latent_store
    ):
        raise dash.exceptions.PreventUpdate

    try:
        # Load the latent features
        latent_df = pd.read_json(latent_store["latent_features"], orient="split")

        if latent_df.empty:
            raise dash.exceptions.PreventUpdate

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(latent_df.to_csv, f"{filename}.csv", index=False)

    except Exception as e:
        print(f"Error saving latent features: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# =============================================================================
# GRAPH 1 SELECTION DOWNLOAD
# =============================================================================

@callback(
    Output("download-selection", "data"),
    Input("save-selection-btn", "n_clicks"),
    State("selection-filename", "value"),
    State("selected-points-store", "data"),
    State("combined-data-store", "data"),
    prevent_initial_call=True,
)
def download_selected_points(n_clicks, filename, selectedData, combined_data_json):
    """Generate CSV file of selected points for download in original format."""
    if not n_clicks or not filename or not selectedData or not combined_data_json:
        raise dash.exceptions.PreventUpdate

    try:
        # Load the combined dataframe and UMAP coordinates
        combined_df = pd.DataFrame()
        if combined_data_json.get("combined_df", "{}") != "{}":
            combined_df = pd.read_json(
                combined_data_json["combined_df"], orient="split"
            )

        umap_coords = pd.read_json(
            combined_data_json["umap_coords"], orient="split"
        ).reset_index(drop=True)

        if combined_df.empty:
            raise dash.exceptions.PreventUpdate

        # Get indices of selected points
        indices = []

        # Handle box selection
        if "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]

            selected_mask = (
                (umap_coords["UMAP1"] >= x_range[0])
                & (umap_coords["UMAP1"] <= x_range[1])
                & (umap_coords["UMAP2"] >= y_range[0])
                & (umap_coords["UMAP2"] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            points_array = np.column_stack([umap_coords["UMAP1"], umap_coords["UMAP2"]])
            inside_lasso = lasso_path.contains_points(points_array)

            indices = np.where(inside_lasso)[0].tolist()

        # Handle direct point selection
        elif "points" in selectedData:
            indices = [pt["pointIndex"] for pt in selectedData["points"]]

        if not indices:
            raise dash.exceptions.PreventUpdate

        # Extract only the selected points from the combined dataframe
        selected_df = combined_df.iloc[indices].reset_index(drop=True)

        # Convert to original format for export
        original_format_df = prepare_df_for_export(selected_df)

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving selection: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# =============================================================================
# GENETIC FEATURES DOWNLOAD
# =============================================================================

@callback(
    Output("download-genetic-features", "data"),
    Input("save-genetic-features-btn", "n_clicks"),
    State("genetic-features-filename", "value"),
    State("genetic-features-store", "data"),
    prevent_initial_call=True,
)
def download_genetic_features(n_clicks, filename, genetic_features_store):
    """Generate CSV file of genetic features for download."""
    if (
        not n_clicks
        or not filename
        or not genetic_features_store
        or "genetic_features" not in genetic_features_store
    ):
        raise dash.exceptions.PreventUpdate

    try:
        # Load the genetic features
        gp_df = pd.read_json(genetic_features_store["genetic_features"], orient="split")

        if gp_df.empty:
            raise dash.exceptions.PreventUpdate

        # Create expressions text as header comment
        expressions = genetic_features_store.get("expressions", [])
        expressions_header = "# Generated features and their expressions:\n"
        for i, expr in enumerate(expressions):
            expressions_header += f"# GP_Feature_{i+1}: {expr}\n"

        # Write to string buffer with the expressions as header comment
        buffer = io.StringIO()
        buffer.write(expressions_header)
        gp_df.to_csv(buffer, index=False)

        # Return as download
        return dict(content=buffer.getvalue(), filename=f"{filename}.csv")

    except Exception as e:
        print(f"Error saving genetic features: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# =============================================================================
# MI FEATURES DOWNLOAD
# =============================================================================

@callback(
    Output("download-mi-features", "data"),
    Input("save-mi-features-btn", "n_clicks"),
    State("mi-features-filename", "value"),
    State("mi-features-store", "data"),
    prevent_initial_call=True,
)
def download_mi_features(n_clicks, filename, mi_features_store):
    """Generate CSV file of MI-selected features and latent representations for download."""
    if (
        not n_clicks
        or not filename
        or not mi_features_store
        or "latent_features" not in mi_features_store
    ):
        raise dash.exceptions.PreventUpdate

    try:
        # Load the latent features
        latent_df = pd.read_json(mi_features_store["latent_features"], orient="split")

        if latent_df.empty:
            raise dash.exceptions.PreventUpdate

        # Get selected features for header information
        selected_features = mi_features_store.get("selected_features", [])
        mi_scores = mi_features_store.get("mi_scores", {})

        # Create header with MI information
        mi_header = "# Mutual Information Feature Selection Results\n"
        mi_header += "# Selected Features and their MI scores:\n"

        for feature in selected_features:
            score = mi_scores.get(feature, 0.0)
            mi_header += f"# {feature}: {score:.6f}\n"

        mi_header += "#\n# Latent Features (from Autoencoder):\n"
        latent_dim = mi_features_store.get("latent_dim", 7)
        for i in range(latent_dim):
            mi_header += f"# Latent_{i}\n"

        # Write to string buffer with the MI information as header
        buffer = io.StringIO()
        buffer.write(mi_header)
        latent_df.to_csv(buffer, index=False)

        # Return as download
        return dict(content=buffer.getvalue(), filename=f"{filename}.csv")

    except Exception as e:
        print(f"Error saving MI features: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# =============================================================================
# GRAPH 1.5 (CUSTOM SCATTER PLOT) DOWNLOAD
# =============================================================================

@callback(
    Output("download-selection-graph15", "data"),
    Input("save-selection-graph15-btn", "n_clicks"),
    State("selection-filename-graph15", "value"),
    State("selected-points-store-graph15", "data"),
    State("x-axis-feature-graph15", "value"),
    State("y-axis-feature-graph15", "value"),
    State("scatter-graph15", "figure"),
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
    State("file-config-assignments-store", "data"),
    State("configuration-profiles-store", "data"),
    prevent_initial_call=True,
)
def download_selected_points_graph15(
    n_clicks,
    filename,
    selectedData,
    x_feature,
    y_feature,
    figure_data,
    selected_ids,
    stored_files,
    assignments_store,
    profiles_store,
):
    """Generate CSV file of selected points from Graph 1.5 for download."""
    if (
        not n_clicks
        or not filename
        or not selectedData
        or not selected_ids
        or not stored_files
    ):
        raise dash.exceptions.PreventUpdate

    try:
        # Build combined dataframe from selected files
        sampled_dfs = []
        for f in stored_files:
            if f["id"] in selected_ids and not f.get("is_selection", False):
                try:
                    df = pd.read_json(f["data"], orient="split")

                    # Get profile for this file if assigned
                    profile_name = (
                        assignments_store.get(f["filename"])
                        if assignments_store
                        else None
                    )
                    profile_config = (
                        profiles_store.get(profile_name) if profile_name else None
                    )

                    # Calculate physics features if profile available
                    if profile_config:
                        try:
                            df = calculate_physics_features_with_profile(
                                df, profile_config
                            )
                        except Exception as e:
                            print(f"Error calculating features: {e}")

                    # Add file label
                    df["file_label"] = f["filename"]
                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']} for download: {str(e)}")

        if not sampled_dfs:
            raise dash.exceptions.PreventUpdate

        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)

        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            raise dash.exceptions.PreventUpdate

        # Extract selected points
        selected_indices = []

        # Handle box selection
        if "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]

            selected_mask = (
                (combined_df[x_feature] >= x_range[0])
                & (combined_df[x_feature] <= x_range[1])
                & (combined_df[y_feature] >= y_range[0])
                & (combined_df[y_feature] <= y_range[1])
            )
            selected_indices = np.where(selected_mask)[0].tolist()

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            points_array = np.column_stack(
                [combined_df[x_feature].values, combined_df[y_feature].values]
            )
            inside_lasso = lasso_path.contains_points(points_array)

            selected_indices = np.where(inside_lasso)[0].tolist()

        # Handle direct point selection
        elif "points" in selectedData and selectedData["points"]:
            for point in selectedData["points"]:
                if "pointIndex" in point:
                    selected_indices.append(point["pointIndex"])
                elif "customdata" in point and point["customdata"]:
                    selected_indices.append(point["customdata"])
                else:
                    x_val = point.get("x")
                    y_val = point.get("y")
                    if x_val is not None and y_val is not None:
                        distances = (combined_df[x_feature] - x_val) ** 2 + (
                            combined_df[y_feature] - y_val
                        ) ** 2
                        closest_idx = distances.idxmin()
                        selected_indices.append(closest_idx)

        if not selected_indices:
            raise dash.exceptions.PreventUpdate

        # Extract the selected points
        selected_df = combined_df.iloc[selected_indices].reset_index(drop=True)

        # Convert to original format for export
        original_format_df = prepare_df_for_export(selected_df)

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving Graph 1.5 selection: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# =============================================================================
# GRAPH 2.5 DOWNLOAD
# =============================================================================

@callback(
    Output("download-selection-graph25", "data"),
    Input("save-selection-graph25-btn", "n_clicks"),
    State("selection-filename-graph25", "value"),
    State("selected-points-store-graph15", "data"),
    State("x-axis-feature-graph15", "value"),
    State("y-axis-feature-graph15", "value"),
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
    State("file-config-assignments-store", "data"),
    State("configuration-profiles-store", "data"),
    prevent_initial_call=True,
)
def download_selected_points_graph25(
    n_clicks,
    filename,
    selectedData,
    x_feature,
    y_feature,
    selected_ids,
    stored_files,
    assignments_store,
    profiles_store,
):
    """Generate CSV file of selected points from Graph 2.5 for download."""
    if (
        not n_clicks
        or not filename
        or not selectedData
        or not selected_ids
        or not stored_files
    ):
        raise dash.exceptions.PreventUpdate

    try:
        # Build combined dataframe from selected files
        sampled_dfs = []
        for f in stored_files:
            if f["id"] in selected_ids and not f.get("is_selection", False):
                try:
                    df = pd.read_json(f["data"], orient="split")

                    # Get profile for this file if assigned
                    profile_name = (
                        assignments_store.get(f["filename"])
                        if assignments_store
                        else None
                    )
                    profile_config = (
                        profiles_store.get(profile_name) if profile_name else None
                    )

                    # Calculate physics features if profile available
                    if profile_config:
                        try:
                            df = calculate_physics_features_with_profile(
                                df, profile_config
                            )
                        except Exception as e:
                            print(f"Error calculating features: {e}")

                    df["file_label"] = f["filename"]
                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']} for download: {str(e)}")

        if not sampled_dfs:
            raise dash.exceptions.PreventUpdate

        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)

        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            raise dash.exceptions.PreventUpdate

        # Extract selected points
        selected_indices = []

        # Handle box selection
        if "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]

            selected_mask = (
                (combined_df[x_feature] >= x_range[0])
                & (combined_df[x_feature] <= x_range[1])
                & (combined_df[y_feature] >= y_range[0])
                & (combined_df[y_feature] <= y_range[1])
            )
            selected_indices = np.where(selected_mask)[0].tolist()

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            points_array = np.column_stack(
                [combined_df[x_feature].values, combined_df[y_feature].values]
            )
            inside_lasso = lasso_path.contains_points(points_array)

            selected_indices = np.where(inside_lasso)[0].tolist()

        # Handle direct point selection
        elif "points" in selectedData and selectedData["points"]:
            for point in selectedData["points"]:
                x_val = point.get("x")
                y_val = point.get("y")
                if x_val is not None and y_val is not None:
                    distances = (combined_df[x_feature] - x_val) ** 2 + (
                        combined_df[y_feature] - y_val
                    ) ** 2
                    closest_idx = distances.idxmin()
                    selected_indices.append(closest_idx)

        if not selected_indices:
            raise dash.exceptions.PreventUpdate

        # Extract the selected points
        selected_df = combined_df.iloc[selected_indices].reset_index(drop=True)

        # Convert to original format for export
        original_format_df = prepare_df_for_export(selected_df)

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving Graph 2.5 selection: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# =============================================================================
# FILTERED DATA DOWNLOAD
# =============================================================================

@callback(
    Output("download-filtered-data", "data"),
    Output("save-filtered-data-status", "children"),
    Input("save-filtered-data-btn", "n_clicks"),
    State("filtered-data-filename", "value"),
    State("filtered-data-store", "data"),
    State("file-config-assignments-store", "data"),
    State("configuration-profiles-store", "data"),
    prevent_initial_call=True,
)
def download_filtered_data(
    n_clicks, filename, filtered_data_store, assignments_store, profiles_store
):
    """Generate CSV file of filtered data points for download."""
    if (
        not n_clicks
        or not filename
        or not filtered_data_store
        or "filtered_df" not in filtered_data_store
    ):
        raise dash.exceptions.PreventUpdate

    try:
        # Load the filtered data
        filtered_df = pd.read_json(filtered_data_store["filtered_df"], orient="split")

        if filtered_df.empty:
            raise dash.exceptions.PreventUpdate

        # Convert to original format for export
        original_format_df = prepare_df_for_export(filtered_df)

        # Ensure no duplicate columns
        original_format_df = original_format_df.loc[
            :, ~original_format_df.columns.duplicated()
        ]

        # Return the dataframe as a CSV for download
        return (
            dcc.send_data_frame(
                original_format_df.to_csv, f"{filename}.csv", index=False
            ),
            f"Successfully saved {len(original_format_df)} filtered points to {filename}.csv",
        )

    except Exception as e:
        print(f"Error saving filtered data: {e}")
        import traceback
        traceback.print_exc()
        return dash.no_update, f"Error saving filtered data: {str(e)}"


# =============================================================================
# UMAP FILTERED DATA DOWNLOAD
# =============================================================================


@callback(
    Output("download-umap-filtered-data", "data"),
    Output("save-umap-filtered-data-status", "children"),
    Input("save-umap-filtered-data-btn", "n_clicks"),
    State("umap-filtered-data-filename", "value"),
    State("umap-filtered-data-store", "data"),
    prevent_initial_call=True,
)
def download_umap_filtered_data(n_clicks, filename, filtered_data_store):
    """Generate CSV file of filtered UMAP data points for download.
    
    Always includes momentum data (converted to original column names) so that
    re-uploaded files can have physics features recalculated. UMAP coordinates
    and file_label are also preserved so the file is recognized as a selection
    file on re-upload.
    """
    if not n_clicks or not filename or not filtered_data_store:
        raise dash.exceptions.PreventUpdate

    try:
        # Prefer the full combined data (has momentum + physics features)
        if (
            "filtered_data_df" in filtered_data_store
            and filtered_data_store["filtered_data_df"] != "{}"
        ):
            filtered_df = pd.read_json(
                filtered_data_store["filtered_data_df"], orient="split"
            )
        elif "filtered_umap_df" in filtered_data_store:
            # Fallback to UMAP coordinates dataframe
            filtered_df = pd.read_json(
                filtered_data_store["filtered_umap_df"], orient="split"
            )
        else:
            raise dash.exceptions.PreventUpdate

        if filtered_df.empty:
            raise dash.exceptions.PreventUpdate

        # Always export with full data — convert particle columns to original names
        # but KEEP UMAP1, UMAP2, and file_label so the file is recognized as a
        # selection file on re-upload and can be overlaid on UMAP plots.
        export_df = prepare_df_for_export(
            filtered_df, remove_umap=False, remove_file_label=False
        )

        # Remove density column if present (internal use only)
        if "density" in export_df.columns:
            export_df = export_df.drop(columns=["density"])

        # Return the dataframe as a CSV for download
        return (
            dcc.send_data_frame(export_df.to_csv, f"{filename}.csv", index=False),
            f"Successfully saved {len(export_df)} filtered UMAP points to {filename}.csv",
        )

    except Exception as e:
        print(f"Error saving filtered UMAP data: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error saving filtered UMAP data: {str(e)}"