import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc
from matplotlib.path import Path  # For lasso selection

from sculpt.utils.metrics.physics_features import calculate_physics_features


# Callback to handle the download of selected points from the new graph
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

        # Process the data for saving
        # Remove UMAP coordinates if they exist
        if "UMAP1" in feature_data.columns:
            feature_data = feature_data.drop(columns=["UMAP1"])
        if "UMAP2" in feature_data.columns:
            feature_data = feature_data.drop(columns=["UMAP2"])

        # The file_label column would have been added during processing, so remove it
        if "file_label" in feature_data.columns:
            feature_data = feature_data.drop(columns=["file_label"])

        # Get only the particle momentum columns (original data format)
        momentum_columns = [
            col for col in feature_data.columns if col.startswith("particle_")
        ]

        # If we have the momentum columns, convert back to original format
        if (
            momentum_columns and len(momentum_columns) == 15
        ):  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                "Px_ion1",
                "Py_ion1",
                "Pz_ion1",
                "Px_ion2",
                "Py_ion2",
                "Pz_ion2",
                "Px_neutral",
                "Py_neutral",
                "Pz_neutral",
                "Px_electron1",
                "Py_electron1",
                "Pz_electron1",
                "Px_electron2",
                "Py_electron2",
                "Pz_electron2",
            ]

            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = feature_data[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = feature_data

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving Graph 3 selection UMAP data: {e}")
        import traceback

        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# Callback to handle the download of selected points from Graph 3
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
        # Load the combined dataframe and UMAP coordinates
        combined_df = pd.DataFrame()
        if combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(
                combined_data_json["combined_df"], orient="split"
            )

        if combined_df.empty:
            raise dash.exceptions.PreventUpdate

        # Get indices of selected points
        indices = []

        # Handle box selection
        if "range" in selectedData:
            # We can't directly use combined_data_store because the points in Graph 3
            # have different UMAP coordinates after re-running UMAP
            # Instead, we'll extract the points directly from the selection
            if "points" in selectedData:
                indices = [pt["pointIndex"] for pt in selectedData["points"]]

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            # Extract directly from points for Graph 3
            if "points" in selectedData:
                indices = [pt["pointIndex"] for pt in selectedData["points"]]

        # Handle direct point selection
        elif "points" in selectedData:
            indices = [pt["pointIndex"] for pt in selectedData["points"]]

        if not indices:
            raise dash.exceptions.PreventUpdate

        # Extract only the selected points from the combined dataframe
        # For Graph 3, we need to be careful because the indices in the re-run
        # correspond to points that were already selected from Graph 1
        # We need to get the original indices from the first selection

        # If we have fewer points in combined_df than our indices, it means
        # we're working with a subset already
        if len(indices) <= len(combined_df):
            selected_df = combined_df.iloc[indices].reset_index(drop=True)
        else:
            # This is a fallback if something is wrong with the indices
            # Just take the first N points where N is the number of indices
            selected_df = combined_df.head(min(len(indices), len(combined_df)))

        # Remove UMAP coordinates if they exist
        if "UMAP1" in selected_df.columns:
            selected_df = selected_df.drop(columns=["UMAP1"])
        if "UMAP2" in selected_df.columns:
            selected_df = selected_df.drop(columns=["UMAP2"])

        # Convert back to original format with standardized column names
        # The file_label column would have been added during processing, so remove it
        if "file_label" in selected_df.columns:
            selected_df = selected_df.drop(columns=["file_label"])

        # Get only the particle momentum columns (original data format)
        momentum_columns = [
            col for col in selected_df.columns if col.startswith("particle_")
        ]

        # If we have the momentum columns, convert back to original format
        if (
            momentum_columns and len(momentum_columns) == 15
        ):  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                "Px_ion1",
                "Py_ion1",
                "Pz_ion1",
                "Px_ion2",
                "Py_ion2",
                "Pz_ion2",
                "Px_neutral",
                "Py_neutral",
                "Pz_neutral",
                "Px_electron1",
                "Py_electron1",
                "Pz_electron1",
                "Px_electron2",
                "Py_electron2",
                "Pz_electron2",
            ]

            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = selected_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = selected_df

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving Graph 3 selection: {e}")
        import traceback

        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# Callback to save latent features
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


# Callback to handle the download of selected points
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
        umap_coords = pd.read_json(combined_data_json["umap_coords"], orient="split")
        combined_df = pd.DataFrame()
        if combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(
                combined_data_json["combined_df"], orient="split"
            )

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

        # Handle direct point selection
        elif "points" in selectedData:
            indices = [pt["pointIndex"] for pt in selectedData["points"]]

        if not indices:
            raise dash.exceptions.PreventUpdate

        # Extract only the selected points from the combined dataframe
        selected_df = combined_df.iloc[indices].reset_index(drop=True)

        # Remove UMAP coordinates if they exist
        if "UMAP1" in selected_df.columns:
            selected_df = selected_df.drop(columns=["UMAP1"])
        if "UMAP2" in selected_df.columns:
            selected_df = selected_df.drop(columns=["UMAP2"])

        # Convert back to original format with standardized column names
        # The file_label column would have been added during processing, so remove it
        if "file_label" in selected_df.columns:
            selected_df = selected_df.drop(columns=["file_label"])

        # Get only the particle momentum columns (original data format)
        momentum_columns = [
            col for col in selected_df.columns if col.startswith("particle_")
        ]

        # If we have the momentum columns, convert back to original format
        if (
            momentum_columns and len(momentum_columns) == 15
        ):  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                "Px_ion1",
                "Py_ion1",
                "Pz_ion1",
                "Px_ion2",
                "Py_ion2",
                "Pz_ion2",
                "Px_neutral",
                "Py_neutral",
                "Pz_neutral",
                "Px_electron1",
                "Py_electron1",
                "Pz_electron1",
                "Px_electron2",
                "Py_electron2",
                "Pz_electron2",
            ]

            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = selected_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = selected_df

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving selection: {e}")
        import traceback

        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


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
        import io

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
        import io

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


# Callback to download selected points from Graph 1.5
@callback(
    Output("download-selection-graph15", "data"),
    Input("save-selection-graph15-btn", "n_clicks"),
    State("selection-filename-graph15", "value"),
    State("selected-points-store-graph15", "data"),
    State("x-axis-feature-graph15", "value"),
    State("y-axis-feature-graph15", "value"),
    State("scatter-graph15", "figure"),  # Add the figure data
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
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
):
    """Generate CSV file of selected points from Graph 1.5 for download."""
    if not n_clicks or not filename or not selectedData:
        raise dash.exceptions.PreventUpdate

    print(f"Save button clicked for Graph 1.5, filename: {filename}")

    try:
        # Process selected files to get the combined dataframe
        sampled_dfs = []

        # This part is different from Graph 1 - need to build combined dataset
        for f in stored_files:
            if f["id"] in selected_ids:
                try:
                    df = pd.read_json(f["data"], orient="split")
                    is_selection = f.get("is_selection", False)

                    df["file_label"] = f["filename"]  # Add file name as a label

                    if not is_selection:
                        # Calculate physics features
                        df = calculate_physics_features(df)

                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']} for download: {str(e)}")

        # Combine datasets
        if not sampled_dfs:
            print("No valid files selected")
            raise dash.exceptions.PreventUpdate

        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        print(f"Combined dataframe shape: {combined_df.shape}")

        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            print(f"Features {x_feature} or {y_feature} not found in data")
            raise dash.exceptions.PreventUpdate

        # Extract selected points using the coordinates - this is the key part
        selected_indices = []

        # Print debug info about selectedData
        print(f"Selected data type: {type(selectedData)}")
        print(f"Selected data content: {selectedData}")

        # Handle box selection
        if "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]

            print(
                f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"
            )

            # Use masked selection
            selected_mask = (
                (combined_df[x_feature] >= x_range[0])
                & (combined_df[x_feature] <= x_range[1])
                & (combined_df[y_feature] >= y_range[0])
                & (combined_df[y_feature] <= y_range[1])
            )
            selected_indices = np.where(selected_mask)[0].tolist()
            print(f"Found {len(selected_indices)} points in box selection")

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            from matplotlib.path import Path

            # Extract lasso polygon coordinates
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            print(f"Lasso selection with {len(lasso_x)} points")

            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

            # Check which points are within the lasso path
            points_array = np.column_stack(
                [combined_df[x_feature].values, combined_df[y_feature].values]
            )
            inside_lasso = lasso_path.contains_points(points_array)

            # Get indices of points inside the lasso
            selected_indices = np.where(inside_lasso)[0].tolist()
            print(f"Found {len(selected_indices)} points in lasso selection")

        # Handle direct point selection
        elif "points" in selectedData and selectedData["points"]:
            print(f"Direct selection with {len(selectedData['points'])} points")

            # Try to extract points directly from the selection
            for point in selectedData["points"]:
                print(f"Point data: {point}")
                # Try different ways to get index
                if "pointIndex" in point:
                    selected_indices.append(point["pointIndex"])
                elif "customdata" in point and point["customdata"]:
                    # Some plots store index in customdata
                    selected_indices.append(point["customdata"])
                else:
                    # Find by coordinates
                    x_val = point.get("x")
                    y_val = point.get("y")

                    if x_val is not None and y_val is not None:
                        # Find closest point
                        distances = (combined_df[x_feature] - x_val) ** 2 + (
                            combined_df[y_feature] - y_val
                        ) ** 2
                        closest_idx = distances.idxmin()
                        selected_indices.append(closest_idx)

        if not selected_indices:
            print("No valid indices found")
            raise dash.exceptions.PreventUpdate

        print(f"Processing {len(selected_indices)} selected points")

        # Extract the selected points
        selected_df = combined_df.iloc[selected_indices].reset_index(drop=True)

        # Remove UMAP coordinates if they exist
        if "UMAP1" in selected_df.columns:
            selected_df = selected_df.drop(columns=["UMAP1"])
        if "UMAP2" in selected_df.columns:
            selected_df = selected_df.drop(columns=["UMAP2"])

        # Remove file_label
        if "file_label" in selected_df.columns:
            selected_df = selected_df.drop(columns=["file_label"])

        # Get only the particle momentum columns (original data format)
        momentum_columns = [
            col for col in selected_df.columns if col.startswith("particle_")
        ]

        # Convert to original format
        if (
            momentum_columns and len(momentum_columns) == 15
        ):  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                "Px_ion1",
                "Py_ion1",
                "Pz_ion1",
                "Px_ion2",
                "Py_ion2",
                "Pz_ion2",
                "Px_neutral",
                "Py_neutral",
                "Pz_neutral",
                "Px_electron1",
                "Py_electron1",
                "Pz_electron1",
                "Px_electron2",
                "Py_electron2",
                "Pz_electron2",
            ]

            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = selected_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = selected_df

        print(f"Final export dataframe shape: {original_format_df.shape}")
        print(f"Final export dataframe columns: {original_format_df.columns.tolist()}")

        # Return the dataframe as a CSV for download
        # The key difference is we're only returning the download output here
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving Graph 1.5 selection: {e}")
        import traceback

        trace = traceback.format_exc()
        print(trace)
        raise dash.exceptions.PreventUpdate


@callback(
    Output("download-selection-graph25", "data"),
    Input("save-selection-graph25-btn", "n_clicks"),
    State("selection-filename-graph25", "value"),
    State("selected-points-store-graph15", "data"),  # Use the data from Graph 1.5
    State("x-axis-feature-graph15", "value"),
    State("y-axis-feature-graph15", "value"),
    State("file-selector-graph15", "value"),
    State("stored-files", "data"),
    prevent_initial_call=True,
)
def download_selected_points_graph25(
    n_clicks, filename, selectedData, x_feature, y_feature, selected_ids, stored_files
):
    """Generate CSV file of selected points from Graph 2.5 for download."""
    if not n_clicks or not filename or not selectedData:
        raise dash.exceptions.PreventUpdate

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
                        # Calculate physics features
                        df = calculate_physics_features(df)

                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']} for download: {str(e)}")

        # Combine datasets
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
            from matplotlib.path import Path

            # Extract lasso polygon coordinates
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

            # Check which points are within the lasso path
            points_array = np.column_stack(
                [combined_df[x_feature].values, combined_df[y_feature].values]
            )
            inside_lasso = lasso_path.contains_points(points_array)

            # Get indices of points inside the lasso
            selected_indices = np.where(inside_lasso)[0].tolist()

        # Handle direct point selection
        elif "points" in selectedData and selectedData["points"]:
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
            raise dash.exceptions.PreventUpdate

        # Extract the selected points
        selected_df = combined_df.iloc[selected_indices].reset_index(drop=True)

        # Remove UMAP coordinates if they exist
        if "UMAP1" in selected_df.columns:
            selected_df = selected_df.drop(columns=["UMAP1"])
        if "UMAP2" in selected_df.columns:
            selected_df = selected_df.drop(columns=["UMAP2"])

        # Keep file_label for reference but prepare for export
        export_df = selected_df.copy()

        # Remove file_label which was added during processing
        if "file_label" in export_df.columns:
            export_df = export_df.drop(columns=["file_label"])

        # Get only the particle momentum columns (original data format)
        momentum_columns = [
            col for col in export_df.columns if col.startswith("particle_")
        ]

        # If we have the momentum columns, convert back to original format
        if (
            momentum_columns and len(momentum_columns) == 15
        ):  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                "Px_ion1",
                "Py_ion1",
                "Pz_ion1",
                "Px_ion2",
                "Py_ion2",
                "Pz_ion2",
                "Px_neutral",
                "Py_neutral",
                "Pz_neutral",
                "Px_electron1",
                "Py_electron1",
                "Pz_electron1",
                "Px_electron2",
                "Py_electron2",
                "Pz_electron2",
            ]

            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = export_df[col]

            # Also include other physics features if available
            for col in export_df.columns:
                if not col.startswith("particle_"):
                    original_format_df[col] = export_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = export_df

        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, f"{filename}.csv", index=False
        )

    except Exception as e:
        print(f"Error saving Graph 2.5 selection: {e}")
        import traceback

        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


# Callback to download filtered data
@callback(
    Output("download-filtered-data", "data"),
    Output("save-filtered-data-status", "children"),
    Input("save-filtered-data-btn", "n_clicks"),
    State("filtered-data-filename", "value"),
    State("filtered-data-store", "data"),
    prevent_initial_call=True,
)
def download_filtered_data(n_clicks, filename, filtered_data_store):
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

        # Remove columns not needed for export
        export_df = filtered_df.copy()

        # Remove density column which was added during filtering
        if "density" in export_df.columns:
            export_df = export_df.drop(columns=["density"])

        # Remove file_label which was added during processing
        if "file_label" in export_df.columns:
            export_df = export_df.drop(columns=["file_label"])

        # Get only the particle momentum columns (original data format)
        momentum_columns = [
            col for col in export_df.columns if col.startswith("particle_")
        ]

        # If we have the momentum columns, convert back to original format
        original_format_df = pd.DataFrame()
        if momentum_columns and len(momentum_columns) == 15:
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                "Px_ion1",
                "Py_ion1",
                "Pz_ion1",
                "Px_ion2",
                "Py_ion2",
                "Pz_ion2",
                "Px_neutral",
                "Py_neutral",
                "Pz_neutral",
                "Px_electron1",
                "Py_electron1",
                "Pz_electron1",
                "Px_electron2",
                "Py_electron2",
                "Pz_electron2",
            ]

            # Extract and rename the momentum columns
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = export_df[col]

            # Also include other physics features if available
            for col in export_df.columns:
                if not col.startswith("particle_"):
                    original_format_df[col] = export_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = export_df

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
        return None, f"Error saving filtered data: {str(e)}"


# Callback to download filtered UMAP data
@callback(
    Output("download-umap-filtered-data", "data"),
    Output("save-umap-filtered-data-status", "children"),
    Input("save-umap-filtered-data-btn", "n_clicks"),
    State("umap-filtered-data-filename", "value"),
    State("umap-filtered-data-store", "data"),
    prevent_initial_call=True,
)
def download_umap_filtered_data(n_clicks, filename, filtered_data_store):
    """Generate CSV file of filtered UMAP data points for download."""
    if not n_clicks or not filename or not filtered_data_store:
        raise dash.exceptions.PreventUpdate

    try:
        # Load the filtered data
        if (
            "filtered_data_df" in filtered_data_store
            and filtered_data_store["filtered_data_df"] != "{}"
        ):
            # Use the original data if available for more complete information
            filtered_df = pd.read_json(
                filtered_data_store["filtered_data_df"], orient="split"
            )
        elif "filtered_umap_df" in filtered_data_store:
            # Fallback to UMAP coordinates
            filtered_df = pd.read_json(
                filtered_data_store["filtered_umap_df"], orient="split"
            )
        else:
            raise dash.exceptions.PreventUpdate

        if filtered_df.empty:
            raise dash.exceptions.PreventUpdate

        # Remove columns not needed for export
        export_df = filtered_df.copy()

        # Remove density column which was added during filtering
        if "density" in export_df.columns:
            export_df = export_df.drop(columns=["density"])

        # If it's UMAP data only, keep the important columns
        if (
            "filtered_data_df" not in filtered_data_store
            or filtered_data_store["filtered_data_df"] == "{}"
        ):
            important_cols = ["UMAP1", "UMAP2", "file_label"]
            export_df = export_df[important_cols]
        else:
            # For full data, prepare for proper export format
            # Remove file_label which was added during processing
            if "file_label" in export_df.columns:
                export_df = export_df.drop(columns=["file_label"])

            # Get only the particle momentum columns (original data format)
            momentum_columns = [
                col for col in export_df.columns if col.startswith("particle_")
            ]

            # If we have the momentum columns, convert back to original format
            if (
                momentum_columns and len(momentum_columns) == 15
            ):  # Should be 5 particles x 3 dimensions
                # Create the reverse mapping from standardized to original column names
                reverse_columns = [
                    "Px_ion1",
                    "Py_ion1",
                    "Pz_ion1",
                    "Px_ion2",
                    "Py_ion2",
                    "Pz_ion2",
                    "Px_neutral",
                    "Py_neutral",
                    "Pz_neutral",
                    "Px_electron1",
                    "Py_electron1",
                    "Pz_electron1",
                    "Px_electron2",
                    "Py_electron2",
                    "Pz_electron2",
                ]

                # Extract and rename the momentum columns
                original_format_df = pd.DataFrame()
                for i, col in enumerate(momentum_columns):
                    if i < len(reverse_columns):
                        original_format_df[reverse_columns[i]] = export_df[col]

                # Also include other physics features if available
                for col in export_df.columns:
                    if not col.startswith("particle_") and col not in [
                        "UMAP1",
                        "UMAP2",
                    ]:
                        original_format_df[col] = export_df[col]

                # Use this for export
                export_df = original_format_df

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
