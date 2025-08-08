import re
import traceback

import dash
import numpy as np
import pandas as pd

# import plotly.graph_objects as go
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import umap.umap_ as umap
from dash import ALL, Input, Output, State, callback, html

# from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import mutual_info_regression

# from itertools import combinations


# from sklearn.metrics import (
#     calinski_harabasz_score,
#     davies_bouldin_score,
#     silhouette_score,
# )
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset

# from sculpt.models.deep_autoencoder import DeepAutoencoder
# from sculpt.utils.file_handlers import extract_selection_indices
# from sculpt.utils.metrics.clustering_quality import cluster_stability, hopkins_statistic


@callback(
    Output("mi-features-store", "data"),
    Output("mi-features-status", "children", allow_duplicate=True),
    Output("mi-scatter-x-feature", "options"),
    Output("mi-scatter-y-feature", "options"),
    Output("mi-sorted-features-info", "children"),
    Input("run-mi-features", "n_clicks"),
    State("mi-data-source", "value"),
    State("mi-target-variables", "value"),
    State("mi-redundancy-threshold", "value"),
    State("mi-max-features", "value"),
    State({"type": "feature-selector-mi", "category": ALL}, "value"),
    State("selected-points-store", "data"),
    State("selected-points-run-store", "data"),
    State("combined-data-store", "data"),
    prevent_initial_call=True,
)
def run_mi_feature_selection(
    n_clicks,
    data_source,
    target_variables,
    redundancy_threshold,
    max_features,
    selected_features_list,
    graph1_selection,
    graph3_selection,
    combined_data_json,
):
    """Run mutual information feature selection to identify the most informative features."""

    print("\n" + "=" * 80)
    print("DEBUG: STARTING MI FEATURE SELECTION")
    print("=" * 80)

    if not n_clicks:
        return {}, "Click 'Run MI Feature Selection' to start.", [], [], ""

    try:
        debug_text = []

        # Debug: Print all inputs
        print(f"DEBUG: n_clicks = {n_clicks}")
        print(f"DEBUG: data_source = {data_source}")
        print(f"DEBUG: target_variables = {target_variables}")
        print(f"DEBUG: redundancy_threshold = {redundancy_threshold}")
        print(f"DEBUG: max_features = {max_features}")
        print(f"DEBUG: selected_features_list = {selected_features_list}")
        print(f"DEBUG: graph1_selection = {graph1_selection}")
        print(f"DEBUG: graph3_selection = {graph3_selection}")

        # Debug: Check combined_data_json
        print("\nDEBUG: Checking combined_data_json...")
        print(f"  - Type: {type(combined_data_json)}")
        print(f"  - Is None: {combined_data_json is None}")
        print(f"  - Is Dict: {isinstance(combined_data_json, dict)}")

        if combined_data_json is not None:
            print(
                f"  - Keys: {list(combined_data_json.keys()) if isinstance(combined_data_json, dict) else 'N/A'}"
            )
            if isinstance(combined_data_json, dict):
                for key, value in combined_data_json.items():
                    print(
                        f"  - {key}: type={type(value)}, len={len(str(value)) if value else 0}"
                    )

        debug_text.append(f"Data source: {data_source}")
        debug_text.append(f"Target variables: {target_variables}")
        debug_text.append(f"Redundancy threshold: {redundancy_threshold}")
        debug_text.append(f"Maximum features: {max_features}")

        # Step 1: Validate inputs
        # -----------------------------------------

        # Default values if not provided
        if not data_source:
            data_source = "all"
        if not target_variables or len(target_variables) == 0:
            target_variables = ["KER", "EESum", "EESharing", "TotalEnergy"]
            debug_text.append(
                f"Using default target variables: {', '.join(target_variables)}"
            )
        if redundancy_threshold is None:
            redundancy_threshold = 0.5
        if max_features is None:
            max_features = 20

        # Step 2: Load and prepare data
        # -----------------------------------------

        print("\nDEBUG: Loading combined dataset...")

        # Check if combined_data_json exists and has the right structure
        if combined_data_json is None:
            error_msg = (
                "No combined data available. Please run UMAP in Basic Analysis first."
            )
            print(f"DEBUG ERROR: {error_msg}")
            return {}, error_msg, [], [], ""

        if not isinstance(combined_data_json, dict):
            error_msg = f"Invalid combined data format. Expected dict, got {type(combined_data_json)}"
            print(f"DEBUG ERROR: {error_msg}")
            return {}, error_msg, [], [], ""

        if "combined_df" not in combined_data_json:
            error_msg = "No 'combined_df' in data store. Please run UMAP in Basic Analysis first."
            print(f"DEBUG ERROR: {error_msg}")
            return {}, error_msg, [], [], ""

        # Check if combined_df is empty
        combined_df_data = combined_data_json.get("combined_df")
        print(f"DEBUG: combined_df_data type: {type(combined_df_data)}")
        print(
            f"DEBUG: combined_df_data content (first 100 chars): {str(combined_df_data)[:100] if combined_df_data else 'None'}"
        )

        if (
            combined_df_data is None
            or combined_df_data == "{}"
            or combined_df_data == ""
        ):
            error_msg = "Combined dataset is empty. Please run feature extraction and UMAP first."
            print(f"DEBUG ERROR: {error_msg}")
            return {}, error_msg, [], [], ""

        # Try to load the dataframe
        try:
            print("DEBUG: Attempting to parse JSON...")
            combined_df = pd.read_json(combined_df_data, orient="split")
            print(
                f"DEBUG: Successfully loaded combined_df with shape {combined_df.shape}"
            )
            print(
                f"DEBUG: Column names: {list(combined_df.columns)[:10]}..."
            )  # First 10 columns
            debug_text.append(f"Loaded combined dataset with {len(combined_df)} rows")
        except Exception as e:
            error_msg = f"Error parsing combined dataset: {str(e)}"
            print(f"DEBUG ERROR: {error_msg}")
            print(f"DEBUG: Full exception: {type(e).__name__}: {e}")

            traceback.print_exc()
            return {}, error_msg, [], [], ""

        if combined_df.empty:
            error_msg = "Combined dataset is empty after loading."
            print(f"DEBUG ERROR: {error_msg}")
            return {}, error_msg, [], [], ""

        # Load UMAP coordinates if needed for selections
        umap_coords = None
        if (
            "umap_coords" in combined_data_json
            and combined_data_json["umap_coords"] != "{}"
        ):
            try:
                print("DEBUG: Loading UMAP coordinates...")
                umap_coords = pd.read_json(
                    combined_data_json["umap_coords"], orient="split"
                )
                print(f"DEBUG: Loaded UMAP coords with shape {umap_coords.shape}")
            except Exception as e:
                print(f"DEBUG WARNING: Could not load UMAP coords: {e}")
                pass  # Not critical for MI analysis

        # Load Graph 3 subset if needed
        graph3_subset_df = None
        if (
            combined_data_json
            and "graph3_subset" in combined_data_json
            and combined_data_json["graph3_subset"] != "{}"
        ):
            try:
                print("DEBUG: Loading Graph 3 subset...")
                graph3_subset_df = pd.read_json(
                    combined_data_json["graph3_subset"], orient="split"
                )
                print(
                    f"DEBUG: Loaded Graph 3 subset with shape {graph3_subset_df.shape}"
                )
            except Exception as e:
                print(f"DEBUG WARNING: Could not load Graph 3 subset: {e}")
                pass  # Not critical

        # Step 3: Collect selected features
        # -----------------------------------------

        print("\nDEBUG: Collecting selected features...")

        # Handle pattern-matching callback results
        all_selected_features = []
        for features in selected_features_list:
            if features:  # Only add non-empty lists
                all_selected_features.extend(features)

        print(f"DEBUG: Found {len(all_selected_features)} selected features")

        # If no features selected, use default particle features
        if not all_selected_features:
            all_selected_features = [
                col for col in combined_df.columns if col.startswith("particle_")
            ]
            debug_text.append(
                f"No features selected, using {len(all_selected_features)} default particle features"
            )
            print(
                f"DEBUG: Using default particle features: {len(all_selected_features)} found"
            )
        else:
            debug_text.append(f"Using {len(all_selected_features)} selected features")

        # Step 4: Prepare data based on source
        # -----------------------------------------

        print("\nDEBUG: Preparing data based on source...")

        # Handle different data source options
        if data_source == "graph1" and graph1_selection and umap_coords is not None:
            print("DEBUG: Using Graph 1 selection")
            # [Graph 1 selection logic here - omitted for brevity]
            df_for_analysis = combined_df  # Simplified for debugging
            debug_text.append("Using Graph 1 selection")
        elif (
            data_source == "graph3"
            and graph3_selection
            and graph3_subset_df is not None
        ):
            print("DEBUG: Using Graph 3 selection")
            df_for_analysis = graph3_subset_df
            debug_text.append(f"Using Graph 3 subset with {len(df_for_analysis)} rows")
        else:
            print("DEBUG: Using all data")
            df_for_analysis = combined_df
            debug_text.append(f"Using all {len(df_for_analysis)} rows for analysis")

        # Step 5: Extract feature data
        # -----------------------------------------

        print("\nDEBUG: Extracting feature data...")

        # Get valid feature columns that exist in the dataset
        feature_cols = [
            col for col in df_for_analysis.columns if col in all_selected_features
        ]
        print(
            f"DEBUG: Valid feature columns: {len(feature_cols)} out of {len(all_selected_features)} requested"
        )

        if not feature_cols:
            error_msg = (
                "No valid features found in dataset. Please check feature selection."
            )
            print(f"DEBUG ERROR: {error_msg}")
            print(f"DEBUG: Available columns: {list(df_for_analysis.columns)[:20]}...")
            print(f"DEBUG: Requested features: {all_selected_features[:20]}...")
            return {}, error_msg, [], [], ""

        debug_text.append(f"Found {len(feature_cols)} valid features")

        # Extract feature matrix
        try:
            feature_matrix = df_for_analysis[feature_cols].copy()
            print(f"DEBUG: Feature matrix shape: {feature_matrix.shape}")

            # Handle NaN/inf values
            print("DEBUG: Handling NaN/inf values...")
            feature_matrix = feature_matrix.fillna(0)
            feature_matrix = feature_matrix.replace([np.inf, -np.inf], 0)

            # Check for target variables
            valid_targets = [
                t for t in target_variables if t in df_for_analysis.columns
            ]
            print(
                f"DEBUG: Valid target variables: {valid_targets} out of {target_variables}"
            )

            if not valid_targets:
                error_msg = f"Target variables {target_variables} not found in dataset."
                print(f"DEBUG ERROR: {error_msg}")
                print(
                    "DEBUG: Available columns for targets: "
                    f"{[col for col in df_for_analysis.columns if any(x in col for x in ['KER', 'EE', 'Energy', 'Total'])]}"
                )
                return {}, error_msg, [], [], ""

            debug_text.append(f"Using valid targets: {', '.join(valid_targets)}")

        except Exception as e:
            error_msg = f"Error preparing feature matrix: {str(e)}"
            print(f"DEBUG ERROR: {error_msg}")

            traceback.print_exc()
            return {}, error_msg, [], [], ""

        # Step 6: Mutual Information Feature Selection
        # -----------------------------------------

        print("\nDEBUG: Computing mutual information...")
        debug_text.append("Computing mutual information with target variables...")

        # Compute MI with each target
        mi_scores = {}
        try:
            for target in valid_targets:
                print(f"DEBUG: Computing MI for target: {target}")
                target_values = df_for_analysis[target].values
                # Handle NaN/inf in target values
                target_values = np.nan_to_num(
                    target_values, nan=0.0, posinf=0.0, neginf=0.0
                )

                mi_scores[target] = mutual_info_regression(
                    feature_matrix, target_values, random_state=42
                )
                print(
                    f"DEBUG: MI scores for {target}: min={mi_scores[target].min():.4f}, max={mi_scores[target].max():.4f}"
                )
        except Exception as e:
            error_msg = f"Error computing mutual information: {str(e)}"
            print(f"DEBUG ERROR: {error_msg}")

            traceback.print_exc()
            return {}, error_msg, [], [], ""

        # Average MI scores across all targets
        try:
            avg_mi_scores = np.mean(
                [mi_scores[target] for target in valid_targets], axis=0
            )
            mi_scores_dict = dict(zip(feature_matrix.columns, avg_mi_scores))

            # Sort features by MI score (highest to lowest)
            sorted_features = sorted(
                mi_scores_dict, key=mi_scores_dict.get, reverse=True
            )

            print("\nDEBUG: Top 5 features by mutual information:")
            debug_text.append("Top 5 features by mutual information:")
            for i, feature in enumerate(sorted_features[:5]):
                score = mi_scores_dict[feature]
                print(f"  {i+1}. {feature}: {score:.4f}")
                debug_text.append(f"{i+1}. {feature}: {score:.4f}")

        except Exception as e:
            error_msg = f"Error processing MI scores: {str(e)}"
            print(f"DEBUG ERROR: {error_msg}")

            traceback.print_exc()
            return {}, error_msg, [], [], ""

        # Step 7: Feature selection with redundancy reduction
        # -----------------------------------------

        print("\nDEBUG: Selecting features with redundancy reduction...")
        debug_text.append(
            f"Selecting up to {max_features} features with redundancy threshold {redundancy_threshold}"
        )

        try:
            # Select top features
            selected_features = []
            feature_correlations = feature_matrix.corr().abs()

            for feature in sorted_features:
                if len(selected_features) >= max_features:
                    break

                # Check redundancy with already selected features
                is_redundant = False
                for selected in selected_features:
                    if (
                        feature in feature_correlations.columns
                        and selected in feature_correlations.index
                    ):
                        if (
                            feature_correlations.loc[selected, feature]
                            > redundancy_threshold
                        ):
                            is_redundant = True
                            break

                if not is_redundant:
                    selected_features.append(feature)

            print(
                f"\nDEBUG: Selected {len(selected_features)} features after redundancy reduction"
            )
            debug_text.append(
                f"Selected {len(selected_features)} features after redundancy reduction"
            )

        except Exception as e:
            print(f"DEBUG WARNING: Error in redundancy reduction: {e}")
            # Fall back to simple selection
            selected_features = sorted_features[:max_features]

        # Step 8: Prepare results
        # -----------------------------------------

        print("\nDEBUG: Preparing results...")

        try:
            # Store results
            mi_store = {
                "selected_features": selected_features,
                "mi_scores": mi_scores_dict,
                "feature_data": df_for_analysis[selected_features].to_json(
                    date_format="iso", orient="split"
                ),
                "file_labels": (
                    df_for_analysis[["file_label"]].to_json(
                        date_format="iso", orient="split"
                    )
                    if "file_label" in df_for_analysis.columns
                    else pd.DataFrame().to_json(orient="split")
                ),
                "target_variables": valid_targets,
            }

            # Create feature options for dropdowns
            feature_options = [{"label": f, "value": f} for f in selected_features]

            # Create info text
            info_lines = [f"Selected {len(selected_features)} features:"]
            for i, feature in enumerate(selected_features[:5]):
                score = mi_scores_dict[feature]
                info_lines.append(f"  {i+1}. {feature}: {score:.4f}")

            if len(selected_features) > 5:
                info_lines.append(f"  ... and {len(selected_features)-5} more")

            info_text = html.Div([html.Div(line) for line in info_lines])

            success_message = f"MI feature selection complete! Selected {len(selected_features)} features "
            f"from {len(feature_cols)} candidates."

            print(f"\nDEBUG: SUCCESS - {success_message}")
            print("=" * 80)

            return (
                mi_store,
                success_message,
                feature_options,
                feature_options,
                info_text,
            )

        except Exception as e:
            error_msg = f"Error preparing results: {str(e)}"
            print(f"DEBUG ERROR: {error_msg}")

            traceback.print_exc()
            return {}, error_msg, [], [], ""

    except Exception as e:

        trace = traceback.format_exc()
        error_message = f"Error in MI feature selection: {str(e)}"
        print("\nDEBUG: CRITICAL ERROR")
        print(f"Error: {error_message}")
        print(f"Traceback:\n{trace}")
        print("=" * 80)

        return {}, error_message, [], [], html.Div(f"Error: {str(e)}")


@callback(
    Output("feature-importance-table", "data"),
    Input("feature-search-button", "n_clicks"),
    Input("feature-sort-option", "value"),
    State("feature-search-input", "value"),
    State("autoencoder-latent-store", "data"),
    State("file-config-assignments-store", "data"),  # ADD THIS
    State("configuration-profiles-store", "data"),  # ADD THIS
    prevent_initial_call=True,
)
def update_feature_importance_table(
    n_clicks, sort_option, search_term, latent_store, assignments_store, profiles_store
):  # ADD PARAMETERS
    """Update the feature importance table based on search and sort options."""
    ctx = dash.callback_context
    if not ctx.triggered or not latent_store:
        raise dash.exceptions.PreventUpdate

    # Get MI and correlation scores from the store
    try:
        mi_scores_dict = latent_store.get("mi_scores", {})
        corr_scores_dict = latent_store.get("corr_scores", {})

        if not mi_scores_dict or not corr_scores_dict:
            return []

        # Get features that are in both dictionaries
        common_features = list(
            set(mi_scores_dict.keys()).intersection(set(corr_scores_dict.keys()))
        )

        if not common_features:
            return []

        # Create DataFrame with all features and scores
        feature_importance_df = pd.DataFrame(
            {
                "Feature": common_features,
                "Mutual_Information": [
                    mi_scores_dict.get(f, 0.0) for f in common_features
                ],
                "Correlation": [
                    abs(corr_scores_dict.get(f, 0.0)) for f in common_features
                ],  # Use absolute correlation value
            }
        )

        # Filter by search term if provided
        if search_term and len(search_term.strip()) > 0:
            pattern = re.compile(search_term, re.IGNORECASE)
            feature_importance_df = feature_importance_df[
                feature_importance_df["Feature"].apply(
                    lambda x: bool(pattern.search(x))
                )
            ]

        # Sort based on selected option
        if sort_option == "mi":
            feature_importance_df = feature_importance_df.sort_values(
                "Mutual_Information", ascending=False
            )
        else:  # sort by correlation
            feature_importance_df = feature_importance_df.sort_values(
                "Correlation", ascending=False
            )

        # Reset index for proper ranking
        feature_importance_df = feature_importance_df.reset_index(drop=True)

        # Prepare the table data
        table_data = []
        for i, row in feature_importance_df.iterrows():
            table_data.append(
                {
                    "Rank": i + 1,
                    "Feature": row["Feature"],
                    "MI Score": f"{row['Mutual_Information']:.4f}",
                    "Correlation": f"{row['Correlation']:.4f}",
                }
            )

        return table_data

    except Exception as e:
        print(f"Error updating feature importance table: {str(e)}")

        traceback.print_exc()
        return []
