import re

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap
from dash import ALL, Input, Output, State, callback, dash_table, dcc, html
from torch.utils.data import DataLoader, TensorDataset

from sculpt.models.deep_autoencoder import DeepAutoencoder
from sculpt.utils.metrics.clustering_quality import cluster_stability, hopkins_statistic
from sculpt.utils.metrics.physics_features import (
    calculate_physics_features_flexible,
    calculate_physics_features_with_profile,
)
from sculpt.utils.ui import create_feature_categories_ui


# Callback for training autoencoder and running UMAP on latent space
@callback(
    Output("autoencoder-umap-graph", "figure"),
    Output("autoencoder-debug-output", "children"),
    Output("autoencoder-latent-store", "data"),
    Output("umap-quality-metrics-autoencoder", "children"),
    Output("feature-importance-container", "children"),
    Input("train-autoencoder", "n_clicks"),
    Input("run-umap-latent", "n_clicks"),
    State("autoencoder-latent-dim", "value"),
    State("autoencoder-epochs", "value"),
    State("autoencoder-batch-size", "value"),
    State("autoencoder-learning-rate", "value"),
    State("autoencoder-data-source", "value"),
    State("selected-points-store", "data"),
    State("selected-points-run-store", "data"),
    State("combined-data-store", "data"),
    State("autoencoder-umap-neighbors", "value"),
    State("autoencoder-umap-min-dist", "value"),
    State({"type": "feature-selector-autoencoder", "category": ALL}, "value"),
    State({"type": "genetic-feature-selector-autoencoder", "category": ALL}, "value"),
    State("umap-graph", "figure"),
    State("autoencoder-latent-store", "data"),
    State("metric-selector-autoencoder", "value"),
    State("genetic-features-store", "data"),
    #background=True,
    #running=[
    #    (Output("train-autoencoder", "disabled"), True, False),
    #    (Output("training-progress", "children"), "Training...", ""),
    #],
    #progress=[Output("training-progress", "children")],
    prevent_initial_call=True,
)
def train_autoencoder_and_run_umap(
    train_clicks,
    umap_clicks,
    latent_dim,
    epochs,
    batch_size,
    learning_rate,
    data_source,
    graph1_selection,
    graph3_selection,
    combined_data_json,
    n_neighbors,
    min_dist,
    selected_features_list,
    selected_genetic_features_list,
    original_figure,
    latent_store,
    selected_metrics,
    genetic_features_store,
    #set_progress,
):
    """Train autoencoder and run UMAP on latent space with genetic features support."""

    # ADD THE DEBUG CODE RIGHT HERE (before the try block)
    print("\n" + "="*50)
    print("🔍 AUTOENCODER DEBUG")
    print("="*50)
    print(f"combined_data_json: {combined_data_json}")
    
    if combined_data_json:
        for key, value in combined_data_json.items():
            print(f"  {key}: {type(value)}")
            if key == "combined_df":
                print(f"    combined_df value: {value}")
                print(f"    Is empty string: {value == '{}'}")
    
    print("="*50)

    try:
        # Import required libraries
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
        from sklearn.preprocessing import StandardScaler

        ctx = dash.callback_context
        if not ctx.triggered:
            return {}, "No action triggered.", {}, [], []

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Initialize debug info
        debug_text = ["Starting callback execution..."]

        # Initialize feature importance container
        feature_importance_container = []

        # Load the combined dataframe - do this regardless of training or not
        if combined_data_json and "combined_df" in combined_data_json and combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
            print(f"✅ SUCCESS: Loaded DataFrame with shape {combined_df.shape}")
            debug_text.append(f"Loaded combined dataset with {len(combined_df)} rows")
        else:
            print("❌ FAILED: combined_df is empty or missing")
            if combined_data_json and "combined_df" in combined_data_json:
                print(f"combined_df value: {combined_data_json['combined_df']}")
            return {}, "No combined dataset available. Please run UMAP first.", {}, [], []

        # Check if we need to train the autoencoder or just run UMAP on existing latent space
        if trigger_id == "run-umap-latent" and latent_store:
            debug_text.append("Using previously trained autoencoder latent space.")
            should_train = False
        else:
            should_train = True
            debug_text.append("Training new autoencoder.")

        if should_train:
            # Step 1: Collect BOTH original and genetic features
            debug_text.append(f"Data source: {data_source}")
            debug_text.append(f"Latent dimension: {latent_dim}")
            debug_text.append(f"Epochs: {epochs}")

            # Collect selected original features
            all_selected_original_features = []
            for features in selected_features_list:
                if features:  # Only add non-empty lists
                    all_selected_original_features.extend(features)

            # Collect selected genetic features
            all_selected_genetic_features = []
            if selected_genetic_features_list:
                for features in selected_genetic_features_list:
                    if features:  # Only add non-empty lists
                        all_selected_genetic_features.extend(features)

            debug_text.append(
                f"Selected original features: {len(all_selected_original_features)}"
            )
            debug_text.append(
                f"Selected genetic features: {len(all_selected_genetic_features)}"
            )

            # Prepare data based on data source
            df_for_training = combined_df.copy()
            labels = combined_df["file_label"].copy()
            debug_text.append(f"Using all {len(df_for_training)} rows for training")

            # Build combined feature matrix
            feature_matrices = []
            feature_names_combined = []
            feature_cols = []  # Initialize here to avoid the error

            # Add original features if selected
            if all_selected_original_features:
                original_feature_cols = [
                    col
                    for col in df_for_training.columns
                    if col in all_selected_original_features
                ]
                if original_feature_cols:
                    feature_cols = original_feature_cols  # Set feature_cols here
                    original_features = df_for_training[
                        original_feature_cols
                    ].to_numpy()
                    feature_matrices.append(original_features)
                    feature_names_combined.extend(original_feature_cols)
                    debug_text.append(
                        f"Added {len(original_feature_cols)} original features"
                    )

            # Add genetic features if available and selected
            if (
                all_selected_genetic_features
                and genetic_features_store
                and "genetic_features" in genetic_features_store
            ):
                try:
                    genetic_df = pd.read_json(
                        genetic_features_store["genetic_features"], orient="split"
                    )
                    genetic_feature_cols = [
                        col
                        for col in genetic_df.columns
                        if col in all_selected_genetic_features
                    ]

                    if genetic_feature_cols:
                        # Make sure genetic features align with current data
                        if len(genetic_df) == len(df_for_training):
                            genetic_features_matrix = genetic_df[
                                genetic_feature_cols
                            ].to_numpy()
                            feature_matrices.append(genetic_features_matrix)
                            feature_names_combined.extend(genetic_feature_cols)
                            debug_text.append(
                                f"Added {len(genetic_feature_cols)} genetic features"
                            )
                        else:
                            debug_text.append(
                                "Warning: Genetic features don't match current data size, skipping"
                            )
                except Exception as e:
                    debug_text.append(
                        f"Warning: Could not load genetic features: {str(e)}"
                    )

            # Combine all feature matrices or use fallback
            if not feature_matrices:
                # Fallback to original momentum features
                original_cols = [
                    col
                    for col in df_for_training.columns
                    if col.startswith("particle_")
                ]
                feature_cols = original_cols  # Set feature_cols for fallback
                feature_data = df_for_training[original_cols].to_numpy()
                feature_names_combined = original_cols
                debug_text.append(
                    "No features selected, using original momentum components as fallback"
                )
            else:
                # Concatenate all selected features
                feature_data = np.concatenate(feature_matrices, axis=1)
                debug_text.append(
                    f"Combined feature matrix shape: {feature_data.shape}"
                )

            # Handle NaN/inf values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Store feature information for later use
            combined_feature_info = {
                "feature_names": feature_names_combined,
                "num_original": (
                    len(all_selected_original_features)
                    if all_selected_original_features
                    else 0
                ),
                "num_genetic": (
                    len(all_selected_genetic_features)
                    if all_selected_genetic_features
                    else 0
                ),
                "total_features": feature_data.shape[1],
            }

            debug_text.append(
                f"Final feature matrix: {feature_data.shape[1]} features, {feature_data.shape[0]} samples"
            )
            debug_text.append(
                f"Feature breakdown: {combined_feature_info['num_original']} original "
                f"+ {combined_feature_info['num_genetic']} genetic"
            )

            # Create PyTorch dataset and dataloader
            feature_tensor = torch.tensor(feature_data, dtype=torch.float32)
            dataset = TensorDataset(feature_tensor)
            dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

            # Set up device (CPU or GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            debug_text.append(f"Using device: {device}")

            # Initialize model
            input_dim = feature_data.shape[1]
            model = DeepAutoencoder(input_dim=input_dim, latent_dim=int(latent_dim)).to(
                device
            )

            # Initialize optimizer and loss function
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))

            # Training loop
            num_epochs = int(epochs)
            losses = []

            debug_text.append(f"Starting training for {num_epochs} epochs...")

            for epoch in range(num_epochs):
                #set_progress(f"Training progress: Epoch {epoch + 1}/{num_epochs}")
                total_loss = 0
                model.train()

                for batch in dataloader:
                    batch_data = batch[0].to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    reconstructed, _ = model(batch_data)

                    # Calculate loss
                    loss = criterion(reconstructed, batch_data)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Average loss for this epoch
                avg_loss = total_loss / len(dataloader)
                losses.append(avg_loss)

                # Only log some epochs to avoid overcrowding
                if (
                    epoch == 0
                    or epoch == num_epochs - 1
                    or (epoch + 1) % max(1, num_epochs // 5) == 0
                ):
                    debug_text.append(
                        f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}"
                    )

            debug_text.append("Training complete!")

            # Extract latent representations
            model.eval()
            with torch.no_grad():
                latent_features = model.encoder(feature_tensor.to(device)).cpu().numpy()

            # Create dataframe with latent features
            latent_df = pd.DataFrame(
                latent_features, columns=[f"Latent_{i}" for i in range(latent_dim)]
            )
            latent_df["file_label"] = labels.values

            # Store the latent features for future use
            latent_store = {
                "latent_features": latent_df.to_json(date_format="iso", orient="split"),
                "feature_cols": feature_cols,
                "latent_dim": latent_dim,
            }

            debug_text.append(
                f"Extracted {len(latent_df)} latent representations with dimension {latent_dim}"
            )

            # If training completed but we don't want to run UMAP yet
            if trigger_id == "train-autoencoder":
                placeholder_fig = {
                    "data": [],
                    "layout": {
                        "title": 'Training complete! Click "Run UMAP on Latent Space" to visualize',
                        "xaxis": {"title": "UMAP1"},
                        "yaxis": {"title": "UMAP2"},
                        "height": 600,
                    },
                }
                debug_text.append("Training complete! Ready for UMAP visualization.")
                return placeholder_fig, "<br>".join(debug_text), latent_store, [], []

        else:
            # Load latent features from store
            try:
                latent_df = pd.read_json(
                    latent_store["latent_features"], orient="split"
                )
                feature_cols = latent_store["feature_cols"]
                latent_dim = latent_store["latent_dim"]
                debug_text.append(
                    f"Loaded {len(latent_df)} latent representations with dimension {latent_dim}"
                )
            except Exception as e:
                return {}, f"Error loading latent features: {str(e)}", {}, [], []

        # Run UMAP on the latent space
        debug_text.append(
            f"Running UMAP on latent space (n_neighbors={n_neighbors}, min_dist={min_dist})..."
        )

        # Prepare data for UMAP
        X_latent = latent_df[[f"Latent_{i}" for i in range(latent_dim)]].to_numpy()

        # Run UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric="euclidean",
            random_state=42,
        )

        # Fit UMAP
        umap_data = reducer.fit_transform(X_latent)

        # Create DataFrame for visualization
        umap_df = pd.DataFrame(umap_data, columns=["UMAP1", "UMAP2"])
        umap_df["file_label"] = latent_df["file_label"].values

        debug_text.append(f"UMAP transformation complete with {len(umap_df)} points")

        # Extract color information from original figure
        import plotly.graph_objects as go

        # Create a color map from the original figure
        color_map = {}
        if original_figure and "data" in original_figure:
            for trace in original_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    # Clean the label if it contains point count
                    clean_name = trace["name"]
                    if " (" in clean_name:
                        clean_name = clean_name.split(" (")[0]
                    color_map[clean_name] = trace["marker"]["color"]

        # Create figure with consistent colors
        fig = go.Figure()

        # Add traces for each file label
        for label in umap_df["file_label"].unique():
            mask = umap_df["file_label"] == label
            df_subset = umap_df[mask]

            # Get color from original figure if available
            color = color_map.get(label, None)

            fig.add_trace(
                go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode="markers",
                    marker=dict(size=8, color=color, opacity=0.7),
                    name=f"{label} ({len(df_subset)} pts)",
                )
            )

        # Update figure layout
        legend_config = dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10),
            itemwidth=30,
            itemsizing="constant",
            tracegroupgap=5,
        )

        num_traces = len(fig.data)
        base_height = 500
        legend_height = max(50, min(100, num_traces * 3))

        fig.update_layout(
            height=base_height,
            title=f"UMAP of Autoencoder Latent Space (dim={latent_dim}, n_neighbors={n_neighbors}, min_dist={min_dist})",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data File",
            legend=legend_config,
            margin=dict(l=50, r=50, t=50, b=legend_height),
            showlegend=True,
        )

        # Add latent UMAP coordinates to the latent store
        latent_store["umap_coords"] = umap_df.to_json(date_format="iso", orient="split")

        # Calculate mutual information between original features and UMAP dimensions
        debug_text.append(
            "Calculating mutual information between original features and UMAP dimensions..."
        )

        try:
            # Get the original feature columns that were used to train the autoencoder
            feature_cols = latent_store.get("feature_cols", [])

            if feature_cols and len(feature_cols) > 0:
                # Check if the combined_df has these feature columns
                missing_cols = [
                    col for col in feature_cols if col not in combined_df.columns
                ]
                if missing_cols:
                    debug_text.append(
                        f"Warning: Some feature columns are missing from the data: {missing_cols}"
                    )
                    feature_cols = [
                        col for col in feature_cols if col in combined_df.columns
                    ]

                if not feature_cols:
                    raise ValueError(
                        "No valid feature columns found for MI calculation"
                    )

                # Get the data source that was used - use all data
                original_features_df = combined_df[feature_cols]

                # Handle NaN/inf values in the original features
                original_features_np = np.nan_to_num(
                    original_features_df.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0
                )

                # Get the UMAP coordinates for mutual information calculation
                umap_coords_np = umap_df[["UMAP1", "UMAP2"]].to_numpy()

                # Ensure original_features_np and umap_coords_np have the same length
                if len(original_features_np) != len(umap_coords_np):
                    debug_text.append(
                        f"Warning: Feature matrix ({len(original_features_np)} rows) and UMAP coordinates "
                        "({len(umap_coords_np)} rows) have different lengths"
                    )
                    min_length = min(len(original_features_np), len(umap_coords_np))
                    original_features_np = original_features_np[:min_length]
                    umap_coords_np = umap_coords_np[:min_length]

                # Compute mutual information between original features and each UMAP dimension
                umap_mi_scores = {}
                for i, dim in enumerate(["UMAP1", "UMAP2"]):
                    umap_mi_scores[dim] = mutual_info_regression(
                        original_features_np, umap_coords_np[:, i]
                    )

                # Average MI scores across both UMAP dimensions
                avg_mi_scores = np.mean(list(umap_mi_scores.values()), axis=0)

                # Create a dictionary mapping each feature name to its average MI score
                mi_scores_dict = dict(zip(feature_cols, avg_mi_scores))

                # Sort features by MI score (highest first)
                sorted_features = sorted(
                    mi_scores_dict, key=mi_scores_dict.get, reverse=True
                )

                # Add top contributing features to debug text
                debug_text.append(
                    "Top Features Contributing to Latent Space Clustering:"
                )
                for feature in sorted_features[:10]:
                    debug_text.append(f"{feature}: {mi_scores_dict[feature]:.4f}")

                # Also calculate correlation between original features and UMAP dimensions
                corr_scores = {}
                for i, dim in enumerate(["UMAP1", "UMAP2"]):
                    corr_scores[dim] = []
                    for j, feature in enumerate(feature_cols):
                        corr = np.corrcoef(
                            original_features_np[:, j], umap_coords_np[:, i]
                        )[0, 1]
                        corr_scores[dim].append(corr)

                # Calculate average absolute correlation across dimensions
                avg_corr_scores = np.mean(
                    [np.abs(corr_scores["UMAP1"]), np.abs(corr_scores["UMAP2"])], axis=0
                )
                corr_scores_dict = dict(zip(feature_cols, avg_corr_scores))

                # Sort features by correlation score (highest first)
                sorted_features_corr = sorted(
                    corr_scores_dict, key=corr_scores_dict.get, reverse=True
                )

                # Add top contributing features by correlation to debug text
                debug_text.append("Top Features by Correlation with UMAP Dimensions:")
                for feature in sorted_features_corr[:10]:
                    debug_text.append(f"{feature}: {corr_scores_dict[feature]:.4f}")

                # Store MI and correlation scores in the latent store for potential future use
                latent_store["mi_scores"] = mi_scores_dict
                latent_store["corr_scores"] = corr_scores_dict

                # Create feature importance UI
                mi_ui = html.Div(
                    [
                        html.H4(
                            "Feature Importance Analysis",
                            style={
                                "fontSize": "16px",
                                "color": "#2e7d32",
                                "marginBottom": "15px",
                            },
                        ),
                        # Create a tabbed interface for better organization
                        dcc.Tabs(
                            id="feature-importance-tabs",
                            value="summary-tab",
                            children=[
                                # Summary tab with top features
                                dcc.Tab(
                                    label="Summary (Top 15)",
                                    value="summary-tab",
                                    children=[
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.H5(
                                                            "Top Features by Mutual Information:",
                                                            style={
                                                                "fontSize": "14px",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            f"{i+1}. {feature}: ",
                                                                            style={
                                                                                "fontWeight": "bold",
                                                                                "fontSize": "12px",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            f"{mi_scores_dict[feature]:.4f}",
                                                                            style={
                                                                                "fontSize": "12px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "marginBottom": "3px"
                                                                    },
                                                                )
                                                                for i, feature in enumerate(
                                                                    sorted_features[:15]
                                                                )
                                                            ],
                                                            style={
                                                                "marginLeft": "10px"
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "50%",
                                                        "display": "inline-block",
                                                        "verticalAlign": "top",
                                                        "paddingRight": "10px",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.H5(
                                                            "Top Features by Correlation:",
                                                            style={
                                                                "fontSize": "14px",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            f"{i+1}. {feature}: ",
                                                                            style={
                                                                                "fontWeight": "bold",
                                                                                "fontSize": "12px",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            f"{corr_scores_dict[feature]:.4f}",
                                                                            style={
                                                                                "fontSize": "12px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "marginBottom": "3px"
                                                                    },
                                                                )
                                                                for i, feature in enumerate(
                                                                    sorted_features_corr[
                                                                        :15
                                                                    ]
                                                                )
                                                            ],
                                                            style={
                                                                "marginLeft": "10px"
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "50%",
                                                        "display": "inline-block",
                                                        "verticalAlign": "top",
                                                        "paddingLeft": "10px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "flexWrap": "wrap",
                                            },
                                        )
                                    ],
                                ),
                                # Complete table tab with all features
                                dcc.Tab(
                                    label="Complete Table",
                                    value="complete-tab",
                                    children=[
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.H5(
                                                            "All Features - Mutual Information",
                                                            style={
                                                                "fontSize": "14px",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            f"{i+1}. {feature}: ",
                                                                            style={
                                                                                "fontWeight": "bold",
                                                                                "fontSize": "11px",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            f"{mi_scores_dict[feature]:.4f}",
                                                                            style={
                                                                                "fontSize": "11px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "marginBottom": "2px"
                                                                    },
                                                                )
                                                                for i, feature in enumerate(
                                                                    sorted_features
                                                                )
                                                            ],
                                                            style={
                                                                "marginLeft": "10px",
                                                                "maxHeight": "300px",
                                                                "overflowY": "auto",
                                                                "border": "1px solid #ddd",
                                                                "padding": "10px",
                                                                "backgroundColor": "#f9f9f9",
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "50%",
                                                        "display": "inline-block",
                                                        "verticalAlign": "top",
                                                        "paddingRight": "10px",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.H5(
                                                            "All Features - Correlation",
                                                            style={
                                                                "fontSize": "14px",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Span(
                                                                            f"{i+1}. {feature}: ",
                                                                            style={
                                                                                "fontWeight": "bold",
                                                                                "fontSize": "11px",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            f"{corr_scores_dict[feature]:.4f}",
                                                                            style={
                                                                                "fontSize": "11px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "marginBottom": "2px"
                                                                    },
                                                                )
                                                                for i, feature in enumerate(
                                                                    sorted_features_corr
                                                                )
                                                            ],
                                                            style={
                                                                "marginLeft": "10px",
                                                                "maxHeight": "300px",
                                                                "overflowY": "auto",
                                                                "border": "1px solid #ddd",
                                                                "padding": "10px",
                                                                "backgroundColor": "#f9f9f9",
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "50%",
                                                        "display": "inline-block",
                                                        "verticalAlign": "top",
                                                        "paddingLeft": "10px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "flexWrap": "wrap",
                                            },
                                        )
                                    ],
                                ),
                            ],
                            style={"marginBottom": "15px"},
                        ),
                        # Explanation section
                        html.Div(
                            [
                                html.Details(
                                    [
                                        html.Summary(
                                            "What does this mean?",
                                            style={
                                                "cursor": "pointer",
                                                "color": "#2e7d32",
                                                "fontSize": "13px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.P(
                                                    "• Mutual Information (MI) measures nonlinear relationships between "
                                                    "features and UMAP dimensions",
                                                    style={
                                                        "fontSize": "11px",
                                                        "marginBottom": "5px",
                                                    },
                                                ),
                                                html.P(
                                                    "• Correlation measures linear relationships only",
                                                    style={
                                                        "fontSize": "11px",
                                                        "marginBottom": "5px",
                                                    },
                                                ),
                                                html.P(
                                                    "• Higher values indicate features that strongly influence the latent "
                                                    "space clustering",
                                                    style={
                                                        "fontSize": "11px",
                                                        "marginBottom": "5px",
                                                    },
                                                ),
                                                html.P(
                                                    "• Features appearing high in both metrics have strong overall influence",
                                                    style={
                                                        "fontSize": "11px",
                                                        "marginBottom": "5px",
                                                    },
                                                ),
                                                html.P(
                                                    "• Features high in MI but low in correlation have primarily nonlinear "
                                                    "influence",
                                                    style={
                                                        "fontSize": "11px",
                                                        "marginBottom": "0px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "paddingLeft": "15px",
                                                "marginTop": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={"marginTop": "15px"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#f1f8e9",
                        "padding": "15px",
                        "borderRadius": "5px",
                        "marginTop": "15px",
                        "border": "1px solid #c8e6c9",
                    },
                )

                # Add the feature importance UI to the container
                feature_importance_container = [mi_ui]

            else:
                debug_text.append(
                    "No feature columns available for mutual information calculation"
                )
                feature_importance_container = [
                    html.Div(
                        "No feature columns available for mutual information calculation"
                    )
                ]

        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            debug_text.append(f"Error calculating mutual information: {str(e)}")
            print(f"Error calculating mutual information: {str(e)}\n{trace}")
            feature_importance_container = [
                html.Div(f"Error calculating feature importance: {str(e)}")
            ]

        # Calculate clustering metrics
        metrics_children = []
        try:
            # Get UMAP coordinates for clustering
            X_umap = umap_df[["UMAP1", "UMAP2"]].to_numpy()

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

            # Count clusters and noise points
            unique_clusters = set(cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = np.sum(cluster_labels == -1)
            noise_ratio = (
                n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            )

            metrics = {}

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

                    metrics["note"] = "Metrics calculated excluding noise points"
                else:
                    metrics["note"] = "Not enough valid points for metrics"
            else:
                # Try KMeans as fallback
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                fallback_labels = kmeans.fit_predict(X_umap_scaled)

                if "silhouette" in selected_metrics:
                    metrics["silhouette"] = silhouette_score(
                        X_umap_scaled, fallback_labels
                    )

                if "davies_bouldin" in selected_metrics:
                    metrics["davies_bouldin"] = davies_bouldin_score(
                        X_umap_scaled, fallback_labels
                    )

                if "calinski_harabasz" in selected_metrics:
                    metrics["calinski_harabasz"] = calinski_harabasz_score(
                        X_umap_scaled, fallback_labels
                    )

                if "hopkins" in selected_metrics:
                    h_stat = hopkins_statistic(X_umap_scaled)
                    metrics["hopkins"] = h_stat

                metrics["note"] = (
                    "DBSCAN found no clusters, metrics based on KMeans fallback"
                )

            # Create UI elements for the metrics
            metrics_children = [
                html.H4(
                    "Clustering Quality Metrics (DBSCAN)",
                    style={"fontSize": "14px", "marginBottom": "5px"},
                ),
                html.Div(
                    [
                        # Existing metrics
                        html.Div(
                            [
                                html.Span(
                                    "Estimated Clusters: ", style={"fontWeight": "bold"}
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
                                html.Span("DBSCAN eps: ", style={"fontWeight": "bold"}),
                                html.Span(f"{best_eps:.3f}"),
                            ]
                        ),
                        # Basic metrics (existing)
                        (
                            html.Div(
                                [
                                    html.Span(
                                        "Silhouette Score: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    html.Span(
                                        f"{metrics.get('silhouette', 'N/A'):.4f}",
                                        style={
                                            "color": (
                                                "green"
                                                if metrics.get("silhouette", 0) > 0.5
                                                else (
                                                    "orange"
                                                    if metrics.get("silhouette", 0)
                                                    > 0.25
                                                    else "red"
                                                )
                                            )
                                        },
                                    ),
                                ]
                            )
                            if "silhouette" in metrics
                            else None
                        ),
                        (
                            html.Div(
                                [
                                    html.Span(
                                        "Davies-Bouldin Index: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    html.Span(
                                        f"{metrics.get('davies_bouldin', 'N/A'):.4f}",
                                        style={
                                            "color": (
                                                "green"
                                                if metrics.get(
                                                    "davies_bouldin", float("inf")
                                                )
                                                < 0.8
                                                else (
                                                    "orange"
                                                    if metrics.get(
                                                        "davies_bouldin", float("inf")
                                                    )
                                                    < 1.5
                                                    else "red"
                                                )
                                            )
                                        },
                                    ),
                                ]
                            )
                            if "davies_bouldin" in metrics
                            else None
                        ),
                        (
                            html.Div(
                                [
                                    html.Span(
                                        "Calinski-Harabasz Index: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    html.Span(
                                        f"{metrics.get('calinski_harabasz', 'N/A'):.1f}",
                                        style={
                                            "color": (
                                                "green"
                                                if metrics.get("calinski_harabasz", 0)
                                                > 100
                                                else (
                                                    "orange"
                                                    if metrics.get(
                                                        "calinski_harabasz", 0
                                                    )
                                                    > 50
                                                    else "red"
                                                )
                                            )
                                        },
                                    ),
                                ]
                            )
                            if "calinski_harabasz" in metrics
                            else None
                        ),
                        # New metrics
                        (
                            html.Div(
                                [
                                    html.Span(
                                        "Hopkins Statistic: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    html.Span(
                                        f"{metrics.get('hopkins', 'N/A'):.4f}",
                                        style={
                                            "color": (
                                                "green"
                                                if metrics.get("hopkins", 0) > 0.75
                                                else (
                                                    "orange"
                                                    if metrics.get("hopkins", 0) > 0.6
                                                    else "red"
                                                )
                                            )
                                        },
                                    ),
                                ]
                            )
                            if "hopkins" in metrics
                            else None
                        ),
                        (
                            html.Div(
                                [
                                    html.Span(
                                        "Cluster Stability: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    html.Span(
                                        f"{metrics.get('stability', 'N/A'):.4f}",
                                        style={
                                            "color": (
                                                "green"
                                                if metrics.get("stability", 0) > 0.8
                                                else (
                                                    "orange"
                                                    if metrics.get("stability", 0) > 0.6
                                                    else "red"
                                                )
                                            )
                                        },
                                    ),
                                ]
                            )
                            if "stability" in metrics
                            else None
                        ),
                        html.Div(
                            metrics.get("note", ""),
                            style={
                                "fontSize": "11px",
                                "fontStyle": "italic",
                                "marginTop": "3px",
                            },
                        ),
                    ]
                ),
            ]

            # Add a tooltip about the metrics
            metrics_children.append(
                html.Div(
                    [
                        html.Hr(style={"marginTop": "10px", "marginBottom": "10px"}),
                        html.Details(
                            [
                                html.Summary(
                                    "What do these metrics mean?",
                                    style={"cursor": "pointer"},
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "• Silhouette Score: Measures how well-separated clusters are (higher is better, "
                                            "range: -1 to 1)"
                                        ),
                                        html.P(
                                            "• Davies-Bouldin Index: Measures average similarity between clusters (lower is "
                                            "better, range: 0 to ∞)"
                                        ),
                                        html.P(
                                            "• Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion "
                                            "(higher is better, range: 0 to ∞)"
                                        ),
                                        html.P(
                                            "• Hopkins Statistic: Measures clusterability of the data (>0.75 indicates good "
                                            "clustering, range: 0 to 1)"
                                        ),
                                        html.P(
                                            "• Cluster Stability: How stable clusters are with small perturbations (higher is "
                                            "better, range: 0 to 1)"
                                        ),
                                        html.P(
                                            "• Physics Consistency: How well clusters align with physical parameters (higher "
                                            "is better, range: 0 to 1)"
                                        ),
                                    ],
                                    style={"fontSize": "11px", "paddingLeft": "10px"},
                                ),
                            ]
                        ),
                    ],
                    style={"marginTop": "10px"},
                )
            )

        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            metrics_children = [html.Div(f"Error calculating metrics: {str(e)}")]

        return (
            fig,
            "<br>".join(debug_text),
            latent_store,
            metrics_children,
            feature_importance_container,
        )

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        error_message = f"Error in callback: {str(e)}\n{trace}"
        print(error_message)  # Print to console for debugging

        # Return minimal valid outputs for all return values
        return {}, f"Error: {str(e)}", {}, [], []


# TODO: Check placement of the following callbacks
# Add this new callback to update genetic features in the autoencoder section


@callback(
    Output("genetic-feature-selection-autoencoder", "children"),
    Output("feature-combination-summary", "children"),
    Input("genetic-features-store", "data"),
    Input({"type": "feature-selector-autoencoder", "category": ALL}, "value"),
    Input({"type": "genetic-feature-selector-autoencoder", "category": ALL}, "value"),  # ✅ FIX: Input
    prevent_initial_call=True,
)
def update_autoencoder_genetic_features(
    genetic_features_store, selected_original_features, selected_genetic_features
):
    """Update genetic features display in autoencoder section and show combination summary."""
    
    print(f"DEBUG: Genetic features callback triggered")
    print(f"DEBUG: selected_original_features = {selected_original_features}")
    print(f"DEBUG: selected_genetic_features = {selected_genetic_features}")
    
    # Handle genetic features display
    genetic_ui = []

    # 🔧 FIX: Properly collect currently selected genetic features
    current_genetic_selection = []
    if selected_genetic_features:
        for features in selected_genetic_features:
            if features and isinstance(features, list):
                current_genetic_selection.extend(features)
            elif features:  # Handle case where it's a single item, not a list
                current_genetic_selection.append(features)

    print(f"DEBUG: current_genetic_selection = {current_genetic_selection}")

    if genetic_features_store and "feature_names" in genetic_features_store:
        feature_names = genetic_features_store["feature_names"]
        expressions = genetic_features_store.get("expressions", [])

        # Create checklist for genetic features with preserved selections
        genetic_ui = [
            dcc.Checklist(
                id={
                    "type": "genetic-feature-selector-autoencoder",
                    "category": "genetic",
                },
                options=[
                    {
                        "label": html.Div(
                            [
                                html.Span(
                                    f"{feat}",
                                    style={"fontWeight": "bold", "fontSize": "12px"},
                                ),
                                html.Br(),
                                html.Span(
                                    f"{expressions[i] if i < len(expressions) else 'Unknown'}",
                                    style={"fontSize": "10px", "color": "gray"},
                                ),
                            ]
                        ),
                        "value": feat,
                    }
                    for i, feat in enumerate(feature_names)
                ],
                value=current_genetic_selection,  # This will now properly preserve selections
                labelStyle={"display": "block", "marginBottom": "8px"},
            )
        ]
        
        print(f"DEBUG: Created genetic UI with {len(feature_names)} features")
        print(f"DEBUG: Preserved {len(current_genetic_selection)} selections")
        
    else:
        genetic_ui = [
            html.Div(
                [
                    html.I("No genetic features available."),
                    html.Br(),
                    html.I(
                        "Run Genetic Feature Discovery first.",
                        style={"fontSize": "11px"},
                    ),
                ],
                style={"color": "gray", "fontStyle": "italic"},
            )
        ]
        print(f"DEBUG: No genetic features available")

    # 🔧 FIX: Create more accurate combination summary
    all_original_features = []
    for features in selected_original_features:
        if features and isinstance(features, list):
            all_original_features.extend(features)
        elif features:  # Handle single items
            all_original_features.append(features)

    all_genetic_features = current_genetic_selection

    total_features = len(all_original_features) + len(all_genetic_features)

    print(f"DEBUG: Summary - Original: {len(all_original_features)}, Genetic: {len(all_genetic_features)}, Total: {total_features}")

    summary_content = []
    if total_features > 0:
        summary_content = [
            html.Div(
                [
                    html.Span(
                        "Selected Features Summary:",
                        style={"fontWeight": "bold", "fontSize": "12px"},
                    ),
                    html.Br(),
                    html.Span(
                        f"• Original features: {len(all_original_features)}",
                        style={"fontSize": "11px"},
                    ),
                    html.Br(),
                    html.Span(
                        f"• Genetic features: {len(all_genetic_features)}",
                        style={"fontSize": "11px", "fontWeight": "bold" if len(all_genetic_features) > 0 else "normal"},
                    ),
                    html.Br(),
                    html.Span(
                        f"• Total input features: {total_features}",
                        style={"fontSize": "11px", "fontWeight": "bold"},
                    ),
                ],
                style={"color": "black"},
            )
        ]
        
        # 🔧 NEW: Add detailed breakdown for debugging
        if len(all_genetic_features) > 0:
            summary_content.append(
                html.Div([
                    html.Br(),
                    html.Span(
                        f"Selected genetic features: {', '.join(all_genetic_features[:3])}{'...' if len(all_genetic_features) > 3 else ''}",
                        style={"fontSize": "10px", "color": "gray"}
                    )
                ])
            )
    else:
        summary_content = [
            html.Div(
                "Select features above to see combination summary",
                style={"color": "gray", "fontSize": "12px"}
            )
        ]

    print(f"DEBUG: Returning genetic UI and summary")
    return genetic_ui, summary_content


# Autoencoder Training callback


@callback(
    Output("mi-features-store", "data", allow_duplicate=True),
    Output("train-mi-autoencoder-status", "children", allow_duplicate=True),
    Input("train-mi-autoencoder", "n_clicks"),
    State("mi-latent-dim", "value"),
    State("mi-epochs", "value"),
    State("mi-batch-size", "value"),
    State("mi-learning-rate", "value"),
    State("mi-features-store", "data"),
    prevent_initial_call=True,
)
def train_mi_autoencoder(
    n_clicks, latent_dim, epochs, batch_size, learning_rate, mi_store
):
    """Train autoencoder on MI-selected features."""

    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if not mi_store or "selected_features" not in mi_store:
        return mi_store, "Run MI Feature Selection first!"

    try:
        # Load feature data
        feature_data_df = pd.read_json(mi_store["feature_data"], orient="split")
        file_labels_data = pd.read_json(mi_store["file_labels"], orient="split")

        # Extract file_label column properly
        if isinstance(file_labels_data, pd.DataFrame):
            if "file_label" in file_labels_data.columns:
                file_labels = file_labels_data["file_label"]
            else:
                # If it's a single column DataFrame, take the first column
                file_labels = file_labels_data.iloc[:, 0]
        else:
            # If it's already a Series
            file_labels = file_labels_data

        # Prepare data for autoencoder
        feature_matrix = feature_data_df.to_numpy()
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        feature_means = np.mean(feature_matrix, axis=0)
        feature_stds = np.std(feature_matrix, axis=0) + 1e-8
        normalized_features = (feature_matrix - feature_means) / feature_stds

        # Train autoencoder
        feature_tensor = torch.tensor(normalized_features, dtype=torch.float32)
        dataset = TensorDataset(feature_tensor)
        dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeepAutoencoder(
            input_dim=feature_matrix.shape[1], latent_dim=int(latent_dim)
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))

        # Training loop
        model.train()
        for epoch in range(int(epochs)):
            total_loss = 0
            for batch in dataloader:
                batch_data = batch[0].to(device)
                optimizer.zero_grad()

                # Forward pass - DeepAutoencoder returns (decoded, encoded)
                reconstructed, _ = model(batch_data)

                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Extract latent features
        model.eval()
        with torch.no_grad():
            # Use the encoder directly to get latent features
            latent_features = model.encoder(feature_tensor.to(device)).cpu().numpy()

        # Create latent dataframe with proper column names
        latent_df = pd.DataFrame(
            latent_features, columns=[f"Latent_{i}" for i in range(int(latent_dim))]
        )

        # Add file labels
        latent_df["file_label"] = file_labels.values

        # Update store with all necessary information
        mi_store["latent_features"] = latent_df.to_json(
            date_format="iso", orient="split"
        )
        mi_store["latent_dim"] = int(latent_dim)
        mi_store["step_completed"] = "autoencoder_trained"

        return (
            mi_store,
            f"Autoencoder training complete! Latent dimension: {latent_dim}",
        )

    except Exception as e:
        print(f"Error in train_mi_autoencoder: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return mi_store, f"Error training autoencoder: {str(e)}"


# UMAP Visualization callback
@callback(
    Output("mi-umap-graph", "figure"),
    Output("run-umap-mi-status", "children", allow_duplicate=True),
    Output("umap-quality-metrics-mi", "children"),
    Output(
        "mi-features-store", "data", allow_duplicate=True
    ),  # Update the store with UMAP MI scores
    Input("run-umap-mi", "n_clicks"),
    State("mi-features-store", "data"),
    State("metric-selector-mi", "value"),
    State("mi-umap-neighbors", "value"),
    State("mi-umap-min-dist", "value"),
    prevent_initial_call=True,
)
def run_umap_on_mi_features(
    n_clicks, mi_store, selected_metrics, n_neighbors, min_dist
):
    """Run UMAP on autoencoder latent features with MI analysis between features and UMAP dimensions."""

    if not n_clicks:
        return {}, "Click 'Run UMAP' to visualize.", [], mi_store

    if not mi_store or "latent_features" not in mi_store:
        return {}, "Train autoencoder first!", [], mi_store

    try:
        # Import required libraries
        from sklearn.feature_selection import mutual_info_regression

        # Load latent features
        latent_df = pd.read_json(mi_store["latent_features"], orient="split")
        latent_dim = mi_store.get("latent_dim", 7)

        # Extract latent features for UMAP
        X_latent = latent_df[[f"Latent_{i}" for i in range(latent_dim)]].to_numpy()

        # Run UMAP with user-specified parameters
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(n_neighbors) if n_neighbors else 15,
            min_dist=float(min_dist) if min_dist else 0.1,
            metric="euclidean",
            random_state=42,
        )

        umap_result = reducer.fit_transform(X_latent)

        # Create UMAP dataframe
        umap_df = pd.DataFrame(
            {
                "UMAP1": umap_result[:, 0],
                "UMAP2": umap_result[:, 1],
                "file_label": latent_df["file_label"].values,
            }
        )

        # Create figure
        fig = go.Figure()

        # Add traces for each file label
        for label in umap_df["file_label"].unique():
            mask = umap_df["file_label"] == label
            df_subset = umap_df[mask]

            fig.add_trace(
                go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode="markers",
                    marker=dict(size=8, opacity=0.7),
                    name=f"{label} ({len(df_subset)} pts)",
                )
            )

        fig.update_layout(
            title=f"UMAP of MI-Selected Features (Latent Dim: {latent_dim}, n_neighbors: {n_neighbors}, min_dist: {min_dist})",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            height=550,
        )

        # Calculate metrics including MI analysis
        metrics_children = []

        # Compute Mutual Information between original features and UMAP dimensions
        if "feature_data" in mi_store:
            try:
                # Load the original feature data
                feature_data_df = pd.read_json(mi_store["feature_data"], orient="split")
                selected_features = mi_store.get("selected_features", [])

                if selected_features:
                    # Get the feature matrix
                    feature_matrix = feature_data_df[selected_features].to_numpy()

                    # Handle NaN/inf values
                    feature_matrix = np.nan_to_num(
                        feature_matrix, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    # Compute MI between each feature and each UMAP dimension
                    umap_mi_scores = {}
                    for i, dim in enumerate(["UMAP1", "UMAP2"]):
                        umap_mi_scores[dim] = mutual_info_regression(
                            feature_matrix, umap_result[:, i]
                        )

                    # Average MI scores across both UMAP dimensions
                    avg_mi_scores = np.mean(list(umap_mi_scores.values()), axis=0)

                    # Create a dictionary mapping each feature name to its average MI score
                    mi_scores_dict_umap = dict(zip(selected_features, avg_mi_scores))

                    # Sort features by MI score (highest first)
                    sorted_features_umap = sorted(
                        mi_scores_dict_umap, key=mi_scores_dict_umap.get, reverse=True
                    )

                    # Also calculate correlation between features and UMAP dimensions
                    corr_scores = {}
                    for i, dim in enumerate(["UMAP1", "UMAP2"]):
                        corr_scores[dim] = []
                        for j, feature in enumerate(selected_features):
                            corr = np.corrcoef(feature_matrix[:, j], umap_result[:, i])[
                                0, 1
                            ]
                            corr_scores[dim].append(corr)

                    # Average absolute correlation across dimensions
                    avg_corr_scores = np.mean(
                        [np.abs(corr_scores["UMAP1"]), np.abs(corr_scores["UMAP2"])],
                        axis=0,
                    )
                    corr_scores_dict_umap = dict(
                        zip(selected_features, avg_corr_scores)
                    )

                    # TODO: Is this needed?
                    # Sort features by correlation score (highest first)
                    # sorted_features_corr = sorted(
                    #     corr_scores_dict_umap,
                    #     key=corr_scores_dict_umap.get,
                    #     reverse=True,
                    # )

                    # Store UMAP MI scores in the store
                    mi_store["umap_mi_scores"] = mi_scores_dict_umap
                    mi_store["umap_corr_scores"] = corr_scores_dict_umap

                    # Create a comprehensive table showing all features
                    feature_data_for_table = []
                    for (
                        feature
                    ) in sorted_features_umap:  # Show all features sorted by MI
                        feature_data_for_table.append(
                            {
                                "Feature": feature,
                                "MI Score": f"{mi_scores_dict_umap[feature]:.4f}",
                                "Correlation": f"{corr_scores_dict_umap[feature]:.4f}",
                            }
                        )

                    # TODO: Consider creating a function that lives in /src/sculpt/utils/ui.py (maybe)
                    # Create the MI analysis UI with a table showing all features
                    mi_analysis_ui = html.Div(
                        [
                            html.H4(
                                "Feature Contribution to UMAP Visualization",
                                style={
                                    "fontSize": "16px",
                                    "color": "#1976d2",
                                    "marginBottom": "10px",
                                    "marginTop": "20px",
                                },
                            ),
                            html.Div(
                                [
                                    html.P(
                                        f"Analysis of how all {len(selected_features)} selected features contribute to the "
                                        "UMAP clustering:",
                                        style={
                                            "fontSize": "12px",
                                            "color": "#666",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    # Summary statistics
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Average MI Score: ",
                                                        style={"fontWeight": "bold"},
                                                    ),
                                                    html.Span(
                                                        f"{np.mean(list(mi_scores_dict_umap.values())):.4f}"
                                                    ),
                                                ],
                                                style={
                                                    "display": "inline-block",
                                                    "marginRight": "30px",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Average Correlation: ",
                                                        style={"fontWeight": "bold"},
                                                    ),
                                                    html.Span(
                                                        f"{np.mean(list(corr_scores_dict_umap.values())):.4f}"
                                                    ),
                                                ],
                                                style={"display": "inline-block"},
                                            ),
                                        ],
                                        style={
                                            "fontSize": "12px",
                                            "marginBottom": "15px",
                                            "padding": "10px",
                                            "backgroundColor": "#f5f5f5",
                                            "borderRadius": "5px",
                                        },
                                    ),
                                    # Full feature table
                                    dash_table.DataTable(
                                        columns=[
                                            {
                                                "name": "Feature",
                                                "id": "Feature",
                                                "type": "text",
                                            },
                                            {
                                                "name": "MI with UMAP",
                                                "id": "MI Score",
                                                "type": "text",
                                            },
                                            {
                                                "name": "Correlation with UMAP",
                                                "id": "Correlation",
                                                "type": "text",
                                            },
                                        ],
                                        data=feature_data_for_table,
                                        style_cell={
                                            "textAlign": "left",
                                            "fontSize": "11px",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"row_index": 0},
                                                "backgroundColor": "#e3f2fd",
                                                "fontWeight": "bold",
                                            },
                                            {
                                                "if": {"row_index": 1},
                                                "backgroundColor": "#e8f5e9",
                                            },
                                            {
                                                "if": {"row_index": 2},
                                                "backgroundColor": "#fff3e0",
                                            },
                                        ],
                                        style_header={
                                            "backgroundColor": "#1976d2",
                                            "fontWeight": "bold",
                                            "fontSize": "12px",
                                            "color": "white",
                                        },
                                        page_size=10,  # Show 10 features per page
                                        style_table={
                                            "maxHeight": "300px",
                                            "overflowY": "auto",
                                        },
                                        sort_action="native",
                                        filter_action="native",
                                    ),
                                    # Interpretation guide
                                    html.Div(
                                        [
                                            html.Details(
                                                [
                                                    html.Summary(
                                                        "How to interpret these scores",
                                                        style={
                                                            "cursor": "pointer",
                                                            "color": "#666",
                                                            "fontSize": "12px",
                                                            "marginTop": "10px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.P(
                                                                "• Mutual Information (MI) captures both linear and non-linear"
                                                                " relationships between features and UMAP dimensions",
                                                                style={
                                                                    "fontSize": "11px",
                                                                    "marginBottom": "5px",
                                                                },
                                                            ),
                                                            html.P(
                                                                "• Higher MI scores indicate features that strongly influence"
                                                                " the UMAP clustering structure",
                                                                style={
                                                                    "fontSize": "11px",
                                                                    "marginBottom": "5px",
                                                                },
                                                            ),
                                                            html.P(
                                                                "• Correlation measures only linear relationships and can"
                                                                " identify features with consistent directional influence",
                                                                style={
                                                                    "fontSize": "11px",
                                                                    "marginBottom": "5px",
                                                                },
                                                            ),
                                                            html.P(
                                                                "• Features high in both metrics are the most important for"
                                                                " the visualization",
                                                                style={
                                                                    "fontSize": "11px"
                                                                },
                                                            ),
                                                            html.P(
                                                                "• The table is sortable and filterable - click column headers"
                                                                " to sort or use the filter boxes",
                                                                style={
                                                                    "fontSize": "11px",
                                                                    "fontStyle": "italic",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "backgroundColor": "#fafafa",
                                                            "padding": "10px",
                                                            "borderRadius": "5px",
                                                            "marginTop": "5px",
                                                        },
                                                    ),
                                                ]
                                            )
                                        ]
                                    ),
                                ],
                                style={
                                    "border": "1px solid #e0e0e0",
                                    "padding": "15px",
                                    "borderRadius": "5px",
                                    "backgroundColor": "white",
                                    "marginTop": "15px",
                                },
                            ),
                        ]
                    )

                    metrics_children.append(mi_analysis_ui)

            except Exception as e:
                print(f"Error computing MI with UMAP dimensions: {str(e)}")
                import traceback

                traceback.print_exc()

        # Calculate standard clustering metrics if selected
        if selected_metrics and len(selected_metrics) > 0:
            try:
                # Initialize metrics dictionary
                metrics = {}

                # Prepare data for clustering
                X_umap_scaled = umap_result.copy()

                # Run DBSCAN for clustering-based metrics
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler

                # Scale the UMAP coordinates
                scaler = StandardScaler()
                X_umap_scaled = scaler.fit_transform(X_umap_scaled)

                # Run DBSCAN
                dbscan = DBSCAN(eps=0.3, min_samples=5)
                cluster_labels = dbscan.fit_predict(X_umap_scaled)

                # Calculate selected metrics
                if (
                    "silhouette" in selected_metrics
                    and len(np.unique(cluster_labels)) > 1
                ):
                    from sklearn.metrics import silhouette_score

                    metrics["silhouette"] = silhouette_score(
                        X_umap_scaled, cluster_labels
                    )

                if (
                    "davies_bouldin" in selected_metrics
                    and len(np.unique(cluster_labels)) > 1
                ):
                    from sklearn.metrics import davies_bouldin_score

                    metrics["davies_bouldin"] = davies_bouldin_score(
                        X_umap_scaled, cluster_labels
                    )

                if (
                    "calinski_harabasz" in selected_metrics
                    and len(np.unique(cluster_labels)) > 1
                ):
                    from sklearn.metrics import calinski_harabasz_score

                    metrics["calinski_harabasz"] = calinski_harabasz_score(
                        X_umap_scaled, cluster_labels
                    )

                # Create metrics display
                if metrics:
                    metrics_display = html.Div(
                        [
                            html.H4(
                                "Clustering Quality Metrics",
                                style={"fontSize": "14px", "marginBottom": "10px"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                f"{metric.replace('_', ' ').title()}: ",
                                                style={
                                                    "fontWeight": "bold",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                            html.Span(
                                                f"{value:.4f}",
                                                style={"fontSize": "12px"},
                                            ),
                                        ],
                                        style={"marginBottom": "5px"},
                                    )
                                    for metric, value in metrics.items()
                                ]
                            ),
                        ],
                        style={
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5",
                            "borderRadius": "5px",
                            "marginTop": "10px",
                        },
                    )

                    metrics_children.append(metrics_display)

            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")

        return (
            fig,
            "UMAP visualization complete with feature importance analysis!",
            metrics_children,
            mi_store,
        )

    except Exception as e:
        print(f"Error running UMAP: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {}, f"Error running UMAP: {str(e)}", [], mi_store


# Scatter Plot callback


@callback(
    Output("mi-feature-scatter", "figure"),
    Input("update-mi-scatter-btn", "n_clicks"),
    State("mi-scatter-x-feature", "value"),
    State("mi-scatter-y-feature", "value"),
    State("mi-features-store", "data"),
    prevent_initial_call=True,
)
def update_mi_scatter_plot(n_clicks, x_feature, y_feature, mi_store):
    """Update the scatter plot of selected features."""

    if not n_clicks or not x_feature or not y_feature:
        return (
            go.Figure()
            .add_annotation(
                x=0.5,
                y=0.5,
                text="Select features and click 'Update Scatter Plot'",
                showarrow=False,
                font=dict(size=16),
            )
            .update_layout(
                title="Feature Scatter Plot",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                height=400,
            )
        )

    if not mi_store or "feature_data" not in mi_store:
        return (
            go.Figure()
            .add_annotation(
                x=0.5,
                y=0.5,
                text="Run MI Feature Selection first",
                showarrow=False,
                font=dict(size=16),
            )
            .update_layout(height=400)
        )

    try:
        # Load feature data
        feature_data_df = pd.read_json(mi_store["feature_data"], orient="split")
        file_labels_data = pd.read_json(mi_store["file_labels"], orient="split")

        # Extract file labels properly
        if isinstance(file_labels_data, pd.DataFrame):
            if "file_label" in file_labels_data.columns:
                file_labels = file_labels_data["file_label"]
            else:
                file_labels = file_labels_data.iloc[:, 0]
        else:
            file_labels = file_labels_data

        # Create scatter plot
        fig = go.Figure()

        # Add traces for each file label
        for label in file_labels.unique():
            mask = file_labels == label

            fig.add_trace(
                go.Scatter(
                    x=feature_data_df[x_feature][mask],
                    y=feature_data_df[y_feature][mask],
                    mode="markers",
                    marker=dict(size=6, opacity=0.7),
                    name=f"{label}",
                )
            )

        fig.update_layout(
            title=f"Feature Scatter Plot: {x_feature} vs {y_feature}",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            height=400,
        )

        return fig

    except Exception as e:
        return (
            go.Figure()
            .add_annotation(
                x=0.5,
                y=0.5,
                text=f"Error: {str(e)}",
                showarrow=False,
                font=dict(size=16),
            )
            .update_layout(height=400)
        )


@callback(
    Output("mi-features-status", "children", allow_duplicate=True),
    Output("train-mi-autoencoder-status", "children", allow_duplicate=True),
    Output("run-umap-mi-status", "children", allow_duplicate=True),
    Input("run-mi-features", "n_clicks"),
    Input("train-mi-autoencoder", "n_clicks"),
    Input("run-umap-mi", "n_clicks"),
    Input("mi-umap-graph", "figure"),
    prevent_initial_call=True,
)
def update_mi_status(mi_clicks, train_clicks, umap_clicks, umap_fig):
    """Update status messages for MI section."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Initialize all status values
    mi_status = dash.no_update
    train_status = dash.no_update
    umap_status = dash.no_update

    if trigger_id == "run-mi-features":
        mi_status = "Running MI feature selection..."
    elif trigger_id == "train-mi-autoencoder":
        train_status = "Training autoencoder on selected features..."
    elif trigger_id == "run-umap-mi":
        umap_status = "Running UMAP on latent features..."
    elif trigger_id == "mi-umap-graph":
        umap_status = "UMAP visualization complete!"

    return mi_status, train_status, umap_status


# Add this callback to update the MI feature importance table
@callback(
    Output("mi-feature-importance-table", "data"),
    Input("mi-feature-search-button", "n_clicks"),
    Input("mi-feature-sort-option", "value"),
    Input("mi-features-store", "data"),  # Trigger update when MI analysis completes
    State("mi-feature-search-input", "value"),
    State("mi-features-store", "data"),
    prevent_initial_call=True,
)
def update_mi_feature_importance_table(
    n_clicks, sort_option, mi_store_trigger, search_term, mi_store
):
    """Update the MI feature importance table based on search and sort options."""
    ctx = dash.callback_context
    if not ctx.triggered or not mi_store or "mi_scores" not in mi_store:
        return []

    try:
        # Get MI scores and selected features from the store
        mi_scores_dict = mi_store.get("mi_scores", {})
        selected_features = mi_store.get("selected_features", [])

        # Get UMAP MI scores if available
        umap_mi_scores = mi_store.get("umap_mi_scores", {})
        umap_corr_scores = mi_store.get("umap_corr_scores", {})

        if not mi_scores_dict:
            return []

        # Create DataFrame with ALL features and scores
        all_features = list(mi_scores_dict.keys())
        feature_importance_df = pd.DataFrame(
            {
                "Feature": all_features,
                "MI_Score": [mi_scores_dict.get(f, 0.0) for f in all_features],
                "Is_Selected": [f in selected_features for f in all_features],
            }
        )

        # Add UMAP scores if available
        if umap_mi_scores:
            feature_importance_df["UMAP_MI"] = [
                umap_mi_scores.get(f, 0.0) for f in all_features
            ]
            feature_importance_df["UMAP_Corr"] = [
                umap_corr_scores.get(f, 0.0) for f in all_features
            ]

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
                "MI_Score", ascending=False
            )
        elif sort_option == "order":
            # First show selected features in their selection order, then others by MI score
            selected_df = feature_importance_df[feature_importance_df["Is_Selected"]]
            unselected_df = feature_importance_df[~feature_importance_df["Is_Selected"]]

            # Sort selected by their order in selected_features list
            selected_df["order"] = selected_df["Feature"].apply(
                lambda x: selected_features.index(x) if x in selected_features else 999
            )
            selected_df = selected_df.sort_values("order")

            # Sort unselected by MI score
            unselected_df = unselected_df.sort_values("MI_Score", ascending=False)

            # Combine
            feature_importance_df = pd.concat([selected_df, unselected_df])
        elif sort_option == "umap_mi" and "UMAP_MI" in feature_importance_df.columns:
            feature_importance_df = feature_importance_df.sort_values(
                "UMAP_MI", ascending=False
            )
        elif (
            sort_option == "umap_corr" and "UMAP_Corr" in feature_importance_df.columns
        ):
            feature_importance_df = feature_importance_df.sort_values(
                "UMAP_Corr", ascending=False
            )

        # Reset index for proper ranking
        feature_importance_df = feature_importance_df.reset_index(drop=True)

        # Prepare the table data
        table_data = []
        for i, row in feature_importance_df.iterrows():
            table_row = {
                "Rank": i + 1,
                "Feature": row["Feature"],
                "MI Score": f"{row['MI_Score']:.4f}",
                "Selected": (
                    "✓" if row["Is_Selected"] else ""
                ),  # Using checkmark for selected
            }

            # Add UMAP columns if they exist
            if "UMAP_MI" in feature_importance_df.columns:
                table_row["UMAP MI"] = (
                    f"{row.get('UMAP_MI', 0.0):.4f}"
                    if row.get("UMAP_MI", 0.0) > 0
                    else "-"
                )
                table_row["UMAP Corr"] = (
                    f"{row.get('UMAP_Corr', 0.0):.4f}"
                    if row.get("UMAP_Corr", 0.0) > 0
                    else "-"
                )

            table_data.append(table_row)

        return table_data

    except Exception as e:
        print(f"Error updating MI feature importance table: {str(e)}")
        import traceback

        traceback.print_exc()
        return []


@callback(
    Output("features-data-store", "data", allow_duplicate=True),
    Output("feature-selection-ui-graph1", "children", allow_duplicate=True),
    Output("feature-selection-ui-graph3", "children", allow_duplicate=True),
    Output("feature-selection-ui-graph3-selection", "children", allow_duplicate=True),
    Output("feature-calculation-status", "children"),
    Input("file-config-assignments-store", "data"),
    State("stored-files", "data"),
    State("configuration-profiles-store", "data"),
    State("features-data-store", "data"),
    prevent_initial_call=True,
)
def calculate_features_after_assignment(
    assignments_store, stored_files, profiles_store, features_store
):
    """Calculate physics features for all files after profile assignments are made."""
    if not assignments_store or not stored_files:
        raise dash.exceptions.PreventUpdate

    if features_store is None:
        features_store = {}

    status_messages = []
    all_feature_columns = set()

    # Process each file with its assigned profile
    for file_info in stored_files:
        if file_info.get("is_selection", False):
            continue  # Skip selection files

        filename = file_info["filename"]
        profile_name = assignments_store.get(filename)

        if profile_name == "none" or not profile_name:
            status_messages.append(f"Skipping {filename} (no profile assigned)")
            continue

        try:
            # Load the dataframe
            df = pd.read_json(file_info["data"], orient="split")

            # Sample for feature calculation
            sample_size = min(1000, len(df))
            df_sample = (
                df.sample(n=sample_size, random_state=42)
                if len(df) > sample_size
                else df
            )

            # Calculate features with the assigned profile
            if profile_name in profiles_store:
                profile_config = profiles_store[profile_name]
                df_features = calculate_physics_features_with_profile(
                    df_sample, profile_config
                )
                status_messages.append(
                    f"✓ {filename}: calculated with profile '{profile_name}'"
                )
            else:
                # This shouldn't happen but handle gracefully
                df_features = calculate_physics_features_flexible(df_sample, None)
                status_messages.append(
                    f"⚠ {filename}: profile '{profile_name}' not found, used default"
                )

            # Collect all feature columns
            all_feature_columns.update(df_features.columns)

        except Exception as e:
            status_messages.append(f"✗ {filename}: error - {str(e)}")
            print(f"Error calculating features for {filename}: {e}")
            import traceback

            traceback.print_exc()

    # Update features store with all possible columns
    features_store["column_names"] = sorted(list(all_feature_columns))

    # Create feature selection UIs
    feature_ui_graph1 = create_feature_categories_ui(
        features_store["column_names"], "graph1"
    )
    feature_ui_graph3 = create_feature_categories_ui(
        features_store["column_names"], "graph3"
    )
    feature_ui_graph3_selection = create_feature_categories_ui(
        features_store["column_names"], "graph3-selection"
    )

    # Create status message
    status_div = html.Div(
        [
            html.H5("Feature Calculation Results:", style={"marginBottom": "10px"}),
            html.Div(
                [html.Div(msg) for msg in status_messages],
                style={"fontSize": "12px", "lineHeight": "1.5"},
            ),
        ],
        style={
            "marginTop": "15px",
            "padding": "10px",
            "backgroundColor": "#f0f0f0",
            "borderRadius": "5px",
        },
    )

    return (
        features_store,
        feature_ui_graph1,
        feature_ui_graph3,
        feature_ui_graph3_selection,
        status_div,
    )
