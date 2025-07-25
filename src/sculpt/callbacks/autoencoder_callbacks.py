import dash
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap
from dash import ALL, Input, Output, State, callback, html
from torch.utils.data import DataLoader, TensorDataset

from sculpt.models.deep_autoencoder import DeepAutoencoder
from sculpt.utils.metrics.clustering_quality import cluster_stability, hopkins_statistic


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
    State("umap-graph", "figure"),
    State("autoencoder-latent-store", "data"),
    State("metric-selector-autoencoder", "value"),
    background=True,
    running=[
        (Output("train-autoencoder", "disabled"), True, False),
        (Output("training-progress", "children"), "Training...", ""),
    ],
    progress=[Output("training-progress", "children")],
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
    original_figure,
    latent_store,
    selected_metrics,
    set_progress,
):
    """Train autoencoder and run UMAP on latent space."""
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
            return (
                {},
                "No combined dataset available. Please run UMAP first.",
                {},
                [],
                [],
            )

        # Check if we need to train the autoencoder or just run UMAP on existing latent space
        if trigger_id == "run-umap-latent" and latent_store:
            debug_text.append("Using previously trained autoencoder latent space.")
            should_train = False
        else:
            should_train = True
            debug_text.append("Training new autoencoder.")

        if should_train:
            # Step 1: Prepare data based on selected source
            # -----------------------------------------
            debug_text.append(f"Data source: {data_source}")
            debug_text.append(f"Latent dimension: {latent_dim}")
            debug_text.append(f"Epochs: {epochs}")

            # Load UMAP coordinates
            # TODO: Check why umap_coords are needed
            if (
                "umap_coords" in combined_data_json
                and combined_data_json["umap_coords"] != "{}"
            ):
                umap_coords = pd.read_json(  # noqa: F841
                    combined_data_json["umap_coords"], orient="split"
                )

            # Collect all selected features for autoencoder
            all_selected_features = []
            for features in selected_features_list:
                if features:  # Only add non-empty lists
                    all_selected_features.extend(features)

            if not all_selected_features:
                # Use particle momentum as default if nothing is selected
                all_selected_features = [
                    col for col in combined_df.columns if col.startswith("particle_")
                ]
                debug_text.append(
                    f"No features selected, using {len(all_selected_features)} default momentum features"
                )
            else:
                debug_text.append(
                    f"Using {len(all_selected_features)} selected features"
                )

            # Default to all data
            df_for_training = combined_df.copy()
            labels = combined_df["file_label"].copy()
            debug_text.append(f"Using all {len(df_for_training)} rows for training")

            # Extract only the selected features
            feature_cols = [
                col for col in df_for_training.columns if col in all_selected_features
            ]

            if not feature_cols:
                return {}, "No valid features selected for the autoencoder.", {}, [], []

            # Extract feature data
            feature_data = df_for_training[feature_cols].to_numpy()

            # Handle NaN/inf values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Create PyTorch dataset and dataloader
            feature_tensor = torch.tensor(feature_data, dtype=torch.float32)
            dataset = TensorDataset(feature_tensor)
            dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

            # Set up device (CPU or GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            debug_text.append(f"Using device: {device}")

            # Initialize model
            input_dim = len(feature_cols)
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
                set_progress(f"Training progress: Epoch {epoch + 1}/{num_epochs}")
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
        fig.update_layout(
            height=600,
            title=f"UMAP of Autoencoder Latent Space (dim={latent_dim}, n_neighbors={n_neighbors}, min_dist={min_dist})",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data File",
        )

        # Add latent UMAP coordinates to the latent store
        latent_store["umap_coords"] = umap_df.to_json(date_format="iso", orient="split")

        # Calculate mutual information between original features and UMAP dimensions
        debug_text.append(
            "Calculating mutual information between original features and UMAP dimensions..."
        )

        # We need to get back to the original features that were used
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
                    # Only use the columns that exist
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
                # This provides a simpler linear relationship measure to compare with MI
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
                            style={"fontSize": "14px", "color": "#2e7d32"},
                        ),
                        # Summary section - top features
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            "Top Features by Mutual Information:",
                                            style={
                                                "fontSize": "13px",
                                                "marginBottom": "5px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            f"{i+1}. {feature}: ",
                                                            style={
                                                                "fontWeight": "bold"
                                                            },
                                                        ),
                                                        html.Span(
                                                            f"{mi_scores_dict[feature]:.4f}"
                                                        ),
                                                    ]
                                                )
                                                for i, feature in enumerate(
                                                    sorted_features[:10]
                                                )
                                            ],
                                            style={
                                                "marginLeft": "10px",
                                                "fontSize": "12px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "50%",
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H5(
                                            "Top Features by Correlation:",
                                            style={
                                                "fontSize": "13px",
                                                "marginBottom": "5px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            f"{i+1}. {feature}: ",
                                                            style={
                                                                "fontWeight": "bold"
                                                            },
                                                        ),
                                                        html.Span(
                                                            f"{corr_scores_dict[feature]:.4f}"
                                                        ),
                                                    ]
                                                )
                                                for i, feature in enumerate(
                                                    sorted_features_corr[:10]
                                                )
                                            ],
                                            style={
                                                "marginLeft": "10px",
                                                "fontSize": "12px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "50%",
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                    },
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap"},
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
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.P(
                                                    "• Mutual Information (MI) measures nonlinear relationships between "
                                                    "features and UMAP dimensions"
                                                ),
                                                html.P(
                                                    "• Correlation measures linear relationships only"
                                                ),
                                                html.P(
                                                    "• Higher values indicate features that strongly influence the latent "
                                                    "space clustering"
                                                ),
                                                html.P(
                                                    "• Features appearing high in both metrics have strong overall influence"
                                                ),
                                                html.P(
                                                    "• Features high in MI but low in correlation have primarily nonlinear "
                                                    "influence"
                                                ),
                                            ],
                                            style={
                                                "fontSize": "11px",
                                                "paddingLeft": "10px",
                                            },
                                        ),
                                    ]
                                )
                            ],
                            style={"marginTop": "10px"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#f1f8e9",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "marginTop": "15px",
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
