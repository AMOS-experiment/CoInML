import re
import traceback
from itertools import combinations

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap
from dash import ALL, Input, Output, State, callback, html
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from sculpt.models.deep_autoencoder import DeepAutoencoder
from sculpt.utils.file_handlers import extract_selection_indices
from sculpt.utils.metrics.clustering_quality import cluster_stability, hopkins_statistic


@callback(
    Output("mi-features-graph", "figure"),
    Output("mi-features-debug-output", "children"),
    Output("mi-features-store", "data"),
    Output("run-umap-mi-status", "children", allow_duplicate=True),
    Output("umap-quality-metrics-mi", "children"),  # Add this output
    Input("run-mi-features", "n_clicks"),
    Input("run-umap-mi", "n_clicks"),
    State("mi-data-source", "value"),
    State("mi-target-variables", "value"),
    State("mi-redundancy-threshold", "value"),
    State("mi-max-features", "value"),
    State("autoencoder-latent-dim", "value"),
    State("autoencoder-epochs", "value"),
    State("autoencoder-batch-size", "value"),
    State("autoencoder-learning-rate", "value"),
    State({"type": "feature-selector-mi", "category": ALL}, "value"),
    State("selected-points-store", "data"),
    State("selected-points-run-store", "data"),
    State("combined-data-store", "data"),
    State("umap-graph", "figure"),
    State("mi-features-store", "data"),
    State("metric-selector-mi", "value"),  # Add this state
    prevent_initial_call=True,
)
def run_mi_feature_selection_and_umap(
    run_mi_clicks,
    run_umap_clicks,
    data_source,
    target_variables,
    redundancy_threshold,
    max_features,
    latent_dim,
    epochs,
    batch_size,
    learning_rate,
    selected_features_list,
    graph1_selection,
    graph3_selection,
    combined_data_json,
    original_figure,
    mi_features_store,
    selected_metrics,
):
    """Run mutual information feature selection followed by autoencoder for dimensionality reduction."""

    # Initialize debug info and check which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, "No action triggered.", {}, "", []

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    debug_text = []

    # If UMAP button was clicked and we have stored features, skip the MI part
    if (
        trigger_id == "run-umap-mi"
        and mi_features_store
        and "latent_features" in mi_features_store
    ):
        debug_text.append(
            "Running UMAP visualization on previously computed MI-based features..."
        )

        try:
            # Load the latent features from the store
            latent_df = pd.read_json(
                mi_features_store["latent_features"], orient="split"
            )
            selected_features = mi_features_store.get("selected_features", [])

            if latent_df.empty:
                return (
                    {},
                    "No MI-based features found. Run MI Feature Selection first.",
                    mi_features_store,
                    "",
                    [],
                )

            # Run UMAP on the latent space
            latent_dim = mi_features_store.get("latent_dim", 7)
            X_latent = latent_df[[f"Latent_{i}" for i in range(latent_dim)]].to_numpy()

            # Run UMAP
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                random_state=42,
            )

            umap_result = reducer.fit_transform(X_latent)

            # Create DataFrame for UMAP visualization
            umap_df = pd.DataFrame(
                {
                    "UMAP1": umap_result[:, 0],
                    "UMAP2": umap_result[:, 1],
                    "file_label": latent_df["file_label"].values,
                }
            )

            # Create visualization

            fig = go.Figure()

            # Extract color information from original figure
            color_map = {}
            if original_figure and "data" in original_figure:
                for trace in original_figure["data"]:
                    if (
                        "name" in trace
                        and "marker" in trace
                        and "color" in trace["marker"]
                    ):
                        # Clean the label if it contains point count
                        clean_name = trace["name"]
                        if " (" in clean_name:
                            clean_name = clean_name.split(" (")[0]
                        color_map[clean_name] = trace["marker"]["color"]

            # Add traces for each file label with consistent colors
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
                title=f"UMAP of MI-Selected Features and Autoencoder (latent dim={latent_dim})",
                xaxis_title="UMAP1",
                yaxis_title="UMAP2",
                legend_title="Data File",
            )

            # Add information about which features were selected
            debug_text.append(
                f"Selected {len(selected_features)} features using mutual information:"
            )
            debug_text.append(
                ", ".join(selected_features[:10])
                + ("..." if len(selected_features) > 10 else "")
            )

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

                    if (
                        n_clusters >= 2
                        and noise_ratio < 0.5
                        and n_clusters > max_clusters
                    ):
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
                                        "Estimated Clusters: ",
                                        style={"fontWeight": "bold"},
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
                                        "DBSCAN eps: ", style={"fontWeight": "bold"}
                                    ),
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
                                                    if metrics.get("silhouette", 0)
                                                    > 0.5
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
                                                            "davies_bouldin",
                                                            float("inf"),
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
                                                    if metrics.get(
                                                        "calinski_harabasz", 0
                                                    )
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
                                                        if metrics.get("hopkins", 0)
                                                        > 0.6
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
                                                        if metrics.get("stability", 0)
                                                        > 0.6
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
                            html.Hr(
                                style={"marginTop": "10px", "marginBottom": "10px"}
                            ),
                            html.Details(
                                [
                                    html.Summary(
                                        "What do these metrics mean?",
                                        style={"cursor": "pointer"},
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                "• Silhouette Score: Measures how well-separated clusters are (higher is "
                                                "better, range: -1 to 1)"
                                            ),
                                            html.P(
                                                "• Davies-Bouldin Index: Measures average similarity between clusters (lower"
                                                " is better, range: 0 to ∞)"
                                            ),
                                            html.P(
                                                "• Calinski-Harabasz Index: Ratio of between-cluster to within-cluster "
                                                "dispersion (higher is better, range: 0 to ∞)"
                                            ),
                                            html.P(
                                                "• Hopkins Statistic: Measures clusterability of the data (>0.75 indicates "
                                                "good clustering, range: 0 to 1)"
                                            ),
                                            html.P(
                                                "• Cluster Stability: How stable clusters are with small perturbations (higher"
                                                " is better, range: 0 to 1)"
                                            ),
                                            html.P(
                                                "• Physics Consistency: How well clusters align with physical parameters "
                                                "(higher is better, range: 0 to 1)"
                                            ),
                                        ],
                                        style={
                                            "fontSize": "11px",
                                            "paddingLeft": "10px",
                                        },
                                    ),
                                ]
                            ),
                        ],
                        style={"marginTop": "10px"},
                    )
                )

            except Exception as e:

                trace = traceback.format_exc()
                metrics_children = [html.Div(f"Error calculating metrics: {str(e)}")]

            return (
                fig,
                "<br>".join(debug_text),
                mi_features_store,
                "UMAP visualization complete!",
                metrics_children,
            )

        except Exception as e:

            trace = traceback.format_exc()
            error_message = f"Error running UMAP on MI features: {str(e)}<br>{trace}"
            return {}, error_message, mi_features_store, "", []

    # If we're here, we're running the full MI feature selection
    if trigger_id == "run-mi-features":
        debug_text.append(f"Data source: {data_source}")
        debug_text.append(f"Target variables: {target_variables}")
        debug_text.append(f"Redundancy threshold: {redundancy_threshold}")
        debug_text.append(f"Maximum features: {max_features}")

        try:
            # Step 1: Prepare data based on selected source
            # -----------------------------------------

            # Load the combined dataframe
            if (
                combined_data_json
                and "combined_df" in combined_data_json
                and combined_data_json["combined_df"] != "{}"
            ):
                combined_df = pd.read_json(
                    combined_data_json["combined_df"], orient="split"
                )
                debug_text.append(
                    f"Loaded combined dataset with {len(combined_df)} rows"
                )
            else:
                return (
                    {},
                    "No combined dataset available. Please run UMAP first.",
                    {},
                    "",
                    [],
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

            # Collect all selected features for MI analysis
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

            # Verify that target variables are valid
            if not target_variables or len(target_variables) == 0:
                # Default targets if none are specified
                target_variables = ["KER", "EESum", "TotalEnergy"]
                debug_text.append(
                    f"Using default target variables: {', '.join(target_variables)}"
                )

            # Check if targets exist in the data
            valid_targets = [t for t in target_variables if t in combined_df.columns]
            if not valid_targets:
                return (
                    {},
                    "No valid target variables found in the data. Please run feature extraction first.",
                    {},
                    "",
                    [],
                )

            debug_text.append(f"Using valid targets: {', '.join(valid_targets)}")

            # Prepare data based on data source
            if data_source == "all":
                # Use all data
                df_for_analysis = combined_df.copy()
                labels = combined_df["file_label"].copy()
                debug_text.append(f"Using all {len(df_for_analysis)} rows for analysis")

            elif (
                data_source == "graph1-selection"
                and graph1_selection
                and umap_coords is not None
            ):
                # Use selection from Graph 1
                indices = extract_selection_indices(graph1_selection, umap_coords)
                if not indices:
                    return {}, "No valid points found in Graph 1 selection.", {}, "", []

                df_for_analysis = combined_df.iloc[indices].copy()
                labels = df_for_analysis["file_label"].copy()
                debug_text.append(
                    f"Selected {len(df_for_analysis)} rows from Graph 1 selection"
                )

            elif (
                data_source == "graph3-selection"
                and graph3_selection
                and graph3_subset_df is not None
            ):
                # Use selection from Graph 3
                df_for_analysis = graph3_subset_df.copy()
                labels = df_for_analysis["file_label"].copy()
                debug_text.append(
                    f"Using Graph 3 selection with {len(df_for_analysis)} rows"
                )

            else:
                # Default to all data
                df_for_analysis = combined_df.copy()
                labels = combined_df["file_label"].copy()
                debug_text.append(f"Defaulting to all {len(df_for_analysis)} rows")

            # Extract feature data - only include columns that are in all_selected_features
            feature_cols = [
                col for col in df_for_analysis.columns if col in all_selected_features
            ]

            if not feature_cols:
                return {}, "No valid features selected for MI analysis.", {}, "", []

            # Create feature matrix for MI analysis
            feature_matrix = df_for_analysis[feature_cols].copy()

            # Handle NaN/inf values
            feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
            feature_matrix = feature_matrix.fillna(0)

            # Step 2: Mutual Information Feature Selection
            # -----------------------------------------

            debug_text.append("Computing mutual information with target variables...")

            # Compute MI with each target
            mi_scores = {}
            for target in valid_targets:
                target_values = df_for_analysis[target].values
                mi_scores[target] = mutual_info_regression(
                    feature_matrix, target_values, random_state=42
                )

            # Average MI scores across all targets
            avg_mi_scores = np.mean(
                [mi_scores[target] for target in valid_targets], axis=0
            )
            mi_scores_dict = dict(zip(feature_matrix.columns, avg_mi_scores))

            # Sort features by MI score (highest to lowest)
            sorted_features = sorted(
                mi_scores_dict, key=mi_scores_dict.get, reverse=True
            )

            debug_text.append("Top 5 features by mutual information:")
            for i, feature in enumerate(sorted_features[:5]):
                debug_text.append(f"{i+1}. {feature}: {mi_scores_dict[feature]:.4f}")

            # Compute pairwise MI between features (to remove redundancy)
            debug_text.append(
                "Computing pairwise mutual information to reduce redundancy..."
            )

            # Limit number of features to consider for pairwise MI to avoid excessive computation
            top_k_features = sorted_features[: min(100, len(sorted_features))]

            pairwise_mi = {}
            for f1, f2 in combinations(top_k_features, 2):
                pairwise_mi[(f1, f2)] = mutual_info_regression(
                    feature_matrix[[f1]], feature_matrix[f2].values, random_state=42
                )[0]

            # Select features with maximum MI & minimum redundancy
            debug_text.append(
                f"Selecting non-redundant features (threshold={redundancy_threshold})..."
            )

            selected_features = []
            for feature in sorted_features:
                if len(selected_features) >= int(max_features):
                    break

                if len(selected_features) == 0:
                    selected_features.append(feature)
                    continue

                # Check redundancy with already selected features
                redundant = False
                for sel_feature in selected_features:
                    if (feature, sel_feature) in pairwise_mi:
                        mi_value = pairwise_mi[(feature, sel_feature)]
                    else:
                        mi_value = pairwise_mi.get((sel_feature, feature), 0)

                    if mi_value > float(redundancy_threshold):
                        redundant = True
                        break

                if not redundant:
                    selected_features.append(feature)

            debug_text.append(
                f"Selected {len(selected_features)} non-redundant features out of {len(feature_cols)}"
            )

            # Create compressed feature matrix
            compressed_feature_matrix = feature_matrix[selected_features]

            # Step 3: Train autoencoder on selected features
            # -----------------------------------------
            debug_text.append(
                f"Training autoencoder with latent dimension {latent_dim}..."
            )

            # Normalize features
            feature_matrix_np = compressed_feature_matrix.to_numpy()
            num_features = feature_matrix_np.shape[1]

            feature_means = np.mean(feature_matrix_np, axis=0)
            feature_stds = np.std(feature_matrix_np, axis=0) + 1e-8
            normalized_features = (feature_matrix_np - feature_means) / feature_stds

            # Convert to PyTorch tensor
            feature_tensor = torch.tensor(normalized_features, dtype=torch.float32)

            # Create PyTorch dataset and dataloader
            dataset = TensorDataset(feature_tensor)
            dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

            # Set up device (CPU or GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            debug_text.append(f"Using device: {device}")

            # Initialize model
            model = DeepAutoencoder(
                input_dim=num_features, latent_dim=int(latent_dim)
            ).to(device)

            # Initialize optimizer and loss function
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))

            # Training loop
            num_epochs = int(epochs)
            losses = []

            debug_text.append(f"Starting training for {num_epochs} epochs...")

            for epoch in range(num_epochs):
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
            mi_features_store = {
                "latent_features": latent_df.to_json(date_format="iso", orient="split"),
                "selected_features": selected_features,
                "feature_cols": feature_cols,
                "latent_dim": latent_dim,
                "mi_scores": mi_scores_dict,
            }

            debug_text.append(
                f"Extracted {len(latent_df)} latent representations with dimension {latent_dim}"
            )

            # Create a placeholder figure until UMAP is run
            placeholder_fig = {
                "data": [],
                "layout": {
                    "title": 'MI feature selection and autoencoder training complete! Click "Run UMAP on MI Features" to '
                    "visualize",
                    "xaxis": {"title": "UMAP1"},
                    "yaxis": {"title": "UMAP2"},
                    "height": 600,
                },
            }

            return (
                placeholder_fig,
                "<br>".join(debug_text),
                mi_features_store,
                "MI feature selection and autoencoder training complete!",
                [],
            )

        except Exception as e:

            trace = traceback.format_exc()
            error_message = f"Error in MI feature selection: {str(e)}<br>{trace}"
            return {}, error_message, {}, "", []

    # If neither button was properly triggered, return empty states
    return {}, "Click 'Run MI Feature Selection' to start.", {}, "", []


@callback(
    Output("feature-importance-table", "data"),
    Input("feature-search-button", "n_clicks"),
    Input("feature-sort-option", "value"),
    State("feature-search-input", "value"),
    State("autoencoder-latent-store", "data"),
    prevent_initial_call=True,
)
def update_feature_importance_table(n_clicks, sort_option, search_term, latent_store):
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
                    "Correlation": f"{row['Correlation']:.4f}",  # Use absolute correlation value
                }
            )

        return table_data

    except Exception as e:
        print(f"Error updating feature importance table: {str(e)}")

        traceback.print_exc()
        return []
