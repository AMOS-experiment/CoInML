import dash
import numpy as np
import pandas as pd
import umap.umap_ as umap
from dash import Input, Output, State, callback, html
from dash.dependencies import ALL
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from sculpt.utils.file_handlers import extract_selection_indices
from sculpt.utils.metrics.clustering_quality import cluster_stability, hopkins_statistic


@callback(
    Output("genetic-features-graph", "figure"),
    Output("genetic-features-debug-output", "children"),
    Output("genetic-features-store", "data"),
    Output("run-umap-genetic-status", "children", allow_duplicate=True),
    Output("umap-quality-metrics-genetic", "children"),  # Add this output
    Input("run-genetic-features", "n_clicks"),
    Input("run-umap-genetic", "n_clicks"),
    State("genetic-data-source", "value"),
    State("clustering-method", "value"),
    State("dbscan-eps", "value"),
    State("dbscan-min-samples", "value"),
    State("kmeans-n-clusters", "value"),
    State("agglomerative-n-clusters", "value"),
    State("agglomerative-linkage", "value"),
    State("gp-generations", "value"),
    State("gp-population-size", "value"),
    State("gp-n-components", "value"),
    State("gp-functions", "value"),
    State({"type": "feature-selector-genetic", "category": ALL}, "value"),
    State("selected-points-store", "data"),
    State("selected-points-run-store", "data"),
    State("autoencoder-latent-store", "data"),
    State("combined-data-store", "data"),
    State("umap-graph", "figure"),
    State("genetic-features-store", "data"),
    State("metric-selector-genetic", "value"),  # Add this state
    prevent_initial_call=True,
)
def run_genetic_feature_discovery_and_umap(
    run_gp_clicks,
    run_umap_clicks,
    data_source,
    clustering_method,
    dbscan_eps,
    dbscan_min_samples,
    kmeans_n_clusters,
    agglo_n_clusters,
    agglo_linkage,
    gp_generations,
    gp_population_size,
    gp_n_components,
    gp_functions,
    selected_features_list,
    graph1_selection,
    graph3_selection,
    autoencoder_latent,
    combined_data_json,
    original_figure,
    genetic_features_store,
    selected_metrics,
):
    """Run genetic programming to discover features that explain clustering patterns,
    or run UMAP on previously discovered genetic features."""

    # Initialize debug info and check which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, "No action triggered.", {}, "", []

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    debug_text = []

    # If the UMAP button was clicked and we have stored genetic features, skip the GP part
    if (
        trigger_id == "run-umap-genetic"
        and genetic_features_store
        and "genetic_features" in genetic_features_store
    ):
        debug_text.append(
            "Running UMAP visualization on previously discovered genetic features..."
        )

        try:
            # Load the genetic features from the store
            gp_df = pd.read_json(
                genetic_features_store["genetic_features"], orient="split"
            )
            feature_names = genetic_features_store.get("feature_names", [])
            expressions = genetic_features_store.get("expressions", [])

            if gp_df.empty or not feature_names:
                return (
                    {},
                    "No genetic features found. Run Genetic Feature Discovery first.",
                    genetic_features_store,
                    "Error: No features to visualize",
                    [],
                )

            # Get user-selected features for UMAP
            all_selected_features = []
            for features in selected_features_list:
                if features:  # Only add non-empty lists
                    all_selected_features.extend(features)

            # Filter to only include genetic features (GP_Feature_X)
            gp_selected_features = [
                f for f in all_selected_features if f.startswith("GP_Feature_")
            ]

            if not gp_selected_features:
                # If no GP features selected, use all available GP features
                gp_selected_features = feature_names
                debug_text.append(
                    f"No genetic features specifically selected, using all {len(gp_selected_features)} available genetic"
                    " features"
                )
            else:
                debug_text.append(
                    f"Using {len(gp_selected_features)} selected genetic features for UMAP"
                )

            # Extract the genetic feature data
            X_gp = gp_df[gp_selected_features].to_numpy()

            # Handle NaN/inf values
            X_gp = np.nan_to_num(X_gp, nan=0.0, posinf=0.0, neginf=0.0)

            # Run UMAP on the selected genetic features
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                random_state=42,
            )

            umap_result = reducer.fit_transform(X_gp)

            # Create DataFrame for UMAP visualization
            umap_df = pd.DataFrame(
                {
                    "UMAP1": umap_result[:, 0],
                    "UMAP2": umap_result[:, 1],
                    "Cluster": gp_df["Cluster"].values,
                    "file_label": gp_df["file_label"].values,
                }
            )

            # Create visualization
            import plotly.graph_objects as go

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
                title=f"UMAP of Selected Genetic Features ({len(gp_selected_features)} features)",
                xaxis_title="UMAP1",
                yaxis_title="UMAP2",
                legend_title="Data File",
            )

            # Add information about which features were used
            debug_text.append("Features used for UMAP visualization:")
            for feature in gp_selected_features:
                idx = (
                    int(feature.split("_")[-1]) - 1
                )  # Extract feature number and adjust for 0-indexing
                if idx < len(expressions):
                    debug_text.append(f"{feature}: {expressions[idx]}")

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
                                                "• Davies-Bouldin Index: Measures average similarity between clusters (lower "
                                                "is better, range: 0 to ∞)"
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
                import traceback

                trace = traceback.format_exc()
                metrics_children = [html.Div(f"Error calculating metrics: {str(e)}")]

            return (
                fig,
                "<br>".join(debug_text),
                genetic_features_store,
                "UMAP visualization complete!",
                metrics_children,
            )

        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            error_message = (
                f"Error running UMAP on genetic features: {str(e)}<br>{trace}"
            )
            return {}, error_message, genetic_features_store, "Error running UMAP", []

    # If we're here, we're running the full genetic programming discovery
    if trigger_id == "run-genetic-features":
        debug_text.append(f"Data source: {data_source}")
        debug_text.append(f"Clustering method: {clustering_method}")

        # Define a custom exponential function with overflow protection
        def custom_exp(x):
            return np.where(x < 50, np.exp(x), np.exp(50))  # Prevents overflow

        exp_function = make_function(function=custom_exp, name="exp", arity=1)

        # Build function set based on user selection
        function_set = []
        if "basic" in gp_functions:
            function_set.extend(["add", "sub", "mul", "div"])
        if "trig" in gp_functions:
            function_set.extend(["sin", "cos", "tan"])
        if "exp_log" in gp_functions:
            function_set.extend(["log", exp_function])
        if "sqrt_pow" in gp_functions:
            function_set.extend(["sqrt"])
        if "special" in gp_functions:
            function_set.extend(["abs", "inv"])

        if not function_set:
            function_set = ["add", "sub", "mul", "div"]  # Default to basic functions

        debug_text.append(f"Function set: {', '.join([str(f) for f in function_set])}")
        debug_text.append(
            f"Generations: {gp_generations}, Population: {gp_population_size}, Features: {gp_n_components}"
        )

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

            # Collect all selected features for genetic programming
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

            elif data_source == "autoencoder-latent" and autoencoder_latent:
                # Use autoencoder latent space
                try:
                    latent_df = pd.read_json(
                        autoencoder_latent["latent_features"], orient="split"
                    )
                    # Keep only the latent features, not the labels
                    latent_cols = [
                        col for col in latent_df.columns if col.startswith("Latent_")
                    ]
                    df_for_analysis = latent_df[latent_cols].copy()
                    # We still need the labels for visualization
                    labels = latent_df["file_label"].copy()
                    debug_text.append(
                        f"Using autoencoder latent space with {len(df_for_analysis)} rows and {len(latent_cols)} dimensions"
                    )
                    # Override selected features with latent features
                    all_selected_features = latent_cols
                except Exception as e:
                    return (
                        {},
                        f"Error loading autoencoder latent space: {str(e)}",
                        {},
                        "",
                        [],
                    )

            else:
                # Default to all data
                df_for_analysis = combined_df.copy()
                labels = combined_df["file_label"].copy()
                debug_text.append(f"Defaulting to all {len(df_for_analysis)} rows")

            # Extract feature data
            feature_cols = [
                col for col in df_for_analysis.columns if col in all_selected_features
            ]

            if not feature_cols:
                return (
                    {},
                    "No valid features selected for genetic programming.",
                    {},
                    "",
                    [],
                )

            # Extract feature data
            X = df_for_analysis[feature_cols].to_numpy()

            # Handle NaN/inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Step 2: Apply clustering to get a rough idea of structure (optional for visualization)
            # -----------------------------------------

            # Standardize data before clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            debug_text.append(
                f"Standardized data for analysis, shape: {X_scaled.shape}"
            )

            # Apply the selected clustering method
            if clustering_method == "dbscan":
                debug_text.append(
                    f"Running DBSCAN with eps={dbscan_eps}, min_samples={dbscan_min_samples}"
                )
                clusterer = DBSCAN(
                    eps=float(dbscan_eps), min_samples=int(dbscan_min_samples)
                )
                cluster_labels = clusterer.fit_predict(X_scaled)

            elif clustering_method == "kmeans":
                debug_text.append(f"Running KMeans with n_clusters={kmeans_n_clusters}")
                clusterer = KMeans(n_clusters=int(kmeans_n_clusters), random_state=42)
                cluster_labels = clusterer.fit_predict(X_scaled)

            elif clustering_method == "agglomerative":
                debug_text.append(
                    f"Running Agglomerative clustering with n_clusters={agglo_n_clusters}, linkage={agglo_linkage}"
                )
                clusterer = AgglomerativeClustering(
                    n_clusters=int(agglo_n_clusters), linkage=agglo_linkage
                )
                cluster_labels = clusterer.fit_predict(X_scaled)

            # Handle case where all points are noise in DBSCAN
            if clustering_method == "dbscan" and np.all(cluster_labels == -1):
                debug_text.append(
                    "All points were labeled as noise. Using KMeans clustering as fallback."
                )
                clusterer = KMeans(n_clusters=3, random_state=42)
                cluster_labels = clusterer.fit_predict(X_scaled)

            # Count clusters and noise points
            unique_labels = np.unique(cluster_labels)
            num_clusters = len([label for label in unique_labels if label != -1])
            num_noise = np.sum(cluster_labels == -1) if -1 in unique_labels else 0

            debug_text.append(
                f"Found {num_clusters} clusters and {num_noise} noise points"
            )

            # Step 3: Run genetic programming with standard metric
            # -----------------------------------------
            debug_text.append(
                "Training genetic program to discover features with high information content..."
            )

            try:
                # Use a standard built-in metric instead of custom function
                # Options are 'mean absolute error', 'mse' (mean squared error),
                # 'rmse' (root mean squared error), 'pearson', 'spearman'

                # Run genetic programming with standard metric
                gp = SymbolicTransformer(
                    generations=int(gp_generations),
                    population_size=int(gp_population_size),
                    hall_of_fame=100,
                    n_components=int(gp_n_components),
                    function_set=function_set,
                    metric="spearman",  # Use Spearman correlation as metric
                    random_state=42,
                    parsimony_coefficient=0.05,  # Adds penalty for complexity
                    n_jobs=-1,
                )

                # For target, we'll use a combination of the clustering labels
                # and the first principal component to encourage diversity
                from sklearn.decomposition import PCA

                # Get principal components for diversity
                pca = PCA(n_components=1)
                pc1 = pca.fit_transform(X_scaled).flatten()

                # Combine with cluster labels to create a synthetic target
                # that encourages finding features that separate clusters
                # and capture variance
                synthetic_target = pc1 + cluster_labels * 10

                # Fit the model with this synthetic target
                gp.fit(X_scaled, synthetic_target)

                # Transform dataset with new features
                genetic_features = gp.transform(X_scaled)

                # Get the symbolic expressions
                expressions = []
                for sub_est in gp._best_programs:
                    expressions.append(str(sub_est))

                debug_text.append("Genetic programming complete")
                debug_text.append(f"Generated {genetic_features.shape[1]} features")

                # Show discovered expressions
                debug_text.append("Discovered expressions:")
                for i, expr in enumerate(expressions):
                    debug_text.append(f"Feature {i+1}: {expr}")

                # Create DataFrame with genetic features
                feature_names = [
                    f"GP_Feature_{i+1}" for i in range(genetic_features.shape[1])
                ]
                gp_df = pd.DataFrame(genetic_features, columns=feature_names)

                # Add cluster labels and original labels
                gp_df["Cluster"] = cluster_labels
                gp_df["file_label"] = labels.values

                # Add the generated features to the combined data
                combined_df_copy = combined_df.copy()
                for i, col in enumerate(feature_names):
                    # Add features to the combined dataframe
                    combined_df_copy[col] = np.nan  # Initialize with NaN

                    if data_source == "all":
                        # If using all data, we can add features directly
                        combined_df_copy[col] = genetic_features[:, i]
                    elif (
                        data_source == "graph1-selection"
                        and graph1_selection
                        and umap_coords is not None
                    ):
                        # For Graph 1 selection, add features to the selected rows
                        indices = extract_selection_indices(
                            graph1_selection, umap_coords
                        )
                        if indices:
                            for j, idx in enumerate(indices):
                                if idx < len(combined_df_copy):
                                    combined_df_copy.loc[idx, col] = genetic_features[
                                        j, i
                                    ]
                    else:
                        # For other selections, just note that features are partial
                        debug_text.append(
                            f"Note: Discovered feature {col} is only populated for the selected data"
                        )

                # Update the features store with the new features
                if "features_data_store" in combined_data_json:
                    features_data = combined_data_json["features_data_store"].copy()
                    if "column_names" in features_data:
                        if not any(
                            name in features_data["column_names"]
                            for name in feature_names
                        ):
                            features_data["column_names"].extend(feature_names)
                            debug_text.append(
                                f"Added {len(feature_names)} new features to the feature list"
                            )
                else:
                    # Create a new features data store if one doesn't exist
                    features_data = {
                        "column_names": list(combined_df.columns) + feature_names
                    }

                # Now create the placeholder figure and store
                placeholder_fig = {
                    "data": [],
                    "layout": {
                        "title": 'Genetic features discovered! Select features and click "Run UMAP on Genetic Features" to '
                        "visualize",
                        "xaxis": {"title": "UMAP1"},
                        "yaxis": {"title": "UMAP2"},
                        "height": 600,
                    },
                }

                # Store the genetic features but don't run UMAP yet
                new_genetic_features_store = {
                    "genetic_features": gp_df.to_json(
                        date_format="iso", orient="split"
                    ),
                    "expressions": expressions,
                    "feature_cols": feature_cols,
                    "feature_names": feature_names,
                }

                # Add information about the discovered features to the debug text
                debug_text.append("Discovered genetic features:")
                for i, expr in enumerate(expressions):
                    debug_text.append(f"{feature_names[i]}: {expr}")

                debug_text.append(
                    "Select features of interest and click 'Run UMAP on Genetic Features' to visualize."
                )

                return (
                    placeholder_fig,
                    "<br>".join(debug_text),
                    new_genetic_features_store,
                    "Genetic feature discovery complete!",
                    [],
                )

            except Exception as e:
                import traceback

                trace = traceback.format_exc()
                error_message = f"Error in genetic programming: {str(e)}<br>{trace}"
                debug_text.append(error_message)
                return (
                    {},
                    "<br>".join(debug_text),
                    {},
                    "Error occurred during genetic programming",
                    [],
                )

        except Exception as e:
            import traceback

            trace = traceback.format_exc()
            error_message = f"Error in genetic feature discovery: {str(e)}<br>{trace}"
            return {}, error_message, {}, "Error occurred", []

    # If neither button was properly triggered, return empty states
    return {}, "Click 'Run Genetic Feature Discovery' to start.", {}, "", []


@callback(
    Output("run-umap-genetic-status", "children"),
    Input("run-umap-genetic", "n_clicks"),
    Input("genetic-features-graph", "figure"),
    prevent_initial_call=True,
)
def update_genetic_umap_status(n_clicks, figure):
    """Update status for UMAP on genetic features."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "run-umap-genetic":
        return "Running UMAP on selected genetic features..."
    elif trigger_id == "genetic-features-graph":
        # This will be overwritten by the main callback
        return dash.no_update

    return dash.no_update
