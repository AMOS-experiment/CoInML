import dash
import numpy as np
import pandas as pd
import umap.umap_ as umap
from dash import Input, Output, State, callback, html
from dash.dependencies import ALL

# from gplearn.functions import make_function
# from gplearn.genetic import SymbolicTransformer
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# from sculpt.utils.file_handlers import extract_selection_indices
from sculpt.utils.metrics.clustering_quality import cluster_stability, hopkins_statistic
from sculpt.utils.metrics.confidence_assessment import (
    calculate_adaptive_confidence_score,
)
from sculpt.utils.ui import create_smart_confidence_ui


@callback(
    Output("genetic-features-graph", "figure"),
    Output("genetic-features-debug-output", "children"),
    Output("genetic-features-store", "data"),
    Output("run-umap-genetic-status", "children", allow_duplicate=True),
    Output("umap-quality-metrics-genetic", "children"),
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
    State("metric-selector-genetic", "value"),
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

    try:

        # Initialize debug info and check which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return {}, "No action triggered.", {}, "", []

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        debug_text = []

        # Default return values
        empty_fig = {
            "data": [],
            "layout": {
                "title": 'Click "Run Genetic Feature Discovery" to start',
                "xaxis": {"title": "UMAP1"},
                "yaxis": {"title": "UMAP2"},
                "height": 600,
            },
        }
        empty_store = {}
        empty_metrics = []

        # If the UMAP button was clicked and we have stored genetic features
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
                        empty_fig,
                        "No genetic features found. Run Genetic Feature Discovery first.",
                        genetic_features_store,
                        "Error: No features to visualize",
                        empty_metrics,
                    )

                # Get user-selected features for UMAP
                all_selected_features = []
                for features in selected_features_list:
                    if features:
                        all_selected_features.extend(features)

                # Filter to only include genetic features (GP_Feature_X)
                gp_selected_features = [
                    f for f in all_selected_features if f.startswith("GP_Feature_")
                ]

                if not gp_selected_features:
                    # Use all available GP features
                    gp_selected_features = feature_names
                    debug_text.append(
                        f"No genetic features selected, using all {len(gp_selected_features)} available features"
                    )
                else:
                    debug_text.append(
                        f"Using {len(gp_selected_features)} selected genetic features"
                    )

                # Extract the genetic feature data
                X_gp = gp_df[gp_selected_features].to_numpy()
                X_gp = np.nan_to_num(X_gp, nan=0.0, posinf=0.0, neginf=0.0)

                # Run UMAP
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
                            clean_name = trace["name"]
                            if " (" in clean_name:
                                clean_name = clean_name.split(" (")[0]
                            color_map[clean_name] = trace["marker"]["color"]

                # Add traces for each file label
                for label in umap_df["file_label"].unique():
                    mask = umap_df["file_label"] == label
                    df_subset = umap_df[mask]

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

                # Calculate clustering metrics if requested
                metrics_children = []
                if selected_metrics:
                    try:
                        X_umap = umap_df[["UMAP1", "UMAP2"]].to_numpy()
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
                            noise_ratio = (
                                noise_count / len(labels) if len(labels) > 0 else 0
                            )

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

                        metrics = {}
                        unique_clusters = set(cluster_labels)
                        n_clusters = len(unique_clusters) - (
                            1 if -1 in unique_clusters else 0
                        )
                        n_noise = np.sum(cluster_labels == -1)
                        noise_ratio = (
                            n_noise / len(cluster_labels)
                            if len(cluster_labels) > 0
                            else 0
                        )

                        if n_clusters >= 2:
                            mask = cluster_labels != -1
                            non_noise_points = np.sum(mask)
                            non_noise_clusters = len(set(cluster_labels[mask]))

                            if (
                                non_noise_points > non_noise_clusters
                                and non_noise_clusters > 1
                            ):
                                if "silhouette" in selected_metrics:
                                    metrics["silhouette"] = silhouette_score(
                                        X_umap_scaled[mask], cluster_labels[mask]
                                    )
                                if "davies_bouldin" in selected_metrics:
                                    metrics["davies_bouldin"] = davies_bouldin_score(
                                        X_umap_scaled[mask], cluster_labels[mask]
                                    )
                                if "calinski_harabasz" in selected_metrics:
                                    metrics["calinski_harabasz"] = (
                                        calinski_harabasz_score(
                                            X_umap_scaled[mask], cluster_labels[mask]
                                        )
                                    )
                                if "hopkins" in selected_metrics:
                                    metrics["hopkins"] = hopkins_statistic(
                                        X_umap_scaled
                                    )
                                if "stability" in selected_metrics:
                                    metrics["stability"] = cluster_stability(
                                        X_umap_scaled, best_eps, 5, n_iterations=3
                                    )

                        # Create metrics UI
                        if metrics:
                            confidence_data = calculate_adaptive_confidence_score(
                                metrics, clustering_method="dbscan"
                            )
                            metrics_children = [
                                create_smart_confidence_ui(confidence_data)
                            ]

                    except Exception as e:
                        metrics_children = [
                            html.Div(f"Error calculating metrics: {str(e)}")
                        ]

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
                error_message = f"Error running UMAP on genetic features: {str(e)}"
                debug_text.append(error_message)
                return (
                    empty_fig,
                    "<br>".join(debug_text),
                    genetic_features_store,
                    "Error running UMAP",
                    empty_metrics,
                )

        # If we're running genetic feature discovery
        elif trigger_id == "run-genetic-features":
            debug_text.append(f"Data source: {data_source}")
            debug_text.append(f"Clustering method: {clustering_method}")

            try:
                # Load combined dataframe
                if (
                    not combined_data_json
                    or "combined_df" not in combined_data_json
                    or combined_data_json["combined_df"] == "{}"
                ):
                    return (
                        empty_fig,
                        "No combined dataset available. Please run UMAP first.",
                        empty_store,
                        "",
                        empty_metrics,
                    )

                combined_df = pd.read_json(
                    combined_data_json["combined_df"], orient="split"
                )
                debug_text.append(
                    f"Loaded combined dataset with {len(combined_df)} rows"
                )

                # TODO: Is this needed?
                # Load UMAP coordinates if needed
                # umap_coords = None
                # if (
                #     "umap_coords" in combined_data_json
                #     and combined_data_json["umap_coords"] != "{}"
                # ):
                #     umap_coords = pd.read_json(
                #         combined_data_json["umap_coords"], orient="split"
                #     )

                # Collect selected features
                all_selected_features = []
                for features in selected_features_list:
                    if features:
                        all_selected_features.extend(features)

                if not all_selected_features:
                    all_selected_features = [
                        col
                        for col in combined_df.columns
                        if col.startswith("particle_")
                    ]
                    debug_text.append(
                        f"No features selected, using {len(all_selected_features)} default momentum features"
                    )
                else:
                    debug_text.append(
                        f"Using {len(all_selected_features)} selected features"
                    )

                # Prepare data based on source
                df_for_analysis = combined_df.copy()
                labels = combined_df["file_label"].copy()
                debug_text.append(f"Using all {len(df_for_analysis)} rows for analysis")

                # Extract feature data
                feature_cols = [
                    col
                    for col in df_for_analysis.columns
                    if col in all_selected_features
                ]
                if not feature_cols:
                    return (
                        empty_fig,
                        "No valid features selected for genetic programming.",
                        empty_store,
                        "",
                        empty_metrics,
                    )

                X = df_for_analysis[feature_cols].to_numpy()
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                # Standardize data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                debug_text.append(
                    f"Standardized data for analysis, shape: {X_scaled.shape}"
                )

                # Apply clustering
                if clustering_method == "dbscan":
                    debug_text.append(
                        f"Running DBSCAN with eps={dbscan_eps}, min_samples={dbscan_min_samples}"
                    )
                    clusterer = DBSCAN(
                        eps=float(dbscan_eps), min_samples=int(dbscan_min_samples)
                    )
                    cluster_labels = clusterer.fit_predict(X_scaled)
                elif clustering_method == "kmeans":
                    debug_text.append(
                        f"Running KMeans with n_clusters={kmeans_n_clusters}"
                    )
                    clusterer = KMeans(
                        n_clusters=int(kmeans_n_clusters), random_state=42, n_init=10
                    )
                    cluster_labels = clusterer.fit_predict(X_scaled)
                elif clustering_method == "agglomerative":
                    debug_text.append("Running Agglomerative clustering")
                    clusterer = AgglomerativeClustering(
                        n_clusters=int(agglo_n_clusters), linkage=agglo_linkage
                    )
                    cluster_labels = clusterer.fit_predict(X_scaled)

                # Handle all noise case
                if clustering_method == "dbscan" and np.all(cluster_labels == -1):
                    debug_text.append(
                        "All points were labeled as noise. Using KMeans as fallback."
                    )
                    clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
                    cluster_labels = clusterer.fit_predict(X_scaled)

                unique_labels = np.unique(cluster_labels)
                num_clusters = len([label for label in unique_labels if label != -1])
                num_noise = np.sum(cluster_labels == -1) if -1 in unique_labels else 0
                debug_text.append(
                    f"Found {num_clusters} clusters and {num_noise} noise points"
                )

                # Run feature engineering (fallback approach)
                debug_text.append("Creating engineered features...")

                try:
                    from sklearn.decomposition import PCA
                    from sklearn.feature_selection import (
                        VarianceThreshold,
                        mutual_info_regression,
                    )
                    from sklearn.preprocessing import PolynomialFeatures

                    # Create polynomial features
                    poly = PolynomialFeatures(
                        degree=2, include_bias=False, interaction_only=True
                    )
                    poly_features = poly.fit_transform(X_scaled)

                    # Limit features
                    max_features = min(int(gp_n_components), poly_features.shape[1])

                    # Use variance threshold
                    variance_selector = VarianceThreshold(threshold=0.01)
                    variance_features = variance_selector.fit_transform(poly_features)

                    # Create synthetic target
                    pca = PCA(n_components=1)
                    pc1 = pca.fit_transform(X_scaled).flatten()
                    synthetic_target = pc1 + cluster_labels * 10

                    # Select by mutual information
                    mi_scores = mutual_info_regression(
                        variance_features, synthetic_target, random_state=42
                    )
                    top_indices = np.argsort(mi_scores)[-max_features:]
                    genetic_features = variance_features[:, top_indices]

                    # Create feature names
                    feature_names_input = [f"X{i}" for i in range(X_scaled.shape[1])]
                    poly_feature_names = poly.get_feature_names_out(feature_names_input)
                    variance_feature_names = [
                        poly_feature_names[i]
                        for i in range(len(poly_feature_names))
                        if variance_selector.get_support()[i]
                    ]
                    expressions = [variance_feature_names[i] for i in top_indices]

                    debug_text.append(
                        f"Created {genetic_features.shape[1]} engineered features"
                    )

                except Exception as e:
                    debug_text.append(f"Feature engineering failed: {str(e)}")
                    # Ultimate fallback: use original features
                    genetic_features = X_scaled[:, : int(gp_n_components)]
                    expressions = [
                        f"Original_Feature_{i+1}"
                        for i in range(genetic_features.shape[1])
                    ]

                # Create results
                feature_names = [
                    f"GP_Feature_{i+1}" for i in range(genetic_features.shape[1])
                ]
                gp_df = pd.DataFrame(genetic_features, columns=feature_names)
                gp_df["Cluster"] = cluster_labels
                gp_df["file_label"] = labels.values

                # Create placeholder figure
                placeholder_fig = {
                    "data": [],
                    "layout": {
                        "title": 'Genetic features discovered! Select features and click "Run UMAP on Genetic Features" '
                        "to visualize",
                        "xaxis": {"title": "UMAP1"},
                        "yaxis": {"title": "UMAP2"},
                        "height": 600,
                    },
                }

                # Store results
                new_genetic_features_store = {
                    "genetic_features": gp_df.to_json(
                        date_format="iso", orient="split"
                    ),
                    "expressions": expressions,
                    "feature_cols": feature_cols,
                    "feature_names": feature_names,
                }

                debug_text.append("Discovered genetic features:")
                for i, expr in enumerate(expressions):
                    debug_text.append(f"{feature_names[i]}: {expr}")

                return (
                    placeholder_fig,
                    "<br>".join(debug_text),
                    new_genetic_features_store,
                    "Genetic feature discovery complete!",
                    empty_metrics,
                )

            except Exception as e:
                import traceback

                trace = traceback.format_exc()
                error_message = f"Error in genetic feature discovery: {str(e)}"
                debug_text.append(error_message)
                return (
                    empty_fig,
                    "<br>".join(debug_text),
                    empty_store,
                    "Error occurred",
                    empty_metrics,
                )

        # Default case
        return (
            empty_fig,
            "Click 'Run Genetic Feature Discovery' to start.",
            empty_store,
            "",
            empty_metrics,
        )

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        error_message = f"Callback error: {str(e)}"
        print(f"Genetic callback error: {error_message}\n{trace}")

        # Return safe defaults
        return (
            {"data": [], "layout": {"title": "Error occurred", "height": 600}},
            error_message,
            {},
            "Error occurred",
            [],
        )


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
