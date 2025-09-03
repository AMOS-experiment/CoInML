import dash
import numpy as np
import pandas as pd
import umap.umap_ as umap
from dash import Input, Output, State, callback, html
from dash.dependencies import ALL
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer
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
        # Import required libraries including gplearn
        import plotly.graph_objects as go
        from gplearn.functions import make_function
        from gplearn.genetic import SymbolicTransformer
        from sklearn.cluster import KMeans  # Add this import here
        from sklearn.decomposition import PCA

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
                fig = go.Figure()

                unique_files = umap_df["file_label"].unique()
                colors = [
                    "blue",
                    "red",
                    "green",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                ]

                for i, file_label in enumerate(unique_files):
                    file_data = umap_df[umap_df["file_label"] == file_label]
                    fig.add_trace(
                        go.Scatter(
                            x=file_data["UMAP1"],
                            y=file_data["UMAP2"],
                            mode="markers",
                            name=f"File: {file_label}",
                            marker=dict(
                                color=colors[i % len(colors)],
                                size=6,
                                opacity=0.7,
                            ),
                            text=file_data["Cluster"],
                            hovertemplate="UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>Cluster: %{text}<br>File: "
                            + file_label
                            + "<extra></extra>",
                        )
                    )

                fig.update_layout(
                    title="UMAP Visualization of Genetic Features",
                    xaxis_title="UMAP1",
                    yaxis_title="UMAP2",
                    height=600,
                    showlegend=True,
                    hovermode="closest",
                )

                # Calculate comprehensive reliability assessment
                try:
                    from sculpt.utils.metrics.clustering_quality import (
                        cluster_stability,
                        hopkins_statistic,
                    )

                    # Collect all available metrics
                    all_metrics = {}
                    cluster_labels_for_metrics = gp_df["Cluster"].values
                    unique_clusters = len(np.unique(cluster_labels_for_metrics))

                    # Calculate clustering quality metrics
                    if unique_clusters > 1:
                        try:
                            all_metrics["silhouette"] = silhouette_score(
                                umap_result, cluster_labels_for_metrics
                            )
                        except:
                            pass

                        try:
                            all_metrics["davies_bouldin"] = davies_bouldin_score(
                                umap_result, cluster_labels_for_metrics
                            )
                        except:
                            pass

                        try:
                            all_metrics["calinski_harabasz"] = calinski_harabasz_score(
                                umap_result, cluster_labels_for_metrics
                            )
                        except:
                            pass

                    # Calculate clusterability metrics
                    try:
                        all_metrics["hopkins"] = hopkins_statistic(umap_result)
                    except:
                        pass

                    # Calculate stability if enough data
                    if len(umap_result) > 100:
                        try:
                            # Use a simple stability calculation
                            from sklearn.cluster import KMeans

                            kmeans = KMeans(
                                n_clusters=unique_clusters, random_state=42, n_init=10
                            )
                            kmeans_labels = kmeans.fit_predict(umap_result)
                            all_metrics["stability"] = cluster_stability(
                                umap_result, eps=0.1, min_samples=5, n_iterations=3
                            )
                        except:
                            pass

                    # Calculate comprehensive confidence score
                    confidence_result = calculate_adaptive_confidence_score(
                        all_metrics,
                        data_characteristics=None,
                        clustering_method="genetic_kmeans",
                    )

                    # Create UI using the proper confidence assessment
                    metrics_children = create_smart_confidence_ui(confidence_result)

                except Exception as metric_error:
                    debug_text.append(
                        f"Reliability assessment failed: {str(metric_error)}"
                    )
                    # Fallback to basic metrics
                    metrics_children = []
                    if unique_clusters > 1 and selected_metrics:
                        try:
                            if "silhouette" in selected_metrics:
                                sil_score = silhouette_score(
                                    umap_result, cluster_labels_for_metrics
                                )
                                metrics_children.append(
                                    html.P(f"Silhouette Score: {sil_score:.3f}")
                                )
                        except:
                            pass

                    if not metrics_children:
                        metrics_children = [html.P("Basic metrics calculation failed")]

                debug_text.append(f"UMAP completed on {X_gp.shape[1]} genetic features")
                debug_text.append(f"Visualizing {len(umap_df)} data points")

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

                # Add feature mapping for better understanding
                debug_text.append("Feature mapping (first 10):")
                for i in range(min(10, len(feature_cols))):
                    debug_text.append(f"  X{i} = {feature_cols[i]}")
                if len(feature_cols) > 10:
                    debug_text.append(
                        f"  ... and {len(feature_cols) - 10} more features"
                    )

                # Apply clustering with proper error handling
                if clustering_method == "dbscan":
                    eps_val = float(dbscan_eps) if dbscan_eps is not None else 0.3
                    min_samples_val = (
                        int(dbscan_min_samples) if dbscan_min_samples is not None else 5
                    )

                    clusterer = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                    cluster_labels = clusterer.fit_predict(X_scaled)
                    debug_text.append(
                        f"Applied DBSCAN clustering with eps={eps_val}, min_samples={min_samples_val}"
                    )
                elif clustering_method == "kmeans":
                    n_clusters_val = (
                        int(kmeans_n_clusters) if kmeans_n_clusters is not None else 3
                    )
                    clusterer = KMeans(
                        n_clusters=n_clusters_val, random_state=42, n_init=10
                    )
                    cluster_labels = clusterer.fit_predict(X_scaled)
                    debug_text.append(
                        f"Applied KMeans clustering with {n_clusters_val} clusters"
                    )
                elif clustering_method == "agglomerative":
                    n_clusters_val = (
                        int(agglo_n_clusters) if agglo_n_clusters is not None else 3
                    )
                    linkage_val = agglo_linkage if agglo_linkage is not None else "ward"
                    clusterer = AgglomerativeClustering(
                        n_clusters=n_clusters_val, linkage=linkage_val
                    )
                    cluster_labels = clusterer.fit_predict(X_scaled)
                    debug_text.append(
                        f"Applied Agglomerative clustering with {n_clusters_val} clusters, linkage={linkage_val}"
                    )
                else:
                    debug_text.append(
                        "Invalid clustering method specified. Using KMeans as fallback."
                    )
                    clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
                    cluster_labels = clusterer.fit_predict(X_scaled)

                unique_labels = np.unique(cluster_labels)
                num_clusters = len([label for label in unique_labels if label != -1])
                num_noise = np.sum(cluster_labels == -1) if -1 in unique_labels else 0
                debug_text.append(
                    f"Found {num_clusters} clusters and {num_noise} noise points"
                )

                # Handle case where clustering failed (all noise or too few clusters)
                if num_clusters < 2:
                    if clustering_method == "dbscan":
                        debug_text.append(
                            "DBSCAN found insufficient clusters. Trying with smaller eps..."
                        )

                        # Try progressively smaller eps values
                        for eps_test in [0.1, 0.05, 0.02]:
                            test_clusterer = DBSCAN(
                                eps=eps_test, min_samples=max(2, min_samples_val // 2)
                            )
                            test_labels = test_clusterer.fit_predict(X_scaled)
                            test_num_clusters = len(
                                [
                                    label
                                    for label in np.unique(test_labels)
                                    if label != -1
                                ]
                            )

                            if test_num_clusters >= 2:
                                cluster_labels = test_labels
                                debug_text.append(
                                    f"Success with eps={eps_test}: found {test_num_clusters} clusters"
                                )
                                break
                        else:
                            # If DBSCAN still fails, fall back to KMeans
                            debug_text.append(
                                "DBSCAN failed to find clusters. Falling back to KMeans."
                            )
                            clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
                            cluster_labels = clusterer.fit_predict(X_scaled)
                    else:
                        # For KMeans/Agglomerative, this shouldn't happen, but just in case
                        debug_text.append(
                            "Clustering produced insufficient groups. Using KMeans fallback."
                        )
                        clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
                        cluster_labels = clusterer.fit_predict(X_scaled)

                # Verify we have good clustering before proceeding
                final_unique_labels = np.unique(cluster_labels)
                final_num_clusters = len(
                    [label for label in final_unique_labels if label != -1]
                )
                final_num_noise = (
                    np.sum(cluster_labels == -1) if -1 in final_unique_labels else 0
                )

                debug_text.append(
                    f"Final clustering: {final_num_clusters} clusters, {final_num_noise} noise points"
                )

                # Check if clustering is suitable for genetic programming
                if final_num_clusters < 2:
                    return (
                        empty_fig,
                        "<br>".join(
                            debug_text
                            + [
                                "Error: Could not find meaningful clusters for genetic programming."
                            ]
                        ),
                        empty_store,
                        "Clustering failed",
                        empty_metrics,
                    )

                # Create synthetic target with improved stability
                pca = PCA(n_components=1)
                pc1 = pca.fit_transform(X_scaled).flatten()

                # Create a more robust synthetic target
                if final_num_noise > 0:
                    # If we have noise points, give them a neutral target value
                    synthetic_target = pc1.copy()
                    cluster_means = {}
                    for cluster_id in final_unique_labels:
                        if cluster_id != -1:  # Skip noise
                            mask = cluster_labels == cluster_id
                            cluster_means[cluster_id] = np.mean(pc1[mask])

                    # Assign cluster-specific offsets
                    for cluster_id, mean_val in cluster_means.items():
                        mask = cluster_labels == cluster_id
                        synthetic_target[mask] = (
                            pc1[mask] + cluster_id * 5
                        )  # Spread clusters apart
                else:
                    # No noise points, simpler approach
                    synthetic_target = pc1 + cluster_labels * 5

                debug_text.append(
                    f"Created synthetic target with shape: {synthetic_target.shape}"
                )
                debug_text.append(
                    f"Target range: [{synthetic_target.min():.3f}, {synthetic_target.max():.3f}]"
                )

                # Verify we have enough samples for genetic programming
                min_required_samples = max(10, int(gp_population_size or 1000) // 100)
                if len(X_scaled) < min_required_samples:
                    return (
                        empty_fig,
                        "<br>".join(
                            debug_text
                            + [
                                f"Error: Need at least {min_required_samples} samples for genetic programming, but only have {len(X_scaled)}."
                            ]
                        ),
                        empty_store,
                        "Insufficient data",
                        empty_metrics,
                    )

                # Create custom functions based on selected options
                function_set = []

                if not gp_functions:
                    gp_functions = ["basic"]  # Default fallback

                if "basic" in gp_functions:
                    function_set.extend(["add", "sub", "mul", "div"])

                if "trig" in gp_functions:
                    # Add safe trigonometric functions
                    def safe_sin(x):
                        return np.sin(np.clip(x, -10, 10))

                    def safe_cos(x):
                        return np.cos(np.clip(x, -10, 10))

                    sin_func = make_function(function=safe_sin, name="sin", arity=1)
                    cos_func = make_function(function=safe_cos, name="cos", arity=1)
                    function_set.extend([sin_func, cos_func])

                if "exp_log" in gp_functions:
                    # Add safe exponential and log functions
                    def safe_exp(x):
                        return np.exp(np.clip(x, -10, 10))

                    def safe_log(x):
                        return np.log(np.abs(x) + 1e-6)

                    exp_func = make_function(function=safe_exp, name="exp", arity=1)
                    log_func = make_function(function=safe_log, name="log", arity=1)
                    function_set.extend([exp_func, log_func])

                if "sqrt_pow" in gp_functions:
                    # Add safe sqrt and power functions
                    def safe_sqrt(x):
                        return np.sqrt(np.abs(x))

                    def safe_pow(x, y):
                        return np.power(np.abs(x), np.clip(y, -3, 3))

                    sqrt_func = make_function(function=safe_sqrt, name="sqrt", arity=1)
                    pow_func = make_function(function=safe_pow, name="pow", arity=2)
                    function_set.extend([sqrt_func, pow_func])

                if "special" in gp_functions:
                    # Add special functions
                    def safe_abs(x):
                        return np.abs(x)

                    def safe_inv(x):
                        return 1.0 / (x + 1e-6)

                    abs_func = make_function(function=safe_abs, name="abs", arity=1)
                    inv_func = make_function(function=safe_inv, name="inv", arity=1)
                    function_set.extend([abs_func, inv_func])

                debug_text.append(f"Using {len(function_set)} mathematical functions")

                # Use SymbolicTransformer with minimal, safe parameters
                generations_val = max(
                    5,
                    min(int(gp_generations) if gp_generations is not None else 10, 50),
                )
                population_val = max(
                    50,
                    min(
                        (
                            int(gp_population_size)
                            if gp_population_size is not None
                            else 200
                        ),
                        1000,
                    ),
                )
                components_val = max(
                    1,
                    min(int(gp_n_components) if gp_n_components is not None else 5, 10),
                )

                debug_text.append(
                    f"GP Parameters: gen={generations_val}, pop={population_val}, comp={components_val}"
                )

                # Try with minimal parameters first
                try:
                    gp = SymbolicTransformer(
                        generations=generations_val,
                        population_size=population_val,
                        n_components=components_val,
                        function_set=function_set,
                        random_state=42,
                        n_jobs=1,
                        verbose=0,
                    )
                except Exception as e:
                    debug_text.append(
                        f"Failed to create SymbolicTransformer with custom functions: {str(e)}"
                    )
                    # Fallback to basic functions only
                    gp = SymbolicTransformer(
                        generations=generations_val,
                        population_size=population_val,
                        n_components=components_val,
                        function_set=[
                            "add",
                            "sub",
                            "mul",
                            "div",
                        ],  # Basic functions only
                        random_state=42,
                        n_jobs=1,
                        verbose=0,
                    )

                debug_text.append(
                    f"Starting genetic programming with {generations_val} generations..."
                )

                # Fit the genetic programming model
                genetic_features = gp.fit_transform(X_scaled, synthetic_target)

                # Extract expressions from the final generation
                expressions = []
                if hasattr(gp, "_programs") and gp._programs:
                    final_programs = gp._programs[-1]  # Get final generation
                    expressions = [str(program) for program in final_programs]
                else:
                    # Fallback if expressions can't be extracted
                    expressions = [
                        f"GP_Expression_{i+1}" for i in range(genetic_features.shape[1])
                    ]

                debug_text.append(
                    f"Generated {genetic_features.shape[1]} genetic features using {len(function_set)} functions"
                )

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
                for i, feat_name in enumerate(feature_names):
                    if i < len(expressions):
                        debug_text.append(f"{feat_name}: {expressions[i]}")
                    else:
                        debug_text.append(f"{feat_name}: Expression not available")

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
                debug_text.append("Full traceback:")
                debug_text.append(trace)
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
