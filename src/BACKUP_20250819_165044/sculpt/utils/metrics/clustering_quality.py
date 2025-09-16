import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def calculate_point_density(points, bandwidth=0.1):
    """Calculate density of each point using KDE"""
    # Make sure points is the right shape (2D array with shape [n_samples, n_dimensions])
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    # KDE expects each row to be a sample and each column to be a dimension
    # No need to transpose since our input is already in this format
    kde = gaussian_kde(points.T, bw_method=bandwidth)

    # Get density for each point
    densities = kde(points.T)
    return densities


def calculate_clustering_metrics(data, labels):
    """Calculate clustering quality metrics."""
    try:
        metrics = {}
        unique_labels = np.unique(labels)

        # Check if we have at least 2 clusters (required for metrics)
        if len(unique_labels) < 2:
            return {"note": "Need at least 2 clusters for metrics"}

        # For DBSCAN which might include noise points (-1)
        if -1 in unique_labels and len(unique_labels) > 2:
            # Calculate on non-noise points
            mask = labels != -1
            if np.sum(mask) > len(unique_labels) and len(np.unique(labels[mask])) > 1:
                metrics["silhouette"] = silhouette_score(data[mask], labels[mask])
                metrics["davies_bouldin"] = davies_bouldin_score(
                    data[mask], labels[mask]
                )
                metrics["calinski_harabasz"] = calinski_harabasz_score(
                    data[mask], labels[mask]
                )
                metrics["noise_ratio"] = 1 - (np.sum(mask) / len(labels))
                metrics["note"] = "Metrics calculated excluding noise points"
            else:
                return {"note": "Not enough valid points for metrics"}
        else:
            # Regular case - all points are assigned to clusters
            metrics["silhouette"] = silhouette_score(data, labels)
            metrics["davies_bouldin"] = davies_bouldin_score(data, labels)
            metrics["calinski_harabasz"] = calinski_harabasz_score(data, labels)
            metrics["note"] = "All points included in metrics"

        return metrics
    except Exception as e:
        return {"error": str(e)}


def hopkins_statistic(X, n_samples=None):
    """Compute Hopkins statistic to assess clustering tendency
    Values close to 1 indicate good clusterability, 0.5 suggests random data"""
    from sklearn.neighbors import NearestNeighbors

    if n_samples is None:
        n_samples = min(len(X) // 10, 30)  # Use at most 10% of data or 30 samples

    # Generate random points from the data space
    X_range = np.ptp(X, axis=0)
    X_min = np.min(X, axis=0)
    X_random = np.random.random((n_samples, X.shape[1])) * X_range + X_min

    # Fit nearest neighbors model
    nn = NearestNeighbors(n_neighbors=2).fit(X)

    # For real data points
    rnd_indices = np.random.choice(range(len(X)), size=n_samples, replace=False)
    real_points = X[rnd_indices]

    # Get distances to nearest neighbors
    u_distances, _ = nn.kneighbors(real_points, n_neighbors=2)
    u_distances = u_distances[:, 1]  # Exclude self

    # For random points
    w_distances, _ = nn.kneighbors(X_random, n_neighbors=1)
    w_distances = w_distances.ravel()

    # Calculate Hopkins statistic
    h = np.sum(w_distances) / (np.sum(u_distances) + np.sum(w_distances))

    return h


def cluster_stability(X, eps, min_samples, n_iterations=5, noise_level=0.05):
    """Assess DBSCAN stability by adding noise multiple times"""
    from sklearn.metrics import adjusted_rand_score

    # Get reference clustering
    reference_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    reference_labels = reference_dbscan.fit_predict(X)

    # Run multiple iterations with noise
    stability_scores = []
    for i in range(n_iterations):
        # Add small Gaussian noise
        noise = np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
        noisy_X = X + noise

        # Cluster the noisy data
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        noisy_labels = dbscan.fit_predict(noisy_X)

        # Compare with reference clustering
        if len(np.unique(noisy_labels)) > 1 and len(np.unique(reference_labels)) > 1:
            score = adjusted_rand_score(reference_labels, noisy_labels)
            stability_scores.append(score)

    # Return mean stability score (higher is better)
    return np.mean(stability_scores) if stability_scores else 0.0


def physics_cluster_consistency(df, cluster_labels):
    """Calculate consistency of physics parameters within clusters."""
    # Map cluster labels back to the full dataframe
    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = cluster_labels

    # Select key physics quantities
    physics_features = [
        "KER",
        "EESum",
        "EEsharing",
        "energy_ion1",
        "energy_ion2",
        "energy_electron1",
        "energy_electron2",
        "TotalEnergy",
    ]
    available_features = [f for f in physics_features if f in df_with_clusters.columns]

    if not available_features:
        return {"physics_consistency": 0.0, "note": "No physics features available"}

    # Calculate physics consistency metrics
    results = {}
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise

    if len(unique_clusters) < 2:
        return {"physics_consistency": 0.0, "note": "Need at least 2 clusters"}

    for feature in available_features:
        # Calculate variance ratio = between_cluster_variance / within_cluster_variance
        # Higher values indicate physics characteristics are more consistent inside clusters
        global_var = df_with_clusters[feature].var()
        if global_var == 0:
            continue

        within_vars = []
        cluster_means = []
        cluster_sizes = []

        for cluster in unique_clusters:
            cluster_data = df_with_clusters[df_with_clusters["cluster"] == cluster][
                feature
            ]
            if len(cluster_data) > 1:
                within_vars.append(cluster_data.var())
                cluster_means.append(cluster_data.mean())
                cluster_sizes.append(len(cluster_data))

        if within_vars and sum(cluster_sizes) > 0:
            # Calculate weighted average of within-cluster variance
            avg_within_var = np.average(within_vars, weights=cluster_sizes)

            # Calculate between-cluster variance (weighted by cluster size)
            global_mean = df_with_clusters[feature].mean()
            between_var = np.average(
                [(mean - global_mean) ** 2 for mean in cluster_means],
                weights=cluster_sizes,
            )

            # Calculate a normalized ratio that bounds between 0 and 1
            # Using the formula: between_var / (between_var + avg_within_var)
            # This is similar to the calculation in Calinski-Harabasz Index
            if between_var + avg_within_var > 0:
                variance_ratio = between_var / (between_var + avg_within_var)
                results[f"{feature}_consistency"] = variance_ratio
            else:
                results[f"{feature}_consistency"] = 0.0

    # Average across features
    if results:
        results["physics_consistency"] = np.mean(list(results.values()))
        # Track which features contributed most to consistency
        best_feature = max(
            results.items(), key=lambda x: x[1] if x[0] != "physics_consistency" else 0
        )
        if best_feature[0] != "physics_consistency":
            results["most_consistent_feature"] = best_feature[0].replace(
                "_consistency", ""
            )
    else:
        results["physics_consistency"] = 0.0
        results["note"] = "Could not calculate consistency for any features"

    return results
