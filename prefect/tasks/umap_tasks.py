"""
Fixed Prefect tasks for UMAP computations in SCULPT
Addresses data structure issues
"""

import time
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from prefect import get_run_logger, task


@task(
    name="load_and_prepare_data",
    description="Load files and prepare combined dataframe",
    retries=1,
    retry_delay_seconds=10,
)
def load_and_prepare_data(
    stored_files: List[Dict], selected_ids: List[int], sample_frac: float
) -> Tuple[pd.DataFrame, List[str]]:
    """Load selected files and prepare combined dataframe"""
    logger = get_run_logger()
    debug_messages = []

    logger.info(f"Loading {len(selected_ids)} files with sample fraction {sample_frac}")
    debug_messages.append(f"Loading {len(selected_ids)} files...")

    # Convert selected_ids to integers if they're strings
    selected_ids_int = []
    for sid in selected_ids:
        try:
            selected_ids_int.append(int(sid))
        except (ValueError, TypeError):
            selected_ids_int.append(sid)

    dfs = []
    for file_dict in stored_files:
        if file_dict["id"] in selected_ids_int:
            try:
                # Handle different data formats
                if isinstance(file_dict["data"], str):
                    # JSON string
                    df = pd.read_json(file_dict["data"], orient="split")
                elif isinstance(file_dict["data"], dict):
                    # Dictionary with 'data' key or direct DataFrame dict
                    if "data" in file_dict["data"]:
                        df = pd.DataFrame(file_dict["data"]["data"])
                    else:
                        df = pd.DataFrame(file_dict["data"])
                else:
                    # Already a DataFrame or list
                    df = pd.DataFrame(file_dict["data"])

                if sample_frac < 1.0:
                    df = df.sample(frac=sample_frac, random_state=42)
                    logger.info(
                        f"Sampled {len(df)} points from {file_dict['filename']}"
                    )

                # Add file metadata
                df["file_label"] = file_dict["filename"]
                df["file_id"] = file_dict["id"]

                dfs.append(df)
                debug_messages.append(
                    f"Loaded {file_dict['filename']}: {len(df)} events"
                )

            except Exception as e:
                logger.warning(
                    f"Error loading file {file_dict.get('filename', 'unknown')}: {str(e)}"
                )
                # Try alternative parsing
                try:
                    # Sometimes the data is double-encoded
                    import json

                    if isinstance(file_dict["data"], str):
                        data_dict = json.loads(file_dict["data"])
                        df = pd.DataFrame(
                            data_dict["data"] if "data" in data_dict else data_dict
                        )

                        if sample_frac < 1.0:
                            df = df.sample(frac=sample_frac, random_state=42)

                        df["file_label"] = file_dict["filename"]
                        df["file_id"] = file_dict["id"]
                        dfs.append(df)
                        logger.info(
                            f"Successfully loaded {file_dict['filename']} with alternative parsing"
                        )
                except Exception as e2:
                    logger.error(
                        f"Failed to load {file_dict.get('filename', 'unknown')}: {str(e2)}"
                    )
                    debug_messages.append(
                        f"Failed to load {file_dict.get('filename', 'unknown')}"
                    )

    if not dfs:
        raise ValueError("No valid files found for selected IDs")

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataframe: {len(combined_df)} total events")
    debug_messages.append(f"Total combined events: {len(combined_df)}")

    return combined_df, debug_messages


@task(
    name="compute_umap_embedding",
    description="Compute UMAP embedding on feature data",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=(
        lambda context, parameters, **kwargs: f"umap_{len(parameters['feature_data'])}_{parameters['num_neighbors']}"
        f"_{parameters['min_dist']}_{hash(str(parameters['feature_cols']))}"
    ),
    cache_expiration=timedelta(hours=2),
)
def compute_umap_embedding(
    feature_data: np.ndarray,
    feature_cols: List[str],
    num_neighbors: int,
    min_dist: float,
) -> Tuple[np.ndarray, float]:
    """Compute UMAP embedding"""
    logger = get_run_logger()

    logger.info(
        f"Starting UMAP: {len(feature_data)} samples, {len(feature_cols)} features"
    )
    logger.info(f"Parameters: n_neighbors={num_neighbors}, min_dist={min_dist}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_data)

    logger.info("Features scaled, initializing UMAP reducer...")

    # Initialize UMAP
    reducer = umap.UMAP(
        n_neighbors=num_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
        verbose=False,
    )

    start_time = time.time()
    logger.info("Computing UMAP embedding... (this may take a while)")

    embedding = reducer.fit_transform(X_scaled)

    computation_time = time.time() - start_time
    logger.info(f"UMAP completed in {computation_time:.2f} seconds")

    return embedding, computation_time


@task(
    name="compute_clustering",
    description="Compute DBSCAN clustering on UMAP embedding",
    retries=1,
    retry_delay_seconds=10,
)
def compute_clustering(
    umap_embedding: np.ndarray, eps: float = 0.5, min_samples: int = 5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute DBSCAN clustering on UMAP coordinates"""
    logger = get_run_logger()

    logger.info(f"Computing DBSCAN clustering: eps={eps}, min_samples={min_samples}")

    # Scale UMAP coordinates
    scaler = StandardScaler()
    X_umap_scaled = scaler.fit_transform(umap_embedding)

    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_umap_scaled)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    logger.info(f"Found {n_clusters} clusters and {n_noise} noise points")

    return cluster_labels, {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "eps": eps,
        "min_samples": min_samples,
    }


@task(name="create_umap_dataframe", description="Create final UMAP dataframe")
def create_umap_dataframe(
    combined_df: pd.DataFrame,
    umap_embedding: np.ndarray,
    cluster_labels: np.ndarray,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Create the final UMAP dataframe"""
    logger = get_run_logger()

    logger.info("Creating final UMAP dataframe...")

    umap_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])

    # Use 'Cluster' with capital C to match expected format
    umap_df["Cluster"] = cluster_labels

    # Add metadata columns
    if "file_label" in combined_df.columns:
        umap_df["file_label"] = combined_df["file_label"].values
    if "file_id" in combined_df.columns:
        umap_df["file_id"] = combined_df["file_id"].values

    # Add original features
    for col in feature_cols:
        if col in combined_df.columns:
            umap_df[col] = combined_df[col].values

    logger.info(f"Final UMAP dataframe: {umap_df.shape}")

    return umap_df
