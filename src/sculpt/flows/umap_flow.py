"""
Fixed Prefect flow for UMAP analysis in SCULPT
File: sculpt/flows/umap_flow.py
"""
import os
import prefect
from prefect import task, flow, get_run_logger

# Ensure Prefect uses the correct API
if os.getenv('DOCKER_CONTAINER') == 'true':
    os.environ['PREFECT_API_URL'] = 'http://prefect-server:4200/api'
    # Force Prefect to use the server API, not local
    prefect.settings.PREFECT_API_URL.value = lambda: 'http://prefect-server:4200/api'

from prefect import flow, get_run_logger
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from sculpt.tasks.umap_tasks import (
    load_and_prepare_data,
    compute_umap_embedding,
    compute_clustering,
    create_umap_dataframe
)


@flow(
    name="SCULPT UMAP Analysis",
    description="Complete UMAP embedding and clustering workflow",
    retries=1,
    log_prints=True,  # ADD THIS - ensures logs go to cloud
    persist_result=True,  # ADD THIS - ensures results are stored
    timeout_seconds=600  # ADD THIS - 10 minute timeout
)
def umap_analysis_flow(
    stored_files: List[Dict],  # Changed from Dict to List[Dict]
    selected_ids: List[int],   # File IDs are integers
    num_neighbors: int,
    min_dist: float,
    sample_frac: float,
    selected_features_list: List[List[str]],
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5
) -> Dict[str, Any]:
    """Complete UMAP analysis workflow for SCULPT"""
    logger = get_run_logger()
    
    # Force a print to ensure cloud registration
    print(f"UMAP Analysis starting in Prefect Cloud...")
    
    logger.info("="*60)
    logger.info("Starting SCULPT UMAP Analysis Flow")
    logger.info("="*60)
    
    # Flatten selected features
    all_selected_features = []
    for features in selected_features_list:
        if features:
            all_selected_features.extend(features)
    
    logger.info(f"Parameters: n_neighbors={num_neighbors}, min_dist={min_dist}")
    logger.info(f"Selected features: {len(all_selected_features)}")
    
    # Step 1: Load data
    logger.info("Step 1/4: Loading data...")
    combined_df, debug_messages = load_and_prepare_data(
        stored_files=stored_files,
        selected_ids=selected_ids,
        sample_frac=sample_frac
    )
    
    # Determine feature columns - look for momentum features or other numeric columns
    if not all_selected_features:
        # Default: use momentum features (Px, Py, Pz columns)
        momentum_cols = [col for col in combined_df.columns 
                        if any(p in col for p in ['Px_', 'Py_', 'Pz_'])]
        
        # Also include physics features if they exist
        physics_cols = [col for col in combined_df.columns 
                       if col.startswith('particle_')]
        
        feature_cols = list(set(momentum_cols + physics_cols))
        
        if not feature_cols:
            # Fallback: use all numeric columns except metadata
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['file_id', 'file_label', 'UMAP1', 'UMAP2', 'Cluster', 'cluster']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} default features")
    else:
        feature_cols = [col for col in combined_df.columns if col in all_selected_features]
        logger.info(f"Using {len(feature_cols)} selected features")
    
    if not feature_cols:
        raise ValueError("No valid feature columns found in the data")
    
    # Extract feature data
    feature_data = combined_df[feature_cols].values
    
    # Step 2: Compute UMAP
    logger.info("Step 2/4: Computing UMAP embedding...")
    umap_embedding, computation_time = compute_umap_embedding(
        feature_data=feature_data,
        feature_cols=feature_cols,
        num_neighbors=num_neighbors,
        min_dist=min_dist
    )
    
    # Step 3: Clustering
    logger.info("Step 3/4: Computing DBSCAN clustering...")
    cluster_labels, clustering_info = compute_clustering(
        umap_embedding=umap_embedding,
        eps=dbscan_eps,
        min_samples=dbscan_min_samples
    )
    
    # Step 4: Create final dataframe
    logger.info("Step 4/4: Creating final UMAP dataframe...")
    umap_df = create_umap_dataframe(
        combined_df=combined_df,
        umap_embedding=umap_embedding,
        cluster_labels=cluster_labels,
        feature_cols=feature_cols
    )
    
    logger.info("="*60)
    logger.info("SCULPT UMAP Analysis Flow COMPLETE!")
    logger.info(f"Total samples: {len(umap_df)}")
    logger.info(f"Clusters found: {clustering_info['n_clusters']}")
    logger.info(f"UMAP computation time: {computation_time:.2f}s")
    logger.info("="*60)
    
    return {
        'success': True,
        'combined_df': combined_df,
        'umap_df': umap_df,
        'cluster_labels': cluster_labels.tolist(),
        'feature_cols': feature_cols,
        'clustering_info': clustering_info,
        'debug_messages': debug_messages,
        'metadata': {
            'n_samples': len(combined_df),
            'n_features': len(feature_cols),
            'n_clusters': clustering_info['n_clusters'],
            'computation_time': computation_time
        }
    }


@flow(name="SCULPT UMAP Test")
def test_umap_flow(num_neighbors: int = 15, min_dist: float = 0.1):
    """Simple test flow for verification"""
    logger = get_run_logger()
    
    logger.info("Testing SCULPT UMAP Flow")
    
    # Create test data
    test_data = np.random.randn(1000, 10)
    feature_cols = [f"feature_{i}" for i in range(10)]
    
    embedding, computation_time = compute_umap_embedding(
        feature_data=test_data,
        feature_cols=feature_cols,
        num_neighbors=num_neighbors,
        min_dist=min_dist
    )
    
    logger.info(f"Test Complete! Computation time: {computation_time:.2f}s")
    
    return {
        'success': True,
        'embedding_shape': embedding.shape,
        'computation_time': computation_time
    }