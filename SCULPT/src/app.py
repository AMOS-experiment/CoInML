#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))


# In[5]:


import base64
import io
import json
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import umap.umap_ as umap
from dash.dependencies import ALL
from matplotlib.path import Path  # Add this import for lasso selection
# Add imports for the autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function
from sklearn.metrics import silhouette_score
from gplearn.fitness import make_fitness
from dash import dash_table
import re
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from datetime import datetime


# Define the required columns as they appear in the CSV.
required_columns = [
    'Px_ion1', 'Py_ion1', 'Pz_ion1',
    'Px_ion2', 'Py_ion2', 'Pz_ion2',
    'Px_neutral', 'Py_neutral', 'Pz_neutral',
    'Px_electron1', 'Py_electron1', 'Pz_electron1',
    'Px_electron2', 'Py_electron2', 'Pz_electron2'
]

# Constants for physics calculations
mass_ion = 2 * 1836  # Deuterium ion (D+)
mass_neutral = 16 * 1836  # Neutral Oxygen atom
mass_electron = 1  # Electron mass

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Add this helper function before the update_files callback
def store_file_data(file_id, data):
    """Store file data in a temporary file and return the path."""
    import tempfile
    import os
    
    # Create a temporary directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), 'dash_coltrims_data')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the data to a temporary file
    temp_path = os.path.join(temp_dir, f'file_{file_id}.json')
    data.to_json(temp_path, orient='split')
    
    return temp_path

def load_file_data(file_path):
    """Load file data from temporary storage."""
    if os.path.exists(file_path):
        return pd.read_json(file_path, orient='split')
    return None
    
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
                metrics["davies_bouldin"] = davies_bouldin_score(data[mask], labels[mask])
                metrics["calinski_harabasz"] = calinski_harabasz_score(data[mask], labels[mask])
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
    df_with_clusters['cluster'] = cluster_labels
    
    # Select key physics quantities
    physics_features = ['KER', 'EESum', 'energy_ion1', 'energy_ion2', 
                       'energy_electron1', 'energy_electron2', 'TotalEnergy']
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
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster][feature]
            if len(cluster_data) > 1:
                within_vars.append(cluster_data.var())
                cluster_means.append(cluster_data.mean())
                cluster_sizes.append(len(cluster_data))
        
        if within_vars and sum(cluster_sizes) > 0:
            # Calculate weighted average of within-cluster variance
            avg_within_var = np.average(within_vars, weights=cluster_sizes)
            
            # Calculate between-cluster variance (weighted by cluster size)
            global_mean = df_with_clusters[feature].mean()
            between_var = np.average([(mean - global_mean)**2 for mean in cluster_means], 
                                    weights=cluster_sizes)
            
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
        best_feature = max(results.items(), key=lambda x: x[1] if x[0] != "physics_consistency" else 0)
        if best_feature[0] != "physics_consistency":
            results["most_consistent_feature"] = best_feature[0].replace("_consistency", "")
    else:
        results["physics_consistency"] = 0.0
        results["note"] = "Could not calculate consistency for any features"
    
    return results

def calculate_adaptive_confidence_score(metrics, data_characteristics=None, clustering_method='dbscan'):
    """Calculate an adaptive confidence score with improved weighting and logic."""
    
    # Enhanced tier definitions with better balance
    tier1_metrics = {
        'silhouette': {
            'weight': 0.35,  # Reduced from 0.4
            'reliability': 0.9,
            'description': 'Most reliable cluster quality measure',
            'min_threshold': 0.2,  # Lowered from 0.3
            'good_threshold': 0.4   # Lowered from 0.5
        },
        'hopkins': {
            'weight': 0.25,  # Reduced from 0.3
            'reliability': 0.85,
            'description': 'Fundamental clusterability assessment',
            'min_threshold': 0.5,
            'good_threshold': 0.7   # Lowered from 0.75
        }
    }
    
    # Tier 2: Useful but context-dependent metrics
    tier2_metrics = {
        'stability': {
            'weight': 0.15,  # Reduced from 0.2
            'reliability': 0.7,
            'description': 'Reproducibility under perturbation',
            'min_threshold': 0.3,  # Lowered from 0.4
            'good_threshold': 0.6,  # Lowered from 0.7
            'conditions': ['sufficient_data']
        },
        'physics_consistency': {
            'weight': 0.2,  # Reduced from 0.3
            'reliability': 0.8,
            'description': 'Domain-specific validation',
            'min_threshold': 0.2,  # Lowered from 0.3
            'good_threshold': 0.5,  # Lowered from 0.6
            'conditions': ['physics_relevant']
        },
        'calinski_harabasz': {
            'weight': 0.1,  # Include CH with low weight
            'reliability': 0.6,
            'description': 'Between-cluster separation',
            'min_threshold': 30,
            'good_threshold': 100
        }
    }
    
    # Tier 3: Often misleading metrics (very low weight)
    tier3_metrics = {
        'davies_bouldin': {
            'weight': 0.05,
            'reliability': 0.4,
            'description': 'Can be misleading with noise',
            'issues': ['sensitive_to_noise', 'poor_with_varying_densities']
        }
    }
    
    # Analyze data characteristics if provided
    if data_characteristics is None:
        data_characteristics = analyze_data_characteristics(metrics)
    
    # Adaptive weighting based on context
    adaptive_weights = calculate_adaptive_weights(
        tier1_metrics, tier2_metrics, tier3_metrics, 
        data_characteristics, clustering_method
    )
    
    # Calculate confidence with reliability-adjusted weights
    confidence_result = calculate_weighted_confidence(metrics, adaptive_weights)
    
    # Apply bonus for exceptional clustering
    confidence_result = apply_clustering_bonus(confidence_result, metrics)
    
    # Add context-aware analysis
    confidence_result['analysis'] = analyze_confidence_context(
        confidence_result, metrics, data_characteristics
    )
    
    return confidence_result

def analyze_data_characteristics(metrics):
    """Infer data characteristics from available metrics."""
    characteristics = {
        'has_noise': False,
        'sufficient_data': True,
        'physics_relevant': False,
        'cluster_quality': 'unknown'
    }
    
    # Detect noise presence
    if 'noise_ratio' in metrics and metrics['noise_ratio'] > 0.1:
        characteristics['has_noise'] = True
    
    # Detect if physics features are relevant
    if 'physics_consistency' in metrics:
        characteristics['physics_relevant'] = True
    
    # Assess cluster quality
    if 'silhouette' in metrics:
        if metrics['silhouette'] > 0.5:
            characteristics['cluster_quality'] = 'good'
        elif metrics['silhouette'] > 0.2:
            characteristics['cluster_quality'] = 'moderate'
        else:
            characteristics['cluster_quality'] = 'poor'
    
    return characteristics

def calculate_adaptive_weights(tier1, tier2, tier3, characteristics, clustering_method):
    """Calculate adaptive weights based on context."""
    weights = {}
    
    # Always include Tier 1 metrics (most reliable)
    for metric, info in tier1.items():
        weights[metric] = {
            'weight': info['weight'],
            'reliability': info['reliability'],
            'tier': 1,
            'reason': info['description']
        }
    
    # Conditionally include Tier 2 metrics
    for metric, info in tier2.items():
        include = True
        exclusion_reason = None
        
        if 'conditions' in info:
            for condition in info['conditions']:
                if condition == 'sufficient_data' and not characteristics['sufficient_data']:
                    include = False
                    exclusion_reason = "Insufficient data for reliable stability assessment"
                elif condition == 'physics_relevant' and not characteristics['physics_relevant']:
                    include = False
                    exclusion_reason = "Physics consistency not applicable"
        
        if include:
            weights[metric] = {
                'weight': info['weight'],
                'reliability': info['reliability'],
                'tier': 2,
                'reason': info['description']
            }
        else:
            weights[metric] = {
                'weight': 0,
                'reliability': 0,
                'tier': 2,
                'excluded': True,
                'exclusion_reason': exclusion_reason
            }
    
    # Generally exclude Tier 3 metrics, but include with warnings
    for metric, info in tier3.items():
        weights[metric] = {
            'weight': info['weight'] * 0.1,  # Heavily downweight
            'reliability': info['reliability'],
            'tier': 3,
            'reason': f"Low reliability: {info['description']}",
            'issues': info['issues']
        }
    
    # Normalize weights
    total_weight = sum(w['weight'] for w in weights.values())
    if total_weight > 0:
        for metric in weights:
            weights[metric]['normalized_weight'] = weights[metric]['weight'] / total_weight
    
    return weights

def calculate_weighted_confidence(metrics, adaptive_weights):
    """Calculate confidence score using adaptive weights with minimum floor."""
    
    normalized_scores = {}
    confidence_components = {}
    weighted_sum = 0
    total_reliability_weight = 0
    
    # Count how many metrics we actually have
    available_metrics = 0
    
    for metric, weight_info in adaptive_weights.items():
        if metric in metrics and weight_info['weight'] > 0:
            available_metrics += 1
            
            # Normalize metric to 0-1 scale
            normalized_value = normalize_metric(metric, metrics[metric])
            normalized_scores[metric] = normalized_value
            
            # Weight by both importance and reliability
            effective_weight = weight_info['weight'] * weight_info['reliability']
            contribution = normalized_value * effective_weight
            
            confidence_components[metric] = {
                'raw_value': metrics[metric],
                'normalized_value': normalized_value,
                'weight': weight_info['weight'],
                'reliability': weight_info['reliability'],
                'effective_weight': effective_weight,
                'contribution': contribution,
                'tier': weight_info['tier'],
                'reason': weight_info['reason']
            }
            
            weighted_sum += contribution
            total_reliability_weight += effective_weight
    
    # Calculate overall confidence with safeguards
    if total_reliability_weight > 0:
        overall_confidence = weighted_sum / total_reliability_weight
    else:
        overall_confidence = 0.1
    
    # Apply minimum confidence based on available metrics
    if available_metrics > 0:
        # If we have at least one metric, ensure minimum confidence
        min_confidence = 0.2 + (available_metrics * 0.05)  # More metrics = higher floor
        overall_confidence = max(overall_confidence, min_confidence)
    
    # Apply boost if primary metrics are good
    if 'silhouette' in normalized_scores and normalized_scores['silhouette'] > 0.6:
        overall_confidence = max(overall_confidence, 0.5)  # At least moderate
    
    # Ensure confidence is in valid range [0, 1]
    overall_confidence = np.clip(overall_confidence, 0, 1)
    
    # Adjust confidence based on critical thresholds
    adjusted_confidence = apply_critical_thresholds(overall_confidence, confidence_components)
    
    # Calculate reliability score
    num_metrics_with_weight = len([w for w in adaptive_weights.values() if w['weight'] > 0])
    if num_metrics_with_weight > 0:
        reliability_score = total_reliability_weight / num_metrics_with_weight
    else:
        reliability_score = 0
    
    return {
        'overall_confidence': adjusted_confidence,
        'raw_confidence': overall_confidence,
        'components': confidence_components,
        'adaptive_weights': adaptive_weights,
        'confidence_level': categorize_confidence(adjusted_confidence),
        'reliability_score': reliability_score,
        'available_metrics': available_metrics,
        'debug_info': {
            'weighted_sum': weighted_sum,
            'total_reliability_weight': total_reliability_weight,
            'num_components': len(confidence_components),
            'num_metrics_available': len(metrics)
        }
    }

def normalize_metric(metric_name, value):
    """Normalize different metrics to 0-1 scale (higher = better) with improved scaling."""
    
    # Handle edge cases
    if value is None or np.isnan(value) or np.isinf(value):
        print(f"DEBUG normalize_metric: Invalid value for {metric_name}: {value}")
        return 0
    
    if metric_name == 'silhouette':
        # Silhouette score range: [-1, 1]
        # More nuanced scaling that doesn't penalize moderate clustering too harshly
        if value < -0.25:
            return 0  # Very poor clustering
        elif value < 0:
            # Scale -0.25 to 0 → 0 to 0.2
            return 0.2 * (1 + value/0.25)
        elif value < 0.25:
            # Scale 0 to 0.25 → 0.2 to 0.5
            return 0.2 + (value / 0.25) * 0.3
        elif value < 0.5:
            # Scale 0.25 to 0.5 → 0.5 to 0.7
            return 0.5 + ((value - 0.25) / 0.25) * 0.2
        elif value < 0.7:
            # Scale 0.5 to 0.7 → 0.7 to 0.85
            return 0.7 + ((value - 0.5) / 0.2) * 0.15
        else:
            # Scale 0.7 to 1.0 → 0.85 to 1.0
            return 0.85 + ((value - 0.7) / 0.3) * 0.15
            
    elif metric_name == 'hopkins':
        # Hopkins statistic range: [0, 1], higher is better
        # More generous scaling for hopkins
        if value < 0.5:
            return value * 0.4  # Poor clustering tendency
        elif value < 0.7:
            return 0.2 + ((value - 0.5) / 0.2) * 0.3  # Moderate
        else:
            return 0.5 + ((value - 0.7) / 0.3) * 0.5  # Good to excellent
        
    elif metric_name == 'stability':
        # Stability range: [0, 1], higher is better
        # Stability is often lower, so be more generous
        if value < 0.3:
            return value * 0.5
        elif value < 0.6:
            return 0.15 + ((value - 0.3) / 0.3) * 0.35
        else:
            return 0.5 + ((value - 0.6) / 0.4) * 0.5
        
    elif metric_name == 'physics_consistency':
        # Physics consistency range: [0, 1], higher is better
        return np.clip(value ** 0.7, 0, 1)  # Slight curve to be more generous
        
    elif metric_name == 'davies_bouldin':
        # Davies-Bouldin range: [0, ∞], lower is better
        # More reasonable transformation
        if value <= 0.3:
            return 1.0  # Excellent
        elif value <= 0.5:
            return 0.9 - (value - 0.3) * 0.5  # Very good
        elif value <= 1.0:
            return 0.8 - (value - 0.5) * 0.6  # Good
        elif value <= 1.5:
            return 0.5 - (value - 1.0) * 0.6  # Moderate
        elif value <= 2.5:
            return 0.2 - (value - 1.5) * 0.15  # Poor
        else:
            return max(0, 0.05)  # Very poor
            
    elif metric_name == 'calinski_harabasz':
        # Calinski-Harabasz range: [0, ∞], higher is better
        # Log transform for better scaling
        if value <= 0:
            return 0
        else:
            # Log scale that's more generous
            log_val = np.log1p(value / 10)  # log(1 + value/10)
            return np.tanh(log_val / 3)  # Smoother curve
            
    elif metric_name == 'noise_ratio':
        # Noise ratio range: [0, 1], lower is better
        # More tolerance for noise
        if value < 0.1:
            return 1.0  # Excellent
        elif value < 0.2:
            return 0.9 - (value - 0.1) * 2  # Good
        elif value < 0.4:
            return 0.7 - (value - 0.2) * 2  # Moderate
        else:
            return max(0.1, 0.3 - (value - 0.4) * 0.5)  # Poor
        
    else:
        # Default: assume [0, 1] range, higher is better
        print(f"DEBUG normalize_metric: Unknown metric {metric_name}, using default normalization")
        return np.clip(value, 0, 1)

def apply_critical_thresholds(confidence, components):
    """Apply critical thresholds with more reasonable penalties."""
    
    critical_failures = []
    moderate_issues = []
    
    # Check for critical failures (but be more lenient)
    if 'silhouette' in components:
        sil_raw = components['silhouette']['raw_value']
        if sil_raw < -0.1:  # Changed from 0.1
            critical_failures.append(f"Silhouette score very low ({sil_raw:.3f})")
        elif sil_raw < 0.2:
            moderate_issues.append(f"Silhouette score moderate ({sil_raw:.3f})")
    
    if 'hopkins' in components:
        hop_raw = components['hopkins']['raw_value']
        if hop_raw < 0.3:
            critical_failures.append(f"Hopkins statistic too low ({hop_raw:.3f}) - data appears random")
        elif hop_raw < 0.6:
            moderate_issues.append(f"Hopkins statistic moderate ({hop_raw:.3f})")
    
    # Apply penalties based on severity
    if critical_failures:
        # Cap at 0.4 instead of 0.3
        confidence = min(confidence, 0.4)
    elif moderate_issues:
        # Minor penalty for moderate issues
        confidence = min(confidence, 0.7)
    
    # Boost for exceptional cases
    exceptional_count = 0
    if 'silhouette' in components and components['silhouette']['raw_value'] > 0.6:
        exceptional_count += 1
    if 'hopkins' in components and components['hopkins']['raw_value'] > 0.75:
        exceptional_count += 1
    if 'stability' in components and components['stability']['raw_value'] > 0.8:
        exceptional_count += 1
    
    if exceptional_count >= 2:
        confidence = min(1.0, confidence * 1.15)  # 15% boost
    
    return confidence

def apply_clustering_bonus(confidence_result, metrics):
    """Apply bonus points for exceptional clustering characteristics."""
    
    bonus = 0
    bonus_reasons = []
    
    # Check for exceptional silhouette score
    if 'silhouette' in metrics and metrics['silhouette'] > 0.6:
        bonus += 0.1
        bonus_reasons.append("Excellent cluster separation")
    
    # Check for very low noise ratio
    if 'noise_ratio' in metrics and metrics['noise_ratio'] < 0.05:
        bonus += 0.05
        bonus_reasons.append("Very low noise")
    
    # Check for high stability
    if 'stability' in metrics and metrics['stability'] > 0.8:
        bonus += 0.05
        bonus_reasons.append("High cluster stability")
    
    # Check for good hopkins statistic
    if 'hopkins' in metrics and metrics['hopkins'] > 0.8:
        bonus += 0.05
        bonus_reasons.append("Strong clustering tendency")
    
    # Apply bonus with cap
    if bonus > 0:
        original_confidence = confidence_result['overall_confidence']
        new_confidence = min(1.0, original_confidence + bonus)
        confidence_result['overall_confidence'] = new_confidence
        confidence_result['bonus_applied'] = bonus
        confidence_result['bonus_reasons'] = bonus_reasons
    
    return confidence_result


def categorize_confidence(score):
    """Categorize confidence score with more generous thresholds."""
    if score >= 0.8:  # Lowered from 0.85
        return {"level": "Excellent", "color": "darkgreen", "description": "Very reliable results"}
    elif score >= 0.65:  # Lowered from 0.7
        return {"level": "High", "color": "green", "description": "Reliable results"}
    elif score >= 0.5:  # Lowered from 0.55
        return {"level": "Moderate", "color": "orange", "description": "Reasonably reliable"}
    elif score >= 0.35:  # Lowered from 0.4
        return {"level": "Low", "color": "red", "description": "Use with caution"}
    else:
        return {"level": "Very Low", "color": "darkred", "description": "Results may be unreliable"}

def analyze_confidence_context(confidence_result, metrics, characteristics):
    """Provide context-aware analysis and recommendations."""
    
    analysis = {
        'primary_factors': [],
        'concerns': [],
        'recommendations': [],
        'reliability_notes': []
    }
    
    # Identify primary confidence drivers
    sorted_components = sorted(
        confidence_result['components'].items(),
        key=lambda x: x[1]['contribution'],
        reverse=True
    )
    
    for metric, data in sorted_components[:2]:
        analysis['primary_factors'].append(
            f"{metric.replace('_', ' ').title()}: {data['normalized_value']:.2f} "
            f"(contributes {data['contribution']/confidence_result['raw_confidence']:.1%})"
        )
    
    # Identify concerns
    for metric, data in confidence_result['components'].items():
        if data['normalized_value'] < 0.4:
            analysis['concerns'].append(
                f"Low {metric.replace('_', ' ')}: {data['raw_value']:.3f}"
            )
    
    # Generate recommendations
    if confidence_result['overall_confidence'] < 0.5:
        analysis['recommendations'].extend([
            "Consider different UMAP parameters (n_neighbors, min_dist)",
            "Try alternative feature selection or engineering",
            "Verify data quality and preprocessing"
        ])
    
    if 'silhouette' in confidence_result['components']:
        sil_val = confidence_result['components']['silhouette']['raw_value']
        if sil_val < 0.3:
            analysis['recommendations'].append(
                "Poor cluster separation - try increasing n_neighbors or different clustering algorithm"
            )
    
    # Add reliability notes
    for metric, weight_info in confidence_result['adaptive_weights'].items():
        if weight_info.get('excluded'):
            analysis['reliability_notes'].append(
                f"{metric}: {weight_info['exclusion_reason']}"
            )
        elif weight_info['tier'] == 3:
            analysis['reliability_notes'].append(
                f"{metric}: {weight_info['reason']}"
            )
    
    return analysis

def get_metric_color(metric_name, value):
    """Get color for metric value based on thresholds."""
    if metric_name == 'silhouette':
        if value > 0.5:
            return "green"
        elif value > 0.25:
            return "orange"
        else:
            return "red"
    elif metric_name == 'davies_bouldin':
        if value < 0.8:
            return "green"
        elif value < 1.5:
            return "orange"
        else:
            return "red"
    elif metric_name == 'calinski_harabasz':
        if value > 100:
            return "green"
        elif value > 50:
            return "orange"
        else:
            return "red"
    elif metric_name == 'hopkins':
        if value > 0.75:
            return "green"
        elif value > 0.6:
            return "orange"
        else:
            return "red"
    elif metric_name == 'stability':
        if value > 0.8:
            return "green"
        elif value > 0.6:
            return "orange"
        else:
            return "red"
    elif metric_name == 'physics_consistency':
        if value > 0.6:
            return "green"
        elif value > 0.3:
            return "orange"
        else:
            return "red"
    elif metric_name == 'noise_ratio':
        # For noise ratio, lower is better
        if value < 0.1:
            return "green"
        elif value < 0.3:
            return "orange"
        else:
            return "red"
    else:
        return "black"


def create_smart_confidence_ui(confidence_data):
    """Create an intelligent confidence UI."""
    
    if not confidence_data:
        return html.Div("No confidence data available")
    
    conf_score = confidence_data['overall_confidence']
    conf_info = confidence_data['confidence_level']
    analysis = confidence_data.get('analysis', {})
    components = confidence_data.get('components', {})  # Added this line - was missing
    
    return html.Div([
        # Header
        html.Div([
            html.H4("UMAP Reliability Assessment", 
                   style={"fontSize": "16px", "marginBottom": "5px", "color": "#2e7d32"}),
            html.Div([
                html.Span(f"{conf_score:.2f}", style={
                    "fontSize": "28px", 
                    "fontWeight": "bold", 
                    "color": conf_info['color'],
                    "marginRight": "15px"
                }),
                html.Div([
                    html.Div(conf_info['level'], style={
                        "fontSize": "16px",
                        "fontWeight": "bold",
                        "color": conf_info['color']
                    }),
                    html.Div(conf_info['description'], style={
                        "fontSize": "12px",
                        "color": "gray",
                        "fontStyle": "italic"
                    })
                ])
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "15px"}),
        ]),
        
        # Confidence bar
        create_confidence_bar(conf_score, conf_info['color']),

        # Raw Metrics Display
        html.Div([
            html.H5("Individual Metrics", 
                   style={"fontSize": "14px", "marginTop": "15px", "marginBottom": "8px", 
                          "color": "#1976d2", "borderBottom": "1px solid #e0e0e0"}),
            html.Div([
                html.Div([
                    html.Span(f"{metric.replace('_', ' ').title()}: ", style={"fontWeight": "bold"}),
                    html.Span(f"{data['raw_value']:.4f}", 
                            style={"color": get_metric_color(metric, data['raw_value'])})
                ], style={"fontSize": "12px", "marginBottom": "3px"})
                for metric, data in components.items()
                if 'raw_value' in data
            ])
        ]),
        
        # Key indicators
        html.Div([
            html.H5("Key Quality Indicators", 
                   style={"fontSize": "14px", "marginTop": "15px", "marginBottom": "8px", 
                          "color": "#1976d2", "borderBottom": "1px solid #e0e0e0"}),
            html.Div([
                html.Div(factor, style={"fontSize": "12px", "marginBottom": "3px"})
                for factor in analysis.get('primary_factors', [])
            ] if analysis.get('primary_factors') else [
                html.Div("No primary factors identified", style={"fontSize": "12px", "color": "gray"})
            ])
        ]),
        
        # Recommendations
        html.Div([
            html.H6("💡 Recommendations", style={
                "fontSize": "12px", 
                "color": "#1976d2", 
                "marginTop": "12px", 
                "marginBottom": "5px",
                "fontWeight": "bold"
            }),
            html.Ul([
                html.Li(rec, style={"fontSize": "11px", "marginBottom": "2px"})
                for rec in analysis.get('recommendations', ["Results appear reliable"])
            ], style={"paddingLeft": "15px", "margin": "0"})
        ]) if analysis.get('recommendations') else html.Div([
            html.H6("✓ Status", style={
                "fontSize": "12px", 
                "color": "#2e7d32", 
                "marginTop": "12px", 
                "marginBottom": "5px",
                "fontWeight": "bold"
            }),
            html.Div("Results appear reliable based on current metrics", 
                    style={"fontSize": "11px", "color": "#2e7d32"})
        ]),  # Added missing comma here

        # Metric explanations tooltip
        html.Div([
            html.Hr(style={"marginTop": "10px", "marginBottom": "10px"}),
            html.Details([
                html.Summary("What do these metrics mean?", style={"cursor": "pointer"}),
                html.Div([
                    html.P("• Silhouette Score: Measures how well-separated clusters are (higher is better, range: -1 to 1)"),
                    html.P("• Davies-Bouldin Index: Measures average similarity between clusters (lower is better, range: 0 to ∞)"),
                    html.P("• Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion (higher is better)"),
                    html.P("• Hopkins Statistic: Measures clusterability of the data (>0.75 indicates good clustering)"),
                    html.P("• Cluster Stability: How stable clusters are with small perturbations (higher is better)"),
                    html.P("• Physics Consistency: How well clusters align with physical parameters (higher is better)")
                ], style={"fontSize": "11px", "paddingLeft": "10px"})
            ])
        ], style={"marginTop": "10px"})
        
    ], style={
        "padding": "15px",
        "border": f"2px solid {conf_info['color']}",
        "borderRadius": "8px",
        "backgroundColor": "#fafafa",
        "marginTop": "15px"
    })

def create_confidence_bar(score, color):
    """Create a segmented confidence bar."""
    
    segments = [
        {"threshold": 0.0, "color": "#d32f2f", "label": "Very Low"},
        {"threshold": 0.4, "color": "#f57c00", "label": "Low"},
        {"threshold": 0.55, "color": "#fbc02d", "label": "Moderate"},
        {"threshold": 0.7, "color": "#689f38", "label": "High"},
        {"threshold": 0.85, "color": "#388e3c", "label": "Excellent"}
    ]
    
    bar_segments = []
    for i, segment in enumerate(segments):
        if i < len(segments) - 1:
            width = segments[i+1]["threshold"] - segment["threshold"]
        else:
            width = 1.0 - segment["threshold"]
        
        is_active = (score >= segment["threshold"] and 
                    (i == len(segments)-1 or score < segments[i+1]["threshold"]))
        
        opacity = 1.0 if is_active else 0.3
        
        bar_segments.append(
            html.Div(
                segment["label"] if is_active else "",
                style={
                    "width": f"{width * 100}%",
                    "height": "25px",
                    "backgroundColor": segment["color"],
                    "opacity": opacity,
                    "textAlign": "center",
                    "fontSize": "10px",
                    "lineHeight": "25px",
                    "color": "white" if is_active else "transparent",
                    "fontWeight": "bold" if is_active else "normal",
                    "border": "1px solid white",
                    "boxSizing": "border-box"  # Add this
                }
            )
        )
    
    return html.Div(bar_segments, style={
        "position": "relative",
        "width": "100%",
        "marginBottom": "15px",
        "display": "flex",        # Change from default to flex
        "flexDirection": "row",   # Ensure horizontal layout
        "flexWrap": "nowrap"      # Prevent wrapping
    })

# Define the DeepAutoencoder class
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=7):
        super(DeepAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim)  # Latent space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.SiLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, input_dim)  # Reconstruct original compressed features
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Supervised Clustering and Uncovering Latent Patterns with Training SCULPT"

app.current_epoch = 0
app.total_epochs = 0

# Add simple CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
   <head>
       {%metas%}
       <title>{%title%}</title>
       {%favicon%}
       {%css%}
       <style>
           body {
               font-family: Arial, sans-serif;
               max-width: 1800px;
               margin: 0 auto;
               padding: 20px;
               background-color: #f5f5f5;
           }
           
           .container {
               padding: 20px;
               border-radius: 5px;
               background-color: white;
               margin-bottom: 20px;
               box-shadow: 0 2px 4px rgba(0,0,0,0.1);
           }
           
           h1, h2, h3, h4 {
               color: #333;
           }
           
           /* Tab styles */
           .tab {
               background-color: #f9f9f9;
               border: 1px solid #ddd;
               border-bottom: none;
               padding: 10px 20px;
               cursor: pointer;
               transition: all 0.3s ease;
           }
           
           .tab--selected {
               background-color: white;
               border-top: 3px solid #4CAF50;
               font-weight: bold;
           }
           
           .tab-container {
               background-color: white;
               border: 1px solid #ddd;
               padding: 20px;
               border-radius: 0 0 5px 5px;
           }
           
           /* Button styles */
           button {
               background-color: #4CAF50;
               color: white;
               border: none;
               padding: 10px 15px;
               border-radius: 4px;
               margin-top: 10px;
               cursor: pointer;
               width: 100%;
               transition: background-color 0.3s ease;
           }
           
           button:hover {
               background-color: #45a049;
           }
           
           .btn-secondary {
               background-color: #008CBA;
           }
           
           .btn-secondary:hover {
               background-color: #007399;
           }
           
           /* Simple pressed state */
           button:active {
               background-color: #555;
               box-shadow: inset 0 3px 5px rgba(0,0,0,0.2);
               transform: translateY(2px);
           }
           
           /* Feature selection checklist */
           .feature-checklist {
               max-height: 300px;
               overflow-y: auto;
               border: 1px solid #ddd;
               padding: 10px;
               background-color: white;
               border-radius: 5px;
           }
           
           .feature-category {
               margin-bottom: 10px;
               border-bottom: 1px solid #eee;
               padding-bottom: 5px;
           }
           
           .feature-category-title {
               font-weight: bold;
               margin-bottom: 5px;
               color: #555;
           }
           
           /* Upload area */
           #upload-data {
               background-color: #fafafa;
               transition: background-color 0.3s ease;
           }
           
           #upload-data:hover {
               background-color: #f0f0f0;
               border-color: #4CAF50;
           }
           
           /* Metric display */
           .metric-value {
               font-size: 18px;
               font-weight: bold;
               margin-left: 10px;
           }
           
           /* Scrollbar styling */
           ::-webkit-scrollbar {
               width: 10px;
           }
           
           ::-webkit-scrollbar-track {
               background: #f1f1f1;
           }
           
           ::-webkit-scrollbar-thumb {
               background: #888;
               border-radius: 5px;
           }
           
           ::-webkit-scrollbar-thumb:hover {
               background: #555;
           }
       </style>
   </head>
   <body>
       {%app_entry%}
       <footer>
           {%config%}
           {%scripts%}
           {%renderer%}
       </footer>
   </body>
</html>
'''

# Replace the main layout section (starting from app.layout) with this:

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Header
    html.Div([
        html.H1("SCULPT", style={'textAlign': 'center', 'marginBottom': '5px'}),
        html.P("Supervised Clustering and Uncovering Latent Patterns with Training", 
               style={'textAlign': 'center', 'color': 'gray', 'marginTop': '0px'})
    ]),
    
    # Main Tabs
    dcc.Tabs(id='main-tabs', value='tab-data', children=[
        
        # Tab 1: Data Management
        dcc.Tab(label='Data & Configuration', value='tab-data', children=[
            html.Div([
                # File Upload Section
                html.Div([
                    html.H3("File Upload"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Drag and drop or ', html.A('select data files')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                        },
                        multiple=True
                    ),
                    html.Div(id='file-list', children=[])
                ], className='container'),
                
                # Configuration Section
                html.Div([
                    html.H3("Molecular Configuration Management"),
                    
                    # Sub-tabs for configuration
                    dcc.Tabs(id='config-tabs', value='config-profiles', children=[
                        # Configuration Profiles tab
                        dcc.Tab(label='Configuration Profiles', value='config-profiles', children=[
                            html.Div([
                                html.Div([
                                    html.H4("Configuration Profiles"),
                                    
                                    # Active Profiles Display
                                    html.Div([
                                        html.H5("Active Profiles", style={'color': '#1976d2', 'marginBottom': '15px'}),
                                        html.Div(id='active-profiles-display', children=[
                                            html.Div("No profiles created yet", style={"color": "gray", "fontStyle": "italic"})
                                        ], style={'minHeight': '100px', 'padding': '15px', 'backgroundColor': '#f5f5f5', 
                                                 'borderRadius': '5px', 'marginBottom': '20px'})
                                    ]),
                                    
                                    # Particle Configuration Section
                                    html.Div([
                                        html.H5("Particle Configuration", style={'color': '#6a1b9a', 'marginBottom': '15px'}),
                                        html.Div([
                                            html.Div([
                                                html.Label("Number of Ions:", style={'marginRight': '10px'}),
                                                dcc.Input(id='num-ions', type='number', value=2, min=0, max=10, 
                                                         style={'width': '80px', 'marginRight': '20px'}),
                                            ], style={'display': 'inline-block', 'marginRight': '30px'}),
                                            
                                            html.Div([
                                                html.Label("Number of Neutrals:", style={'marginRight': '10px'}),
                                                dcc.Input(id='num-neutrals', type='number', value=1, min=0, max=10, 
                                                         style={'width': '80px', 'marginRight': '20px'}),
                                            ], style={'display': 'inline-block', 'marginRight': '30px'}),
                                            
                                            html.Div([
                                                html.Label("Number of Electrons:", style={'marginRight': '10px'}),
                                                dcc.Input(id='num-electrons', type='number', value=2, min=0, max=10, 
                                                         style={'width': '80px'}),
                                            ], style={'display': 'inline-block'}),
                                        ], style={'marginBottom': '15px', 'padding': '10px', 
                                                 'backgroundColor': '#f5f5f5', 'borderRadius': '5px'}),
                                        
                                        # Dynamic particle configuration container
                                        html.Div(id='particle-config-container', children=[]),
                                    ], style={'padding': '15px', 'backgroundColor': '#f3e5f5', 
                                             'borderRadius': '5px', 'marginBottom': '20px'}),
                                    
                                    # Create New Profile Section
                                    html.Div([
                                        html.H5("Create New Profile", style={'color': '#388e3c', 'marginBottom': '15px'}),
                                        html.Div([
                                            html.Div([
                                                html.Label("Profile Name:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                                                dcc.Input(
                                                    id='config-profile-name', 
                                                    type='text', 
                                                    placeholder='e.g., D2O, HDO, H2O', 
                                                    style={'width': '250px', 'marginRight': '20px'}
                                                ),
                                                html.Button(
                                                    "Create New Profile", 
                                                    id='create-profile-btn', 
                                                    n_clicks=0, 
                                                    className="btn-secondary", 
                                                    style={'width': 'auto', 'padding': '10px 20px'}
                                                ),
                                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
                                        ], style={'padding': '15px', 'backgroundColor': '#e8f5e9', 
                                                 'borderRadius': '5px', 'marginBottom': '20px'})
                                    ]),
                                    
                                    # Selected Profile Editor
                                    html.Div([
                                        html.H5("Edit Selected Profile", style={'color': '#f57c00', 'marginBottom': '15px'}),
                                        html.Div([
                                            html.Label("Select Profile to Edit:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                                            dcc.Dropdown(
                                                id='active-profile-dropdown',
                                                options=[],
                                                value=None,
                                                style={'width': '300px', 'marginBottom': '15px'}
                                            ),
                                            html.Button(
                                                "Update Profile", 
                                                id='update-profile-btn', 
                                                n_clicks=0, 
                                                className="btn-secondary", 
                                                style={'width': 'auto', 'padding': '10px 20px', 'marginTop': '10px'}
                                            ),
                                            html.Div(id='profile-edit-status', style={'marginTop': '10px', 'color': 'green'})
                                        ], style={'padding': '15px', 'backgroundColor': '#fff3e0', 
                                                 'borderRadius': '5px'})
                                    ])
                                ], style={'padding': '20px'})
                            ], style={'padding': '20px'})
                        ]),
                        
                        dcc.Tab(label='File Assignment', value='file-assignment', children=[
                            html.Div([
                                html.H4("Assign Configuration Profiles to Files"),
                                html.Div(id='file-configuration-assignment', children=[
                                    html.Div("Upload files first to assign configurations", 
                                            style={"color": "gray", "fontStyle": "italic"})
                                ]),
                                html.Br(),
                                html.Button("Apply File Assignments", id='apply-file-config-btn', 
                                           n_clicks=0, className="btn-secondary"),
                                html.Div(id='file-assignment-status', style={'marginTop': '10px', 'color': 'green'}),
                            ], style={'padding': '20px'})
                        ])
                    ])
                ], className='container'),
            ])
        ]),
        
        # Tab 2: Basic Visualizations
        dcc.Tab(label='Basic Analysis', value='tab-basic', children=[
            dcc.Tabs(id='basic-sub-tabs', value='umap-tab', children=[
                # UMAP Tab
                dcc.Tab(label='UMAP Embedding', value='umap-tab', children=[
                    html.Div([
                        html.H3("UMAP Embedding Analysis", style={'textAlign': 'center'}),
                        # Main container with flex layout
                        html.Div([
                            # Left panel with controls
                            html.Div([
                                html.H4("Select Files for UMAP:"),
                                dcc.Checklist(id='umap-file-selector', options=[], value=[], labelStyle={'display': 'block'}),
                                html.Br(),
                                html.H4("Select Features for UMAP:"),
                                html.Div(id='feature-selection-ui-graph1', children=[
                                    html.Div("Upload files to see available features", style={"color": "gray"})
                                ], className='feature-checklist'),
                                html.Br(),
                                
                                # UMAP Parameters
                                html.Div([
                                    html.H5("UMAP Parameters"),
                                    html.Label("Number of Neighbors:"),
                                    dcc.Input(id='num-neighbors', type='number', value=15, min=1),
                                    html.Br(),
                                    html.Label("Min Dist:"),
                                    dcc.Input(id='min-dist', type='number', value=0.1, step=0.01, min=0),
                                    html.Br(),
                                    html.Label("Sample Fraction (0-1):"),
                                    dcc.Input(id='sample-frac', type='number', value=0.01, min=0.001, max=1, step=0.001),
                                ], style={'backgroundColor': '#f5f5f5', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '15px'}),
                                
                                # Visualization Options
                                html.Div([
                                    html.H5("Visualization Options"),
                                    html.Label("Visualization Type:"),
                                    dcc.RadioItems(
                                        id='visualization-type',
                                        options=[
                                            {'label': 'Scatter Plot', 'value': 'scatter'},
                                            {'label': 'Heatmap', 'value': 'heatmap'}
                                        ],
                                        value='scatter',
                                        labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                                    ),
                                    html.Br(),
                                    html.Div(id='scatter-settings', children=[
                                        html.Label("Point Opacity:"),
                                        dcc.Slider(
                                            id='point-opacity',
                                            min=0.05,
                                            max=1.0,
                                            step=0.05,
                                            value=0.3,
                                            marks={i/10: str(i/10) for i in range(1, 11, 2)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                    ]),
                                    html.Div(id='heatmap-settings', children=[
                                        html.Label("Heatmap Bandwidth:"),
                                        dcc.Slider(
                                            id='heatmap-bandwidth',
                                            min=0.05,
                                            max=1.0,
                                            step=0.05,
                                            value=0.2,
                                            marks={i/10: str(i/10) for i in range(1, 11, 2)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                        html.Br(),
                                        html.Label("Heatmap Color Scale:"),
                                        dcc.Dropdown(
                                            id='heatmap-colorscale',
                                            options=[
                                                {'label': 'Viridis', 'value': 'Viridis'},
                                                {'label': 'Plasma', 'value': 'Plasma'},
                                                {'label': 'Inferno', 'value': 'Inferno'},
                                                {'label': 'Magma', 'value': 'Magma'},
                                                {'label': 'Cividis', 'value': 'Cividis'},
                                                {'label': 'Turbo', 'value': 'Turbo'},
                                                {'label': 'Hot', 'value': 'Hot'},
                                                {'label': 'Jet', 'value': 'Jet'}
                                            ],
                                            value='Viridis',
                                            clearable=False
                                        ),
                                        html.Br(),
                                        html.Label("Show Points Overlay:"),
                                        dcc.RadioItems(
                                            id='show-points-overlay',
                                            options=[
                                                {'label': 'Yes', 'value': 'yes'},
                                                {'label': 'No', 'value': 'no'}
                                            ],
                                            value='yes',
                                            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                                        ),
                                    ], style={'display': 'none'}),
                                ], style={'backgroundColor': '#f5f5f5', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '15px'}),
                                
                                html.Label("Color by:"),
                                dcc.RadioItems(
                                    id='color-mode',
                                    options=[
                                        {'label': 'File Source', 'value': 'file'},
                                        {'label': 'DBSCAN Clusters', 'value': 'cluster'}
                                    ],
                                    value='file',
                                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                                ),
                                html.Br(),
                                html.Button("Run UMAP", id='run-umap', n_clicks=0),
                                html.Div(id="run-umap-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                html.Br(),
                                html.Label("Select Metrics to Calculate:"),
                                dcc.Checklist(
                                    id='metric-selector',
                                    options=[
                                        {'label': 'Silhouette Score', 'value': 'silhouette'},
                                        {'label': 'Davies-Bouldin Index', 'value': 'davies_bouldin'},
                                        {'label': 'Calinski-Harabasz Index', 'value': 'calinski_harabasz'},
                                        {'label': 'Hopkins Statistic', 'value': 'hopkins'},
                                        {'label': 'Cluster Stability', 'value': 'stability'},
                                        {'label': 'Physics Consistency', 'value': 'physics_consistency'}
                                    ],
                                    value=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
                                    labelStyle={'display': 'block'}
                                ),
                            ], style={'width': '25%', 'paddingRight': '20px'}),
                            
                            # Right panel with graph
                            html.Div([
                                dcc.Graph(
                                    id='umap-graph', 
                                    config={
                                        'displayModeBar': True,
                                        'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                                    }, 
                                    style={'height': '600px'}
                                ),
                                html.Div(id='debug-output', style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'}),
                                html.Div(id='umap-quality-metrics', children=[], 
                                        style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #ddd', 
                                               'borderRadius': '5px', 'backgroundColor': '#f9f9f9'}),
                            ], style={'width': '75%'})
                        ], style={'display': 'flex'})
                    ], className='container')
                ]),
                
                # Custom Scatter Tab
                dcc.Tab(label='Custom Feature Plot', value='custom-scatter-tab', children=[
                    html.Div([
                        html.H3("Enhanced Custom Scatter Plot", style={'textAlign': 'center'}),
                        # Main container with flex layout
                        html.Div([
                            # Left panel
                            html.Div([
                                html.H4("Select Files:"),
                                dcc.Checklist(id='file-selector-graph15', options=[], value=[], labelStyle={'display': 'block'}),
                                html.Br(),
                                html.H4("Select Features to Plot:"),
                                html.Div([
                                    html.Label("X-Axis Feature:"),
                                    dcc.Dropdown(
                                        id='x-axis-feature-graph15',
                                        options=[],
                                        value=None,
                                        placeholder="Select X-Axis Feature"
                                    ),
                                    html.Br(),
                                    html.Label("Y-Axis Feature:"),
                                    dcc.Dropdown(
                                        id='y-axis-feature-graph15',
                                        options=[],
                                        value=None,
                                        placeholder="Select Y-Axis Feature"
                                    ),
                                ]),
                                html.Br(),
                                html.Label("Sample Fraction (0-1):"),
                                dcc.Input(id='sample-frac-graph15', type='number', value=0.1, min=0.001, max=1, step=0.001),
                                html.Br(),
                                html.Br(),
                                html.Label("Visualization Type:"),
                                dcc.RadioItems(
                                    id='visualization-type-graph15',
                                    options=[
                                        {'label': 'Scatter Plot', 'value': 'scatter'},
                                        {'label': 'Heatmap', 'value': 'heatmap'}
                                    ],
                                    value='scatter',
                                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                                ),
                                html.Br(),
                                html.Div(id='scatter-settings-graph15', children=[
                                    html.Label("Point Opacity:"),
                                    dcc.Slider(
                                        id='point-opacity-graph15',
                                        min=0.05,
                                        max=1.0,
                                        step=0.05,
                                        value=0.3,
                                        marks={i/10: str(i/10) for i in range(1, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ]),
                                html.Div(id='heatmap-settings-graph15', children=[
                                    html.Label("Heatmap Bandwidth:"),
                                    dcc.Slider(
                                        id='heatmap-bandwidth-graph15',
                                        min=0.05,
                                        max=1.0,
                                        step=0.05,
                                        value=0.2,
                                        marks={i/10: str(i/10) for i in range(1, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    html.Br(),
                                    html.Label("Heatmap Color Scale:"),
                                    dcc.Dropdown(
                                        id='heatmap-colorscale-graph15',
                                        options=[
                                            {'label': 'Viridis', 'value': 'Viridis'},
                                            {'label': 'Plasma', 'value': 'Plasma'},
                                            {'label': 'Inferno', 'value': 'Inferno'},
                                            {'label': 'Magma', 'value': 'Magma'},
                                            {'label': 'Cividis', 'value': 'Cividis'},
                                            {'label': 'Turbo', 'value': 'Turbo'},
                                            {'label': 'Hot', 'value': 'Hot'},
                                            {'label': 'Jet', 'value': 'Jet'}
                                        ],
                                        value='Viridis',
                                        clearable=False
                                    ),
                                    html.Br(),
                                    html.Label("Show Points Overlay:"),
                                    dcc.RadioItems(
                                        id='show-points-overlay-graph15',
                                        options=[
                                            {'label': 'Yes', 'value': 'yes'},
                                            {'label': 'No', 'value': 'no'}
                                        ],
                                        value='yes',
                                        labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                                    ),
                                ], style={'display': 'none'}),
                                html.Br(),
                                html.Label("Color by:"),
                                dcc.RadioItems(
                                    id='color-mode-graph15',
                                    options=[
                                        {'label': 'File Source', 'value': 'file'},
                                        {'label': 'DBSCAN Clusters', 'value': 'cluster'}
                                    ],
                                    value='file',
                                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                                ),
                                html.Br(),
                                html.Button("Generate Plot", id='generate-plot-graph15', n_clicks=0),
                                html.Div(id="generate-plot-graph15-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                html.Br(),
                                # Save selection components
                                html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                html.Label("Save selected points to file:"),
                                html.Div([
                                    dcc.Input(
                                        id="selection-filename-graph15", 
                                        type="text", 
                                        placeholder="Enter filename (without extension)",
                                        style={"width": "100%", "marginBottom": "10px"}
                                    ),
                                    html.Button("Save Selection", id="save-selection-graph15-btn", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="save-selection-graph15-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    dcc.Download(id="download-selection-graph15")
                                ])
                            ], style={'width': '25%', 'paddingRight': '20px'}),
                            
                            # Right panel
                            html.Div([
                                dcc.Graph(
                                    id='scatter-graph15', 
                                    config={
                                        'displayModeBar': True,
                                        'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                                    }, 
                                    style={'height': '600px'}
                                ),
                                html.Div(id='debug-output-graph15', style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'}),
                                html.Div(id='quality-metrics-graph15', children=[], 
                                    style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #ddd', 
                                           'borderRadius': '5px', 'backgroundColor': '#f9f9f9'}),
                                html.Div(id='selected-points-info-graph15', style={"marginTop": "15px", "fontSize": "12px"})
                            ], style={'width': '75%'})
                        ], style={'display': 'flex'})
                    ], className='container')
                ])
            ])
        ]),
        
        # Tab 3: Selection and Filtering
        dcc.Tab(label='Selection & Filtering', value='tab-selection', children=[
            dcc.Tabs(id='selection-sub-tabs', value='selection-view-tab', children=[
                # Selection Viewing Tab
                dcc.Tab(label='View Selections', value='selection-view-tab', children=[
                    html.Div([
                        # Graph 2: Selected Points from UMAP
                        html.Div([
                            html.H3("Selected Points from UMAP", style={'textAlign': 'center'}),
                            html.Div([
                                html.Div([
                                    html.Label("Select points on UMAP visualization using the lasso or box tool, then click below:"),
                                    html.Button("Show Selected Points", id="show-selected", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="show-selected-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    html.Div(id="selected-points-info", style={"marginTop": "15px", "fontSize": "12px"}),
                                    html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                    html.Label("Save selected points to file:"),
                                    html.Div([
                                        dcc.Input(
                                            id="selection-filename", 
                                            type="text", 
                                            placeholder="Enter filename (without extension)",
                                            style={"width": "100%", "marginBottom": "10px"}
                                        ),
                                        html.Button("Save Selection", id="save-selection-btn", n_clicks=0, className="btn-secondary"),
                                        html.Div(id="save-selection-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                        dcc.Download(id="download-selection")
                                    ])
                                ], style={'width': '25%', 'paddingRight': '20px'}),
                                
                                html.Div([
                                    dcc.Graph(id='umap-graph-selected-only', config={'displayModeBar': True}, style={'height': '600px'}),
                                    html.Div(id='debug-output-selected-only', style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'})
                                ], style={'width': '75%'})
                            ], style={'display': 'flex'})
                        ], className='container'),
                        
                        # Graph 2.5: Selected Points from Custom Scatter
                        html.Div([
                            html.H3("Selected Points from Custom Feature Plot", style={'textAlign': 'center'}),
                            html.Div([
                                html.Div([
                                    html.Label("Select points on Custom Feature Plot using the lasso or box tool, then click below:"),
                                    html.Button("Show Selected Points", id="show-selected-graph15", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="show-selected-graph15-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    html.Div(id="selected-points-info-graph25", style={"marginTop": "15px", "fontSize": "12px"}),
                                    html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                    html.Label("Save selected points to file:"),
                                    html.Div([
                                        dcc.Input(
                                            id="selection-filename-graph25", 
                                            type="text", 
                                            placeholder="Enter filename (without extension)",
                                            style={"width": "100%", "marginBottom": "10px"}
                                        ),
                                        html.Button("Save Selection", id="save-selection-graph25-btn", n_clicks=0, className="btn-secondary"),
                                        html.Div(id="save-selection-graph25-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                        dcc.Download(id="download-selection-graph25")
                                    ])
                                ], style={'width': '25%', 'paddingRight': '20px'}),
                                
                                html.Div([
                                    dcc.Graph(id='graph25', config={'displayModeBar': True}, style={'height': '600px'}),
                                    html.Div(id='debug-output-graph25', style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'})
                                ], style={'width': '75%'})
                            ], style={'display': 'flex'})
                        ], className='container')
                    ])
                ]),
                
                # Filtering Tab
                dcc.Tab(label='Data Filtering', value='filtering-tab', children=[
                    html.Div([
                        # UMAP Filtering Section
                        html.Div([
                            html.H3("UMAP Filtering Options", style={'textAlign': 'center'}),
                            html.Div([
                                # Left panel with controls
                                html.Div([
                                    # Density-based filtering section
                                    html.Div([
                                        html.H4("Density-Based UMAP Filtering", style={'marginBottom': '10px'}),
                                        html.Label("Density Calculation Bandwidth:"),
                                        dcc.Slider(
                                            id='umap-density-bandwidth-slider',
                                            min=0.01,
                                            max=1.0,
                                            step=0.01,
                                            value=0.1,
                                            marks={0.1: '0.1', 0.3: '0.3', 0.5: '0.5', 0.7: '0.7', 0.9: '0.9'},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                        html.Br(),
                                        html.Label("Density Threshold Percentile:"),
                                        dcc.Slider(
                                            id='umap-density-threshold-slider',
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=50,
                                            marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                        html.Br(),
                                        html.Button("Apply UMAP Density Filter", id='apply-umap-density-filter', className="btn-secondary"),
                                        html.Div(id="umap-density-filter-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                        html.Br(),
                                        html.Div(id="umap-density-filter-info", style={"fontSize": "12px"})
                                    ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '15px'}),
                                    
                                    # Physics-based filtering section
                                    html.Div([
                                        html.H4("Physics Parameter UMAP Filtering", style={'marginBottom': '10px'}),
                                        html.Div(id='umap-physics-filter-container', children=[
                                            html.Label("Select Physics Parameter:"),
                                            dcc.Dropdown(
                                                id='umap-physics-parameter-dropdown',
                                                options=[
                                                    {'label': 'KER (Kinetic Energy Release)', 'value': 'KER'},
                                                    {'label': 'EESum (Sum of Electron Energies)', 'value': 'EESum'},
                                                    {'label': 'Total Energy', 'value': 'TotalEnergy'},
                                                    {'label': 'Energy Ion 1', 'value': 'energy_ion1'},
                                                    {'label': 'Energy Ion 2', 'value': 'energy_ion2'},
                                                    {'label': 'Energy Electron 1', 'value': 'energy_electron1'},
                                                    {'label': 'Energy Electron 2', 'value': 'energy_electron2'},
                                                    {'label': 'Ion-Ion Angle', 'value': 'angle_ion1_ion2'}
                                                ],
                                                value=None,
                                                placeholder="Select parameter to filter"
                                            ),
                                            html.Br(),
                                            html.Div(id='umap-parameter-filter-controls', style={'display': 'none'}, children=[
                                                html.Label("Parameter Range:"),
                                                dcc.RangeSlider(
                                                    id='umap-parameter-range-slider',
                                                    min=0,
                                                    max=100,
                                                    step=0.1,
                                                    value=[0, 100],
                                                    marks={0: '0', 100: '100'},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                                html.Br(),
                                                html.Button("Apply UMAP Parameter Filter", id='apply-umap-parameter-filter', className="btn-secondary"),
                                                html.Div(id="umap-parameter-filter-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"})
                                            ])
                                        ]),
                                    ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '15px'}),
                                    
                                    # Combined filtering results section
                                    html.Div([
                                        html.H4("Filtered UMAP Results", style={'marginBottom': '10px'}),
                                        html.Div(id="umap-filtered-data-info", style={"fontSize": "12px"}),
                                        html.Br(),
                                        html.Label("Save filtered UMAP points:"),
                                        html.Div([
                                            dcc.Input(
                                                id="umap-filtered-data-filename", 
                                                type="text", 
                                                placeholder="Enter filename (without extension)",
                                                style={"width": "100%", "marginBottom": "10px"}
                                            ),
                                            html.Button("Save Filtered UMAP Data", id="save-umap-filtered-data-btn", className="btn-secondary"),
                                            html.Div(id="save-umap-filtered-data-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                            dcc.Download(id="download-umap-filtered-data")
                                        ])
                                    ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                                ], style={'width': '25%', 'paddingRight': '20px'}),
                                
                                # Right panel with visualization
                                html.Div([
                                    dcc.Graph(
                                        id='umap-filtered-data-graph', 
                                        config={'displayModeBar': True}, 
                                        style={'height': '600px'}
                                    ),
                                    html.Div(id='umap-filtered-data-debug', style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'})
                                ], style={'width': '75%'})
                            ], style={'display': 'flex'})
                        ], className='container'),
                        
                        # Custom Feature Filtering Section
                        html.Div([
                            html.H3("Custom Feature Filtering Options", style={'textAlign': 'center'}),
                            html.Div([
                                # Left panel with controls
                                html.Div([
                                    # Density-based filtering section
                                    html.Div([
                                        html.H4("Density-Based Filtering", style={'marginBottom': '10px'}),
                                        html.Label("Density Calculation Bandwidth:"),
                                        dcc.Slider(
                                            id='density-bandwidth-slider',
                                            min=0.01,
                                            max=1.0,
                                            step=0.01,
                                            value=0.1,
                                            marks={0.1: '0.1', 0.3: '0.3', 0.5: '0.5', 0.7: '0.7', 0.9: '0.9'}, 
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                        html.Br(),
                                        html.Label("Density Threshold Percentile:"),
                                        dcc.Slider(
                                            id='density-threshold-slider',
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=50,
                                            marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                        html.Br(),
                                        html.Button("Apply Density Filter", id='apply-density-filter', className="btn-secondary"),
                                        html.Div(id="density-filter-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                        html.Br(),
                                        html.Div(id="density-filter-info", style={"fontSize": "12px"})
                                    ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '15px'}),
                                    
                                    # Physics-based filtering section
                                    html.Div([
                                        html.H4("Physics Parameter Filtering", style={'marginBottom': '10px'}),
                                        html.Div(id='physics-filter-container', children=[
                                            html.Label("Select Physics Parameter:"),
                                            dcc.Dropdown(
                                                id='physics-parameter-dropdown',
                                                options=[
                                                    {'label': 'KER (Kinetic Energy Release)', 'value': 'KER'},
                                                    {'label': 'EESum (Sum of Electron Energies)', 'value': 'EESum'},
                                                    {'label': 'Total Energy', 'value': 'TotalEnergy'},
                                                    {'label': 'Energy Ion 1', 'value': 'energy_ion1'},
                                                    {'label': 'Energy Ion 2', 'value': 'energy_ion2'},
                                                    {'label': 'Energy Electron 1', 'value': 'energy_electron1'},
                                                    {'label': 'Energy Electron 2', 'value': 'energy_electron2'},
                                                    {'label': 'Ion-Ion Angle', 'value': 'angle_ion1_ion2'}
                                                ],
                                                value=None,
                                                placeholder="Select parameter to filter"
                                            ),
                                            html.Br(),
                                            html.Div(id='parameter-filter-controls', style={'display': 'none'}, children=[
                                                html.Label("Parameter Range:"),
                                                dcc.RangeSlider(
                                                    id='parameter-range-slider',
                                                    min=0,
                                                    max=100,
                                                    step=0.1,
                                                    value=[0, 100],
                                                    marks={0: '0', 100: '100'},
                                                    tooltip={"placement": "bottom", "always_visible": True}
                                                ),
                                                html.Br(),
                                                html.Button("Apply Parameter Filter", id='apply-parameter-filter', className="btn-secondary"),
                                                html.Div(id="parameter-filter-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"})
                                            ])
                                        ]),
                                    ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '15px'}),
                                    
                                    # Combined filtering results section
                                    html.Div([
                                        html.H4("Filtered Data Results", style={'marginBottom': '10px'}),
                                        html.Div(id="filtered-data-info", style={"fontSize": "12px"}),
                                        html.Br(),
                                        html.Label("Save filtered points:"),
                                        html.Div([
                                            dcc.Input(
                                                id="filtered-data-filename", 
                                                type="text", 
                                                placeholder="Enter filename (without extension)",
                                                style={"width": "100%", "marginBottom": "10px"}
                                            ),
                                            html.Button("Save Filtered Data", id="save-filtered-data-btn", className="btn-secondary"),
                                            html.Div(id="save-filtered-data-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                            dcc.Download(id="download-filtered-data")
                                        ])
                                    ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                                ], style={'width': '25%', 'paddingRight': '20px'}),
                                
                                # Right panel with visualization
                                html.Div([
                                    dcc.Graph(
                                        id='filtered-data-graph', 
                                        config={'displayModeBar': True}, 
                                        style={'height': '600px'}
                                    ),
                                    html.Div(id='filtered-data-debug', style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'})
                                ], style={'width': '75%'})
                            ], style={'display': 'flex'})
                        ], className='container')
                    ])
                ])
            ])
        ]),
        
        # Tab 4: Advanced Analysis
        dcc.Tab(label='Advanced Analysis', value='tab-advanced', children=[
            dcc.Tabs(id='advanced-sub-tabs', value='rerun-umap-tab', children=[
                # Re-run UMAP Tab
                dcc.Tab(label='Re-run UMAP', value='rerun-umap-tab', children=[
                    html.Div([
                        # Graph 3: Re-run UMAP on Selected Points
                        html.Div([
                            html.H3("Graph 3: Re-run UMAP on Selected Points from Graph 1", style={'textAlign': 'center'}),
                            html.Div([
                                html.Div([
                                    html.H4("Select Features for Re-run UMAP:"),
                                    html.Div(id='feature-selection-ui-graph3', children=[
                                        html.Div("Upload files to see available features", style={"color": "gray"})
                                    ], className='feature-checklist'),
                                    html.Br(),
                                    html.Label("UMAP Neighbors (Selected Re-run):"),
                                    dcc.Input(id="num-neighbors-selected-run", type="number", value=15, min=1),
                                    html.Br(),
                                    html.Label("Min Dist (Selected Re-run):"),
                                    dcc.Input(id="min-dist-selected-run", type="number", value=0.1, step=0.01, min=0),
                                    html.Br(),
                                    html.Button("Re-run UMAP on Selected Points", id="run-umap-selected-run", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="run-umap-selected-run-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    html.Br(),
                                    html.Label("Select Metrics to Calculate:"),
                                    dcc.Checklist(
                                        id='metric-selector-graph3',
                                        options=[
                                            {'label': 'Silhouette Score', 'value': 'silhouette'},
                                            {'label': 'Davies-Bouldin Index', 'value': 'davies_bouldin'},
                                            {'label': 'Calinski-Harabasz Index', 'value': 'calinski_harabasz'},
                                            {'label': 'Hopkins Statistic', 'value': 'hopkins'},
                                            {'label': 'Cluster Stability', 'value': 'stability'},
                                            {'label': 'Physics Consistency', 'value': 'physics_consistency'}
                                        ],
                                        value=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
                                        labelStyle={'display': 'block'}
                                    ),
                                    html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                    html.Label("Select points in Graph 3 and save:"),
                                    html.Button("Show Graph 3 Selection", id="show-selected-run", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="show-selected-run-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    html.Div(id="selected-points-run-info", style={"marginTop": "10px", "fontSize": "12px"}),
                                    html.Hr(style={"marginTop": "15px", "marginBottom": "15px"}),
                                    html.Label("Save Graph 3 selected points:"),
                                    html.Div([
                                        dcc.Input(
                                            id="selection-run-filename", 
                                            type="text", 
                                            placeholder="Enter filename (without extension)",
                                            style={"width": "100%", "marginBottom": "10px"}
                                        ),
                                        html.Button("Save Graph 3 Selection", id="save-selection-run-btn", n_clicks=0, className="btn-secondary"),
                                        html.Div(id="save-selection-run-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                        dcc.Download(id="download-selection-run")
                                    ])
                                ], style={'width': '25%', 'paddingRight': '20px'}),
                                
                                html.Div([
                                    dcc.Graph(id='umap-graph-selected-run', config={'displayModeBar': True}, style={'height': '600px'}),
                                    html.Div(id='debug-output-selected-run', style={'marginTop': '10px', "fontSize": "12px", "color": "gray"}),
                                    html.Div(id='umap-quality-metrics-graph3', children=[], 
                                        style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #ddd', 
                                               'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
                                ], style={'width': '75%'})
                            ], style={'display': 'flex'})
                        ], className='container'),
                        
                        # Graph 3 Selected Points Visualization
                        html.Div([
                            html.H3("Graph 3 Selected Points Visualization", style={'textAlign': 'center'}),
                            html.Div([
                                html.Div([
                                    html.Label("This shows points selected from Graph 3."),
                                    html.Div(id="graph3-selection-info-viz", style={"marginTop": "15px", "fontSize": "12px"})
                                ], style={'width': '25%', 'paddingRight': '20px'}),
                                
                                html.Div([
                                    dcc.Graph(id='umap-graph-selected-run-only', config={'displayModeBar': True}, style={'height': '600px'}),
                                    html.Div(id='debug-output-selected-run-only', style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'})
                                ], style={'width': '75%'})
                            ], style={'display': 'flex'})
                        ], className='container'),
                        
                        # UMAP on Graph 3 Selected Points
                        html.Div([
                            html.H3("UMAP Re-run on Graph 3 Selected Points", style={'textAlign': 'center'}),
                            html.Div([
                                html.Div([
                                    html.H4("Select Features for Re-run UMAP on Graph 3 Selection:"),
                                    html.Div(id='feature-selection-ui-graph3-selection', children=[
                                        html.Div("Upload files to see available features", style={"color": "gray"})
                                    ], className='feature-checklist'),
                                    html.Br(),
                                    html.Label("UMAP Neighbors:"),
                                    dcc.Input(id="num-neighbors-graph3-selection", type="number", value=15, min=1),
                                    html.Br(),
                                    html.Label("Min Dist:"),
                                    dcc.Input(id="min-dist-graph3-selection", type="number", value=0.1, step=0.01, min=0),
                                    html.Br(),
                                    html.Button("Run UMAP on Graph 3 Selection", id="run-umap-graph3-selection", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="run-umap-graph3-selection-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    html.Br(),
                                    html.Label("Select Metrics to Calculate:"),
                                    dcc.Checklist(
                                        id='metric-selector-graph3-selection',
                                        options=[
                                            {'label': 'Silhouette Score', 'value': 'silhouette'},
                                            {'label': 'Davies-Bouldin Index', 'value': 'davies_bouldin'},
                                            {'label': 'Calinski-Harabasz Index', 'value': 'calinski_harabasz'},
                                            {'label': 'Hopkins Statistic', 'value': 'hopkins'},
                                            {'label': 'Cluster Stability', 'value': 'stability'},
                                            {'label': 'Physics Consistency', 'value': 'physics_consistency'}
                                        ],
                                        value=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
                                        labelStyle={'display': 'block'}
                                    ),
                                    html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                    html.Label("Save selection:"),
                                    html.Div([
                                        dcc.Input(
                                            id="selection-graph3-selection-filename", 
                                            type="text", 
                                            placeholder="Enter filename (without extension)",
                                            style={"width": "100%", "marginBottom": "10px"}
                                        ),
                                        html.Button("Save Selection", id="save-selection-graph3-selection-btn", n_clicks=0, className="btn-secondary"),
                                        html.Div(id="save-selection-graph3-selection-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                        dcc.Download(id="download-selection-graph3-selection")
                                    ])
                                ], style={'width': '25%', 'paddingRight': '20px'}),
                                
                                html.Div([
                                    dcc.Graph(id='umap-graph-graph3-selection', config={'displayModeBar': True}, style={'height': '600px'}),
                                    html.Div(id='debug-output-graph3-selection', style={'marginTop': '10px', "fontSize": "12px", "color": "gray"}),
                                    html.Div(id='umap-quality-metrics-graph3-selection', children=[], 
                                        style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #ddd', 
                                               'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
                                ], style={'width': '75%'})
                            ], style={'display': 'flex'})
                        ], className='container')
                    ])
                ]),
                
                # Custom Feature Plot Tab
                dcc.Tab(label='Custom Feature Analysis', value='custom-feature-tab', children=[
                    html.Div([
                        html.H3("Custom Feature Scatter Plot", style={'textAlign': 'center'}),
                        html.Div([
                            html.Div([
                                html.H4("Select Features to Plot:"),
                                html.Div([
                                    html.Label("X-Axis Feature:"),
                                    dcc.Dropdown(
                                        id='x-axis-feature',
                                        options=[],
                                        value=None,
                                        placeholder="Select X-Axis Feature"
                                    ),
                                    html.Br(),
                                    html.Label("Y-Axis Feature:"),
                                    dcc.Dropdown(
                                        id='y-axis-feature',
                                        options=[],
                                        value=None,
                                        placeholder="Select Y-Axis Feature"
                                    ),
                                    html.Br(),
                                    html.Label("Display Selection From:"),
                                    dcc.RadioItems(
                                        id='selection-source',
                                        options=[
                                            {'label': 'Graph 2 (Selection from Graph 1)', 'value': 'graph2'},
                                            {'label': 'Graph 3 Selection', 'value': 'graph3'},
                                            {'label': 'Both Selections', 'value': 'both'}
                                        ],
                                        value='graph2',
                                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                                    ),
                                    html.Br(),
                                    html.Button(
                                        "Plot Selected Features", 
                                        id="plot-custom-features", 
                                        n_clicks=0, 
                                        className="btn-secondary"
                                    ),
                                    html.Div(
                                        id="plot-custom-features-status", 
                                        style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}
                                    )
                                ], className='feature-checklist'),
                            ], style={'width': '25%', 'paddingRight': '20px'}),
                            
                            html.Div([
                                dcc.Graph(
                                    id='custom-feature-plot', 
                                    config={'displayModeBar': True}, 
                                    style={'height': '600px'}
                                ),
                                html.Div(
                                    id='debug-output-custom-plot', 
                                    style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'}
                                )
                            ], style={'width': '75%'})
                        ], style={'display': 'flex'})
                    ], className='container')
                ])
            ])
        ]),
        
        # Tab 5: Machine Learning
        dcc.Tab(label='Machine Learning', value='tab-ml', children=[
            dcc.Tabs(id='ml-sub-tabs', value='autoencoder-tab', children=[
                # Autoencoder Tab
                dcc.Tab(label='Deep Autoencoder', value='autoencoder-tab', children=[
                    html.Div([
                        html.H3("Deep Autoencoder with UMAP on Latent Space", style={'textAlign': 'center'}),
                        html.Div([
                            html.Div([
                                html.H4("Select Features for Autoencoder:"),
                                html.Div(id='feature-selection-ui-autoencoder', children=[
                                    html.Div("Upload files to see available features", style={"color": "gray"})
                                ], className='feature-checklist'),
                                html.Br(),
                                html.Label("Latent Dimension Size:"),
                                dcc.Input(id="autoencoder-latent-dim", type="number", value=7, min=2, max=20),
                                html.Br(),
                                html.Label("Number of Epochs:"),
                                dcc.Input(id="autoencoder-epochs", type="number", value=50, min=10, max=500),
                                html.Br(),
                                html.Label("Batch Size:"),
                                dcc.Input(id="autoencoder-batch-size", type="number", value=64, min=8, max=512),
                                html.Br(),
                                html.Label("Learning Rate:"),
                                dcc.Input(id="autoencoder-learning-rate", type="number", value=0.001, min=0.0001, max=0.1, step=0.0001),
                                html.Br(),
                                html.Div([
                                    html.Label("Select Data Source:"),
                                    dcc.RadioItems(
                                        id='autoencoder-data-source',
                                        options=[
                                            {'label': 'All Data', 'value': 'all'},
                                            {'label': 'Graph 1 Selection', 'value': 'graph1-selection'},
                                            {'label': 'Graph 3 Selection', 'value': 'graph3-selection'}
                                        ],
                                        value='all',
                                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                                    ),
                                ]),
                                html.Br(),
                                html.Button("Train Autoencoder", id="train-autoencoder", n_clicks=0, className="btn-secondary"),
                                html.Div(id="train-autoencoder-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                html.Div(id="training-progress", style={"marginTop": "5px", "fontSize": "12px", "color": "green"}),
                                html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                html.Label("UMAP on Latent Space:"),
                                html.Div([
                                    html.Label("UMAP Neighbors:"),
                                    dcc.Input(id="autoencoder-umap-neighbors", type="number", value=15, min=1),
                                    html.Br(),
                                    html.Label("Min Dist:"),
                                    dcc.Input(id="autoencoder-umap-min-dist", type="number", value=0.1, step=0.01, min=0),
                                    html.Br(),
                                    html.Button("Run UMAP on Latent Space", id="run-umap-latent", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="run-umap-latent-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    html.Br(),
                                    html.Label("Select Metrics to Calculate:"),
                                    dcc.Checklist(
                                        id='metric-selector-autoencoder',
                                        options=[
                                            {'label': 'Silhouette Score', 'value': 'silhouette'},
                                            {'label': 'Davies-Bouldin Index', 'value': 'davies_bouldin'},
                                            {'label': 'Calinski-Harabasz Index', 'value': 'calinski_harabasz'},
                                            {'label': 'Hopkins Statistic', 'value': 'hopkins'},
                                            {'label': 'Cluster Stability', 'value': 'stability'}
                                        ],
                                        value=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
                                        labelStyle={'display': 'block'}
                                    ),
                                ]),
                                html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                html.Label("Save Latent Features:"),
                                html.Div([
                                    dcc.Input(
                                        id="latent-features-filename", 
                                        type="text", 
                                        placeholder="Enter filename (without extension)",
                                        style={"width": "100%", "marginBottom": "10px"}
                                    ),
                                    html.Button("Save Latent Features", id="save-latent-features-btn", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="save-latent-features-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    dcc.Download(id="download-latent-features")
                                ])
                            ], style={'width': '25%', 'paddingRight': '20px'}),
                            
                            html.Div([
                                dcc.Graph(id='autoencoder-umap-graph', config={'displayModeBar': True}, style={'height': '600px'}),
                                html.Div(id='autoencoder-debug-output', style={'marginTop': '10px', "fontSize": "12px", "color": "gray"}),
                                html.Div(id='umap-quality-metrics-autoencoder', children=[], 
                                    style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #ddd', 
                                           'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
                            ], style={'width': '75%'}),
                            html.Div(id='feature-importance-container', 
                                children=[], 
                                style={"marginTop": "15px"})
                        ], style={'display': 'flex'})
                    ], className='container')
                ]),
                
                # Genetic Features Tab
                dcc.Tab(label='Genetic Feature Engineering', value='genetic-tab', children=[
                    html.Div([
                        html.H3("Genetic Feature Engineering", style={'textAlign': 'center'}),
                        html.Div([
                            html.Div([
                                html.H4("Select Features for Genetic Programming:"),
                                html.Div(id='feature-selection-ui-genetic', children=[
                                    html.Div("Upload files to see available features", style={"color": "gray"})
                                ], className='feature-checklist'),
                                html.Br(),
                                html.H4("Clustering Parameters:"),
                                html.Label("Clustering Method:"),
                                dcc.Dropdown(
                                    id='clustering-method',
                                    options=[
                                        {'label': 'DBSCAN', 'value': 'dbscan'},
                                        {'label': 'KMeans', 'value': 'kmeans'},
                                        {'label': 'Agglomerative', 'value': 'agglomerative'}
                                    ],
                                    value='dbscan'
                                ),
                                html.Div(id='dbscan-params', children=[
                                    html.Label("DBSCAN Epsilon:"),
                                    dcc.Input(id="dbscan-eps", type="number", value=0.3, min=0.01, max=2.0, step=0.05),
                                    html.Br(),
                                    html.Label("DBSCAN Min Samples:"),
                                    dcc.Input(id="dbscan-min-samples", type="number", value=5, min=2, max=50)
                                ]),
                                html.Div(id='kmeans-params', children=[
                                    html.Label("K-Means Clusters:"),
                                    dcc.Input(id="kmeans-n-clusters", type="number", value=5, min=2, max=20)
                                ], style={'display': 'none'}),
                                html.Div(id='agglomerative-params', children=[
                                    html.Label("Agglomerative Clusters:"),
                                    dcc.Input(id="agglomerative-n-clusters", type="number", value=5, min=2, max=20),
                                    html.Br(),
                                    html.Label("Linkage:"),
                                    dcc.Dropdown(
                                        id='agglomerative-linkage',
                                        options=[
                                            {'label': 'Ward', 'value': 'ward'},
                                            {'label': 'Complete', 'value': 'complete'},
                                            {'label': 'Average', 'value': 'average'},
                                            {'label': 'Single', 'value': 'single'}
                                        ],
                                        value='ward'
                                    )
                                ], style={'display': 'none'}),
                                html.Br(),
                                html.H4("Genetic Programming Parameters:"),
                                html.Label("Number of Generations:"),
                                dcc.Input(id="gp-generations", type="number", value=20, min=5, max=100),
                                html.Br(),
                                html.Label("Population Size:"),
                                dcc.Input(id="gp-population-size", type="number", value=1000, min=500, max=5000, step=100),
                                html.Br(),
                                html.Label("Number of Features to Generate:"),
                                dcc.Input(id="gp-n-components", type="number", value=10, min=2, max=20),
                                html.Br(),
                                html.Label("Mathematical Functions:"),
                                dcc.Checklist(
                                    id='gp-functions',
                                    options=[
                                        {'label': 'Basic (add, sub, mul, div)', 'value': 'basic'},
                                        {'label': 'Trigonometric (sin, cos, tan)', 'value': 'trig'},
                                        {'label': 'Exponential & Logarithmic', 'value': 'exp_log'},
                                        {'label': 'Square Root & Power', 'value': 'sqrt_pow'},
                                        {'label': 'Special (abs, inv)', 'value': 'special'}
                                    ],
                                    value=['basic', 'trig', 'sqrt_pow', 'special']
                                ),
                                html.Br(),
                                html.Div([
                                    html.Label("Select Data Source:"),
                                    dcc.RadioItems(
                                        id='genetic-data-source',
                                        options=[
                                            {'label': 'All Data', 'value': 'all'},
                                            {'label': 'Graph 1 Selection', 'value': 'graph1-selection'},
                                            {'label': 'Graph 3 Selection', 'value': 'graph3-selection'},
                                            {'label': 'Autoencoder Latent Space', 'value': 'autoencoder-latent'}
                                        ],
                                        value='all',
                                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                                    ),
                                ]),
                                html.Br(),
                                html.Button("Run Genetic Feature Discovery", id="run-genetic-features", n_clicks=0, className="btn-secondary"),
                                html.Div(id="genetic-features-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                html.Br(),
                                html.Div([
                                    html.H4("Visualize Genetic Features:", style={"marginTop": "15px", "color": "#2e7d32"}),
                                    html.Div("First run genetic discovery above, then select specific features to visualize with UMAP:", 
                                           style={"color": "#555", "fontSize": "12px", "marginBottom": "10px"}),
                                    html.Button(
                                        "Run UMAP on Genetic Features", 
                                        id="run-umap-genetic", 
                                        n_clicks=0, 
                                        className="btn-secondary",
                                        style={"backgroundColor": "#2e7d32", "marginTop": "5px"}
                                    ),
                                    html.Div(id="run-umap-genetic-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    html.Br(),
                                    html.Label("Select Metrics to Calculate:"),
                                    dcc.Checklist(
                                        id='metric-selector-genetic',
                                        options=[
                                            {'label': 'Silhouette Score', 'value': 'silhouette'},
                                            {'label': 'Davies-Bouldin Index', 'value': 'davies_bouldin'},
                                            {'label': 'Calinski-Harabasz Index', 'value': 'calinski_harabasz'},
                                            {'label': 'Hopkins Statistic', 'value': 'hopkins'},
                                            {'label': 'Cluster Stability', 'value': 'stability'}
                                        ],
                                        value=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
                                        labelStyle={'display': 'block'}
                                    ),
                                ], style={"border": "1px solid #c8e6c9", "padding": "10px", "borderRadius": "5px", "backgroundColor": "#f1f8e9"}),
                                html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                html.Label("Save Discovered Features:"),
                                html.Div([
                                    dcc.Input(
                                        id="genetic-features-filename", 
                                        type="text", 
                                        placeholder="Enter filename (without extension)",
                                        style={"width": "100%", "marginBottom": "10px"}
                                    ),
                                    html.Button("Save Genetic Features", id="save-genetic-features-btn", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="save-genetic-features-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    dcc.Download(id="download-genetic-features")
                                ])
                            ], style={'width': '25%', 'paddingRight': '20px'}),
                            
                            html.Div([
                                dcc.Graph(id='genetic-features-graph', config={'displayModeBar': True}, style={'height': '600px'}),
                                html.Div(id='genetic-features-debug-output', style={'marginTop': '10px', "fontSize": "12px", "color": "gray"}),
                                html.Div(id='umap-quality-metrics-genetic', children=[], 
                                    style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #ddd', 
                                           'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
                            ], style={'width': '75%'})
                        ], style={'display': 'flex'})
                    ], className='container')
                ]),
                
                # Mutual Information Tab
                dcc.Tab(label='Mutual Information Feature Selection', value='mi-tab', children=[
                    html.Div([
                        html.H3("Mutual Information Feature Selection", style={'textAlign': 'center'}),
                        html.Div([
                            html.Div([
                                html.H4("Select Features for MI Analysis:"),
                                html.Div(id='feature-selection-ui-mi', children=[
                                    html.Div("Upload files to see available features", style={"color": "gray"})
                                ], className='feature-checklist'),
                                html.Br(),
                                html.H4("MI Analysis Parameters:"),
                                html.Label("Target Variables:"),
                                dcc.Dropdown(
                                    id='mi-target-variables',
                                    options=[
                                        {'label': 'KER (Kinetic Energy Release)', 'value': 'KER'},
                                        {'label': 'EESum (Sum of Electron Energies)', 'value': 'EESum'},
                                        {'label': 'Total Energy', 'value': 'TotalEnergy'},
                                        {'label': 'Energy Ion 1', 'value': 'energy_ion1'},
                                        {'label': 'Energy Ion 2', 'value': 'energy_ion2'},
                                        {'label': 'Energy Electron 1', 'value': 'energy_electron1'},
                                        {'label': 'Energy Electron 2', 'value': 'energy_electron2'}
                                    ],
                                    value=['KER', 'EESum', 'TotalEnergy'],
                                    multi=True
                                ),
                                html.Br(),
                                html.Label("Redundancy Threshold (0-1):"),
                                html.Div([
                                    "Low values reduce redundancy, high values allow more similar features",
                                    dcc.Slider(
                                        id='mi-redundancy-threshold',
                                        min=0.1,
                                        max=0.9,
                                        step=0.1,
                                        value=0.5,
                                        marks={i/10: f'{i/10}' for i in range(1, 10)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ]),
                                html.Br(),
                                html.Label("Maximum Number of Features:"),
                                dcc.Input(id="mi-max-features", type="number", value=20, min=5, max=100),
                                html.Br(),
                                html.Div([
                                    html.Label("Select Data Source:"),
                                    dcc.RadioItems(
                                        id='mi-data-source',
                                        options=[
                                            {'label': 'All Data', 'value': 'all'},
                                            {'label': 'Graph 1 Selection', 'value': 'graph1-selection'},
                                            {'label': 'Graph 3 Selection', 'value': 'graph3-selection'}
                                        ],
                                        value='all',
                                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                                    ),
                                ]),
                                html.Br(),
                                html.H4("Autoencoder Parameters:"),
                                html.Label("Latent Dimension Size:"),
                                dcc.Input(id="mi-latent-dim", type="number", value=7, min=2, max=20),
                                html.Br(),
                                html.Label("Number of Epochs:"),
                                dcc.Input(id="mi-epochs", type="number", value=100, min=10, max=500),
                                html.Br(),
                                html.Label("Batch Size:"),
                                dcc.Input(id="mi-batch-size", type="number", value=64, min=8, max=512),
                                html.Br(),
                                html.Label("Learning Rate:"),
                                dcc.Input(id="mi-learning-rate", type="number", value=0.001, min=0.0001, max=0.1, step=0.0001),
                                html.Br(),
                                html.Button("Run MI Feature Selection", id="run-mi-features", n_clicks=0, className="btn-secondary"),
                                html.Div(id="mi-features-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                html.Br(),
                                html.Div([
                                    html.H4("Visualize MI Features:", style={"marginTop": "15px", "color": "#2e7d32"}),
                                    html.Div("First run MI feature selection above, then visualize with UMAP:", 
                                           style={"color": "#555", "fontSize": "12px", "marginBottom": "10px"}),
                                    html.Button(
                                        "Run UMAP on MI Features", 
                                        id="run-umap-mi", 
                                        n_clicks=0, 
                                        className="btn-secondary",
                                        style={"backgroundColor": "#2e7d32", "marginTop": "5px"}
                                    ),
                                    html.Div(id="run-umap-mi-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    html.Br(),
                                    html.Label("Select Metrics to Calculate:"),
                                    dcc.Checklist(
                                        id='metric-selector-mi',
                                        options=[
                                            {'label': 'Silhouette Score', 'value': 'silhouette'},
                                            {'label': 'Davies-Bouldin Index', 'value': 'davies_bouldin'},
                                            {'label': 'Calinski-Harabasz Index', 'value': 'calinski_harabasz'},
                                            {'label': 'Hopkins Statistic', 'value': 'hopkins'},
                                            {'label': 'Cluster Stability', 'value': 'stability'}
                                        ],
                                        value=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
                                        labelStyle={'display': 'block'}
                                    ),
                                ], style={"border": "1px solid #c8e6c9", "padding": "10px", "borderRadius": "5px", "backgroundColor": "#f1f8e9"}),
                                html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                                html.Label("Save MI Features:"),
                                html.Div([
                                    dcc.Input(
                                        id="mi-features-filename", 
                                        type="text", 
                                        placeholder="Enter filename (without extension)",
                                        style={"width": "100%", "marginBottom": "10px"}
                                    ),
                                    html.Button("Save MI Features", id="save-mi-features-btn", n_clicks=0, className="btn-secondary"),
                                    html.Div(id="save-mi-features-status", style={"marginTop": "5px", "fontSize": "12px", "color": "blue"}),
                                    dcc.Download(id="download-mi-features")
                                ])
                            ], style={'width': '25%', 'paddingRight': '20px'}),
                            
                            html.Div([
                                dcc.Graph(id='mi-features-graph', config={'displayModeBar': True}, style={'height': '600px'}),
                                html.Div(id='mi-features-debug-output', style={'marginTop': '10px', "fontSize": "12px", "color": "gray"}),
                                html.Div(id='umap-quality-metrics-mi', children=[], 
                                    style={'marginTop': '10px', 'padding': '10px', 'border': '1px solid #ddd', 
                                           'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
                            ], style={'width': '75%'})
                        ], style={'display': 'flex'})
                    ], className='container')
                ])
            ])
        ])
    ]),
    
    # Hidden stores - keep all your existing stores
    dcc.Store(id='stored-files', data=[]),
    dcc.Store(id='combined-data-store', data=""),
    dcc.Store(id='features-data-store', data={}),
    dcc.Store(id='selected-points-store', data=[]),
    dcc.Store(id='selected-points-run-store', data=[]),
    dcc.Store(id='graph3-selection-umap-store', data=[]),
    dcc.Store(id='autoencoder-latent-store', data=[]),
    dcc.Interval(id='training-interval', interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id='genetic-features-store', data=[]),
    dcc.Store(id='mi-features-store', data=[]),
    dcc.Store(id='selected-points-store-graph15', data=[]),
    dcc.Store(id='filtered-data-store', data={}),
    dcc.Store(id='umap-filtered-data-store', data={}),
    dcc.Store(id='configuration-profiles-store', data={}),
    dcc.Store(id='file-config-assignments-store', data={}),
    dcc.Store(id='current-profile-store', data=None),
    dcc.Store(id='particle-count-store', data={'ions': 2, 'neutrals': 1, 'electrons': 2}),
])
    
def is_selection_file(df):
    """Check if the dataframe looks like a saved selection."""
    # Check for UMAP coordinates which would indicate a selection file
    return 'UMAP1' in df.columns and 'UMAP2' in df.columns and 'file_label' in df.columns

def parse_contents(contents, filename):
    """Parse the uploaded file contents."""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        print(f"Decoding {filename}...")
        
        # Get a sample of the file for debugging
        sample = decoded[:1000].decode('utf-8', errors='replace')
        print(f"Sample of file {filename}: {sample[:100]}...")
        
        # Try to infer the separator (space or comma)
        first_line = decoded.decode('utf-8', errors='replace').split('\n')[0]
        if ',' in first_line:
            sep = ','
            print(f"Detected comma separator for {filename}")
        else:
            sep = ' '
            print(f"Detected space separator for {filename}")
        
        # Try to read with pandas
        print(f"Reading {filename} with separator '{sep}'")
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='replace')), sep=sep)
            print(f"Successfully read {filename}, shape: {df.shape}")
            
            # Check if this is a saved selection file
            if is_selection_file(df):
                print(f"{filename} is a selection file")
                return df, True  # Return df and flag indicating it's a selection file
            
            # Otherwise check if it's a standard COLTRIMS file
            print(f"Checking if {filename} is a COLTRIMS file...")
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: File {filename} is missing columns: {missing_cols}")
                return None, False
            
            print(f"{filename} is a valid COLTRIMS file")
            return df, False  # Return df and flag indicating it's not a selection file
            
        except Exception as e:
            print(f"Error in pd.read_csv for {filename}: {e}")
            # Try alternative parsing approaches
            if sep == ',':
                try:
                    print(f"Trying to read {filename} with space separator as fallback")
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='replace')), sep=' ')
                    print(f"Successfully read {filename} with space separator, shape: {df.shape}")
                    # Check columns as before
                    if is_selection_file(df):
                        return df, True
                    
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    if missing_cols:
                        print(f"Warning: File {filename} is missing columns: {missing_cols}")
                        return None, False
                    
                    return df, False
                except Exception as e2:
                    print(f"Error in fallback parsing for {filename}: {e2}")
            
            # Try with delim_whitespace for fixed-width files
            try:
                print(f"Trying to read {filename} with delim_whitespace=True as second fallback")
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8', errors='replace')), delim_whitespace=True)
                print(f"Successfully read {filename} with delim_whitespace, shape: {df.shape}")
                # Check columns as before
                if is_selection_file(df):
                    return df, True
                
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"Warning: File {filename} is missing columns: {missing_cols}")
                    return None, False
                
                return df, False
            except Exception as e3:
                print(f"Error in second fallback for {filename}: {e3}")
                raise
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None, False

                    
def reorganize_df(df):           
    """Reorganize the dataframe with standardized column names."""
    df = df[required_columns].copy()
    new_columns = []
    for i in range(5):
        new_columns.extend([f'particle_{i}_Px', f'particle_{i}_Py', f'particle_{i}_Pz'])
    new_df = pd.DataFrame(df.values, columns=new_columns)
    return new_df

def calculate_physics_features_flexible(df, config=None):
    """Calculate physics features with flexible particle configuration."""
    try:
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Get particle configuration
        if config and 'particle_count' in config:
            num_ions = config['particle_count'].get('ions', 2)
            num_neutrals = config['particle_count'].get('neutrals', 1)
            num_electrons = config['particle_count'].get('electrons', 2)
        else:
            # Default configuration
            num_ions = 2
            num_neutrals = 1
            num_electrons = 2
        
        total_particles = num_ions + num_neutrals + num_electrons
        
        # Get masses from configuration
        particle_masses = {}
        if config and 'particles' in config:
            particles_config = config['particles']
            
            # Process ions
            for i in range(num_ions):
                mass = particles_config.get(f'ion_{i}', {}).get('mass', 1) * 1836
                particle_masses[f'ion{i+1}'] = mass
            
            # Process neutrals
            for i in range(num_neutrals):
                mass = particles_config.get(f'neutral_{i}', {}).get('mass', 16) * 1836
                particle_masses[f'neutral{i+1}'] = mass
            
            # Electrons always have mass 1
            for i in range(num_electrons):
                particle_masses[f'electron{i+1}'] = 1
        else:
            # Default masses
            for i in range(num_ions):
                particle_masses[f'ion{i+1}'] = 2 * 1836  # Default deuterium
            for i in range(num_neutrals):
                particle_masses[f'neutral{i+1}'] = 16 * 1836  # Default oxygen
            for i in range(num_electrons):
                particle_masses[f'electron{i+1}'] = 1
        
        # Calculate momentum magnitudes for all particles
        particle_idx = 0
        
        # Process ions
        for i in range(num_ions):
            if particle_idx < total_particles:
                p = df[[f'particle_{particle_idx}_Px', 
                       f'particle_{particle_idx}_Py', 
                       f'particle_{particle_idx}_Pz']].to_numpy()
                result_df[f'mom_mag_ion{i+1}'] = np.linalg.norm(p, axis=1)
                result_df[f'energy_ion{i+1}'] = (result_df[f'mom_mag_ion{i+1}']**2) / (2 * particle_masses[f'ion{i+1}'])
                particle_idx += 1
        
        # Process neutrals
        for i in range(num_neutrals):
            if particle_idx < total_particles:
                p = df[[f'particle_{particle_idx}_Px', 
                       f'particle_{particle_idx}_Py', 
                       f'particle_{particle_idx}_Pz']].to_numpy()
                result_df[f'mom_mag_neutral{i+1}'] = np.linalg.norm(p, axis=1)
                result_df[f'energy_neutral{i+1}'] = (result_df[f'mom_mag_neutral{i+1}']**2) / (2 * particle_masses[f'neutral{i+1}'])
                particle_idx += 1
        
        # Process electrons
        for i in range(num_electrons):
            if particle_idx < total_particles:
                p = df[[f'particle_{particle_idx}_Px', 
                       f'particle_{particle_idx}_Py', 
                       f'particle_{particle_idx}_Pz']].to_numpy()
                result_df[f'mom_mag_electron{i+1}'] = np.linalg.norm(p, axis=1)
                result_df[f'energy_electron{i+1}'] = (result_df[f'mom_mag_electron{i+1}']**2) / (2 * particle_masses[f'electron{i+1}'])
                particle_idx += 1
        
        # Calculate combined energies
        # KER (Kinetic Energy Release) - sum of all ion energies
        ker_cols = [f'energy_ion{i+1}' for i in range(num_ions) if f'energy_ion{i+1}' in result_df.columns]
        if ker_cols:
            result_df['KER'] = result_df[ker_cols].sum(axis=1)
        
        # Sum of electron energies
        ee_cols = [f'energy_electron{i+1}' for i in range(num_electrons) if f'energy_electron{i+1}' in result_df.columns]
        if ee_cols:
            result_df['EESum'] = result_df[ee_cols].sum(axis=1)
        
        # Total energy
        all_energy_cols = [col for col in result_df.columns if col.startswith('energy_')]
        if all_energy_cols:
            result_df['TotalEnergy'] = result_df[all_energy_cols].sum(axis=1)
        
        # Calculate angles for each particle
        particle_types = []
        for i in range(num_ions):
            particle_types.append(('ion', i+1))
        for i in range(num_neutrals):
            particle_types.append(('neutral', i+1))
        for i in range(num_electrons):
            particle_types.append(('electron', i+1))
        
        for idx, (ptype, pnum) in enumerate(particle_types):
            if idx < total_particles:
                particle_name = f'{ptype}{pnum}'
                if f'mom_mag_{particle_name}' in result_df.columns:
                    p_mag = result_df[f'mom_mag_{particle_name}']
                    p_z = df[f'particle_{idx}_Pz']
                    
                    # Calculate theta
                    cos_theta = np.clip(p_z / (p_mag + 1e-8), -1.0, 1.0)
                    result_df[f'theta_{particle_name}'] = np.arccos(cos_theta)
                    
                    # Calculate phi
                    p_x = df[f'particle_{idx}_Px']
                    p_y = df[f'particle_{idx}_Py']
                    result_df[f'phi_{particle_name}'] = np.arctan2(p_y, p_x)
        
        # Calculate relative angles between particle pairs
        for i, (ptype1, pnum1) in enumerate(particle_types):
            for j, (ptype2, pnum2) in enumerate(particle_types):
                if i < j and i < total_particles and j < total_particles:
                    p1_name = f'{ptype1}{pnum1}'
                    p2_name = f'{ptype2}{pnum2}'
                    
                    # Extract momentum vectors
                    vec1 = df[[f'particle_{i}_Px', f'particle_{i}_Py', f'particle_{i}_Pz']].values
                    vec2 = df[[f'particle_{j}_Px', f'particle_{j}_Py', f'particle_{j}_Pz']].values
                    
                    # Calculate dot products
                    dot_products = np.sum(vec1 * vec2, axis=1)
                    
                    # Get magnitudes
                    if f'mom_mag_{p1_name}' in result_df.columns and f'mom_mag_{p2_name}' in result_df.columns:
                        mag1 = result_df[f'mom_mag_{p1_name}'].values
                        mag2 = result_df[f'mom_mag_{p2_name}'].values
                        
                        # Calculate angle
                        cos_angle = dot_products / (mag1 * mag2 + 1e-8)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        result_df[f'angle_{p1_name}_{p2_name}'] = np.arccos(cos_angle)
                        result_df[f'dot_product_{p1_name}_{p2_name}'] = dot_products
        
        return result_df
    
    except Exception as e:
        print(f"Error calculating physics features: {str(e)}")
        import traceback
        traceback.print_exc()
        return df

def calculate_physics_features_with_profile(df, profile_config):
    """Calculate physics features using a specific configuration profile."""
    try:
        result_df = df.copy()
        
        if not profile_config:
            # Use default configuration
            return calculate_physics_features_flexible(df, None)
        
        particle_count = profile_config.get('particle_count', {})
        particles = profile_config.get('particles', {})
        
        num_ions = particle_count.get('ions', 0)
        num_neutrals = particle_count.get('neutrals', 0)
        num_electrons = particle_count.get('electrons', 0)
        total_particles = num_ions + num_neutrals + num_electrons
        
        # Build particle list in order
        particle_list = []
        
        # Add ions
        for i in range(num_ions):
            particle_info = particles.get(f'ion_{i}', {})
            particle_list.append({
                'type': 'ion',
                'index': i,
                'name': particle_info.get('name', f'Ion{i+1}'),
                'mass': particle_info.get('mass', 1) * 1836,  # Convert to electron masses
                'charge': particle_info.get('charge', 1)
            })
        
        # Add neutrals
        for i in range(num_neutrals):
            particle_info = particles.get(f'neutral_{i}', {})
            particle_list.append({
                'type': 'neutral',
                'index': i,
                'name': particle_info.get('name', f'Neutral{i+1}'),
                'mass': particle_info.get('mass', 16) * 1836,
                'charge': 0
            })
        
        # Add electrons
        for i in range(num_electrons):
            particle_list.append({
                'type': 'electron',
                'index': i,
                'name': 'e-',
                'mass': 1,  # Electron mass
                'charge': -1
            })
        
        # Calculate features for each particle
        for p_idx, particle in enumerate(particle_list):
            if p_idx < total_particles:
                # Get momentum components
                px = df[f'particle_{p_idx}_Px']
                py = df[f'particle_{p_idx}_Py']
                pz = df[f'particle_{p_idx}_Pz']
                p_vec = np.column_stack([px, py, pz])
                
                # Calculate momentum magnitude
                p_mag = np.linalg.norm(p_vec, axis=1)
                
                # Create feature names based on particle type and name
                if particle['type'] == 'ion':
                    feature_prefix = f"{particle['name']}_ion{particle['index']+1}"
                elif particle['type'] == 'neutral':
                    feature_prefix = f"{particle['name']}_neutral{particle['index']+1}"
                else:
                    feature_prefix = f"electron{particle['index']+1}"
                
                # Store momentum magnitude
                result_df[f'mom_mag_{feature_prefix}'] = p_mag
                
                # Calculate kinetic energy
                result_df[f'energy_{feature_prefix}'] = (p_mag**2) / (2 * particle['mass'])
                
                # Calculate angles
                cos_theta = np.clip(pz / (p_mag + 1e-8), -1.0, 1.0)
                result_df[f'theta_{feature_prefix}'] = np.arccos(cos_theta)
                result_df[f'phi_{feature_prefix}'] = np.arctan2(py, px)
        
        # Calculate combined energies based on particle types
        # KER - sum of ion energies
        ion_energy_cols = [col for col in result_df.columns 
                          if col.startswith('energy_') and '_ion' in col]
        if ion_energy_cols:
            result_df['KER'] = result_df[ion_energy_cols].sum(axis=1)
        
        # Electron energy sum
        electron_energy_cols = [col for col in result_df.columns 
                               if col.startswith('energy_electron')]
        if electron_energy_cols:
            result_df['EESum'] = result_df[electron_energy_cols].sum(axis=1)
        
        # Total energy
        all_energy_cols = [col for col in result_df.columns if col.startswith('energy_')]
        if all_energy_cols:
            result_df['TotalEnergy'] = result_df[all_energy_cols].sum(axis=1)
        
        # Calculate relative angles between particles
        for i in range(len(particle_list)):
            for j in range(i+1, len(particle_list)):
                if i < total_particles and j < total_particles:
                    p1 = particle_list[i]
                    p2 = particle_list[j]
                    
                    # Get feature prefixes
                    if p1['type'] == 'ion':
                        prefix1 = f"{p1['name']}_ion{p1['index']+1}"
                    elif p1['type'] == 'neutral':
                        prefix1 = f"{p1['name']}_neutral{p1['index']+1}"
                    else:
                        prefix1 = f"electron{p1['index']+1}"
                    
                    if p2['type'] == 'ion':
                        prefix2 = f"{p2['name']}_ion{p2['index']+1}"
                    elif p2['type'] == 'neutral':
                        prefix2 = f"{p2['name']}_neutral{p2['index']+1}"
                    else:
                        prefix2 = f"electron{p2['index']+1}"
                    
                    # Calculate relative angle
                    vec1 = df[[f'particle_{i}_Px', f'particle_{i}_Py', f'particle_{i}_Pz']].values
                    vec2 = df[[f'particle_{j}_Px', f'particle_{j}_Py', f'particle_{j}_Pz']].values
                    
                    dot_product = np.sum(vec1 * vec2, axis=1)
                    mag1 = result_df[f'mom_mag_{prefix1}'].values
                    mag2 = result_df[f'mom_mag_{prefix2}'].values
                    
                    cos_angle = dot_product / (mag1 * mag2 + 1e-8)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    
                    result_df[f'angle_{prefix1}_{prefix2}'] = np.arccos(cos_angle)
                    result_df[f'dot_product_{prefix1}_{prefix2}'] = dot_product
        
        return result_df
        
    except Exception as e:
        print(f"Error calculating physics features with profile: {str(e)}")
        import traceback
        traceback.print_exc()
        return df

def calculate_physics_features(df, config=None):
    """Wrapper function for backward compatibility."""
    if config and 'particles' in config:
        return calculate_physics_features_with_profile(df, config)
    else:
        return calculate_physics_features_flexible(df, config)

def create_feature_categories_ui(feature_columns, id_prefix):
    """Create the feature selection UI organized by categories."""
    # Group features into categories
    feature_categories = {
        'Original Momentum': [col for col in feature_columns if col.startswith('particle_')],
        'Momentum Magnitudes': [col for col in feature_columns if 'mom_mag' in col],
        'Energies': [col for col in feature_columns if any(x in col for x in ['energy_', 'KER', 'EESum', 'TotalEnergy', 'EESharing'])],
        'Angles': [col for col in feature_columns if any(x in col for x in ['theta_', 'phi_', 'angle_'])],
        'Dot Products': [col for col in feature_columns if 'dot_product' in col],
        'Differences': [col for col in feature_columns if any(x in col for x in ['diff_', 'mom_diff'])]
    }
    
    # Create the selection UI with feature categories
    feature_selection_ui = []
    for category, cols in feature_categories.items():
        if cols:  # Only add categories that have features
            category_ui = html.Div([
                html.Div(category, className='feature-category-title'),
                dcc.Checklist(
                    id={'type': f'feature-selector-{id_prefix}', 'category': category},
                    options=[{'label': col, 'value': col} for col in cols],
                    value=[],  # No default selection
                    labelStyle={'display': 'block'}
                )
            ], className='feature-category')
            feature_selection_ui.append(category_ui)
    
    if not feature_selection_ui:
        feature_selection_ui = [html.Div("No features available. Please upload files.", style={"color": "gray"})]
    
    return feature_selection_ui

# Callback to update the feature dropdowns when data is available
@app.callback(
    Output('x-axis-feature', 'options'),
    Output('y-axis-feature', 'options'),
    Input('features-data-store', 'data'),
    prevent_initial_call=True
)
def update_feature_dropdowns(features_data):
    """Update the dropdown options for custom feature plot."""
    if not features_data or 'column_names' not in features_data:
        return [], []
    
    # Get all feature columns
    feature_columns = features_data['column_names']
    
    # Create dropdown options
    options = [{'label': col, 'value': col} for col in feature_columns]
    
    return options, options

@app.callback(
    [Output('heatmap-settings', 'style'),
     Output('scatter-settings', 'style')],
    [Input('visualization-type', 'value')]
)
def toggle_visualization_settings(visualization_type):
    """Show/hide appropriate settings based on visualization type."""
    if visualization_type == 'heatmap':
        return {'display': 'block', 'marginTop': '10px'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block', 'marginTop': '10px'}


# Update the UI for autoencoder feature selection
@app.callback(
    Output('feature-selection-ui-autoencoder', 'children'),
    Input('features-data-store', 'data'),
    prevent_initial_call=True
)
def update_autoencoder_feature_ui(features_data):
    """Update the feature selection UI for the autoencoder."""
    if not features_data or 'column_names' not in features_data:
        return [html.Div("Upload files to see available features", style={"color": "gray"})]
    
    # Create feature selection UI for autoencoder
    return create_feature_categories_ui(features_data['column_names'], 'autoencoder')

# Store selected points when selection changes in Graph 1
@app.callback(
    Output('selected-points-store', 'data'),
    Output('selected-points-info', 'children'),
    Input('umap-graph', 'selectedData'),
    prevent_initial_call=True
)
def store_selected_points(selectedData):
    """Store the selected points from Graph 1."""
    if not selectedData:
        return [], "No points selected."
    
    selection_type = ""
    num_points = 0
    
    # Handle box selection
    if 'range' in selectedData:
        x_range = selectedData['range']['x']
        y_range = selectedData['range']['y']
        selection_type = f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"
    
    # Handle lasso selection
    elif 'lassoPoints' in selectedData:
        selection_type = "Lasso selection"
    
    # Handle individual point selection
    if 'points' in selectedData:
        num_points = len(selectedData['points'])
    
    info_text = [
        html.Div(f"Selection type: {selection_type}"),
        html.Div(f"Number of points: {num_points}")
    ]
    
    return selectedData, info_text

# Callback for file upload and removal
@app.callback(
    Output('stored-files', 'data'),
    Output('file-list', 'children'),
    Output('features-data-store', 'data'),
    Output('feature-selection-ui-graph1', 'children'),
    Output('feature-selection-ui-graph3', 'children'),
    Output('feature-selection-ui-graph3-selection', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input({'type': 'remove-button', 'index': ALL}, 'n_clicks'),
    State('stored-files', 'data'),
    State('features-data-store', 'data'),
    State('configuration-profiles-store', 'data'),
    State('file-config-assignments-store', 'data'),
    prevent_initial_call=True
)
def update_files(new_contents, new_filenames, remove_n_clicks, current_store, features_store, profiles_store, assignments_store):
    """Update the stored files and calculate physics features."""
    try:
        if current_store is None:
            current_store = []
        
        if features_store is None:
            features_store = {}
        
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        triggered_prop = ctx.triggered[0]['prop_id']
        print(f"Triggered by: {triggered_prop}")
        
        # Handle file uploads
        if 'upload-data.contents' in triggered_prop:
            print("Handling file upload...")
            if new_contents is not None and new_filenames is not None:
                next_id = max([f['id'] for f in current_store], default=-1) + 1
                for contents, fname in zip(new_contents, new_filenames):
                    print(f"Processing file: {fname}")
                    try:
                        df_tuple = parse_contents(contents, fname)
                        
                        if df_tuple is not None:
                            df, is_selection = df_tuple
                            
                            if is_selection:
                                # Handle selection file - store it directly
                                print(f"{fname} is a selection file")
                                file_dict = {
                                    'id': next_id,
                                    'filename': fname,
                                    'data': df.to_json(date_format='iso', orient='split'),
                                    'is_selection': True  # Flag to indicate this is a selection file
                                }
                                current_store.append(file_dict)
                                next_id += 1
                            else:
                                # Handle regular COLTRIMS file
                                print(f"{fname} is a regular COLTRIMS file")
                                try:
                                    df_reorg = reorganize_df(df)
                                    print(f"Successfully reorganized dataframe for {fname}")
                                    
                                    # Calculate physics features
                                    try:
                                        print(f"Calculating features for {fname}...")
                                        sample_size = min(1000, len(df_reorg))
                                        df_sample = df_reorg.sample(n=sample_size, random_state=42) if len(df_reorg) > sample_size else df_reorg
                                        
                                        # Get molecular configuration from the stores
                                        profiles_store = ctx.states.get('configuration-profiles-store.data', {})
                                        assignments_store = ctx.states.get('file-config-assignments-store.data', {})
                                        
                                        # Process file with its assigned configuration
                                        profile_name = assignments_store.get(fname) if assignments_store else None
                                        
                                        if profile_name and profiles_store and profile_name in profiles_store:
                                            # Use the assigned profile
                                            profile_config = profiles_store[profile_name]
                                            print(f"Processing {fname} with profile: {profile_name}")
                                            df_features = calculate_physics_features_with_profile(df_sample, profile_config)
                                        else:
                                            # Use default configuration
                                            print(f"Processing {fname} with default configuration")
                                            df_features = calculate_physics_features_flexible(df_sample, None)
                                        print(f"Feature calculation complete for {fname}")
                                        
                                        # Store the column names
                                        if 'column_names' not in features_store:
                                            features_store['column_names'] = list(df_features.columns)
                                            print(f"Updated feature column names")
                                    except Exception as e:
                                        print(f"Error during feature calculation for {fname}: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        df_features = df_reorg
                                    
                                    file_dict = {
                                        'id': next_id,
                                        'filename': fname,
                                        'data': df_reorg.to_json(date_format='iso', orient='split'),
                                        'is_selection': False  # Flag to indicate this is not a selection file
                                    }
                                    current_store.append(file_dict)
                                    next_id += 1
                                    print(f"Added {fname} to store, id={next_id-1}")
                                except Exception as e:
                                    print(f"Error in reorganize_df for {fname}: {e}")
                                    import traceback
                                    traceback.print_exc()
                        else:
                            print(f"parse_contents returned None for {fname}")
                    except Exception as e:
                        print(f"Error processing file {fname}: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Handle file removals
        elif 'remove-button' in triggered_prop:
            try:
                print("Handling file removal...")
                triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
                print(f"Triggered ID: {triggered_id}")
                remove_obj = json.loads(triggered_id)
                remove_id = remove_obj.get('index', None)
                print(f"Removing file with ID: {remove_id}")
                
                # Remove the file from the store
                current_store = [f for f in current_store if f['id'] != remove_id]
                print(f"Files after removal: {len(current_store)}")
                
                # Create file list UI
                file_list_children = []
                for f in current_store:
                    is_selection = f.get('is_selection', False)
                    selection_label = " (Selection)" if is_selection else ""
                    
                    file_list_children.append(
                        html.Div([
                            html.Span(f["filename"] + selection_label, 
                                     style={'color': '#1E88E5' if is_selection else 'black'}),
                            html.Button("×", id={'type': 'remove-button', 'index': f['id']},
                                       n_clicks=0, style={'marginLeft': '10px', 'color': 'red', 'width': 'auto'})
                        ], style={'margin': '5px', 'display': 'inline-block',
                                  'border': '1px solid #ccc', 'padding': '5px', 'borderRadius': '3px',
                                  'backgroundColor': '#e3f2fd' if is_selection else 'white'})
                    )
                
                # IMPORTANT: Don't process features when removing files
                # Just create the UI based on existing feature store
                if features_store and 'column_names' in features_store:
                    feature_ui_graph1 = create_feature_categories_ui(features_store['column_names'], 'graph1')
                    feature_ui_graph3 = create_feature_categories_ui(features_store['column_names'], 'graph3')
                    feature_ui_graph3_selection = create_feature_categories_ui(features_store['column_names'], 'graph3-selection')
                else:
                    feature_ui_graph1 = [html.Div("Upload files to see available features", style={"color": "gray"})]
                    feature_ui_graph3 = [html.Div("Upload files to see available features", style={"color": "gray"})]
                    feature_ui_graph3_selection = [html.Div("Upload files to see available features", style={"color": "gray"})]
                
                # Return early to avoid processing all files again
                return current_store, file_list_children, features_store, feature_ui_graph1, feature_ui_graph3, feature_ui_graph3_selection
                
            except Exception as e:
                print(f"Error during removal: {e}")
                import traceback
                traceback.print_exc()
                # Return current state on error
                return current_store, [], features_store or {}, [], [], []
        
        # Create file list UI, showing special indicator for selection files
        print("Creating file list UI...")
        file_list_children = []
        for f in current_store:
            is_selection = f.get('is_selection', False)
            selection_label = " (Selection)" if is_selection else ""
            
            file_list_children.append(
                html.Div([
                    html.Span(f["filename"] + selection_label, 
                             style={'color': '#1E88E5' if is_selection else 'black'}),
                    html.Button("×", id={'type': 'remove-button', 'index': f['id']},
                               n_clicks=0, style={'marginLeft': '10px', 'color': 'red', 'width': 'auto'})
                ], style={'margin': '5px', 'display': 'inline-block',
                          'border': '1px solid #ccc', 'padding': '5px', 'borderRadius': '3px',
                          'backgroundColor': '#e3f2fd' if is_selection else 'white'})
            )
        
        # Create feature selection UIs for both graphs
        print("Creating feature selection UIs...")
        if not current_store or 'column_names' not in features_store:
            feature_ui_graph1 = [html.Div("Upload files to see available features", style={"color": "gray"})]
            feature_ui_graph3 = [html.Div("Upload files to see available features", style={"color": "gray"})]
            feature_ui_graph3_selection = [html.Div("Upload files to see available features", style={"color": "gray"})]
        else:
            # Create feature selection UI for Graph 1
            feature_ui_graph1 = create_feature_categories_ui(features_store['column_names'], 'graph1')
            
            # Create feature selection UI for Graph 3 (identical structure but different IDs)
            feature_ui_graph3 = create_feature_categories_ui(features_store['column_names'], 'graph3')
            
            # Create feature selection UI for Graph 3 Selection (identical structure but different IDs)
            feature_ui_graph3_selection = create_feature_categories_ui(features_store['column_names'], 'graph3-selection')
        
        print("Callback completed successfully")
        return current_store, file_list_children, features_store, feature_ui_graph1, feature_ui_graph3, feature_ui_graph3_selection
    
    except Exception as e:
        print(f"Fatal error in update_files callback: {e}")
        import traceback
        traceback.print_exc()
        # Return the current values to avoid breaking the app
        empty_ui = [html.Div("Error loading features. Check console for details.", style={"color": "red"})]
        return current_store or [], [], features_store or {}, empty_ui, empty_ui, empty_ui

# Complete callback for "New Graph: UMAP Re-run on Graph 3 Selected Points"
@app.callback(
    Output('umap-graph-graph3-selection', 'figure'),
    Output('debug-output-graph3-selection', 'children'),
    Output('graph3-selection-umap-store', 'data'),
    Output('umap-quality-metrics-graph3-selection', 'children'),
    Input('run-umap-graph3-selection', 'n_clicks'),
    State('selected-points-run-store', 'data'),  # Selected points from Graph 3
    State('umap-graph-selected-run', 'figure'),  # Graph 3 figure data
    State('num-neighbors-graph3-selection', 'value'),
    State('min-dist-graph3-selection', 'value'),
    State({'type': 'feature-selector-graph3-selection', 'category': ALL}, 'value'),
    State('combined-data-store', 'data'),
    State('metric-selector-graph3-selection', 'value'),
    prevent_initial_call=True
)
def update_umap_graph3_selection(n_clicks, graph3_selection, graph3_figure, 
                                num_neighbors, min_dist, selected_features_list, 
                                combined_data_json, selected_metrics):
    """Run UMAP on the selected points from Graph 3, using the same approach as Graph 1."""
    try:
        # Initialize default return values
        empty_fig = {}
        empty_store = []
        empty_metrics = []
        debug_text = ""
        
        # Validate inputs
        if not graph3_selection:
            return empty_fig, "No selection data found for Graph 3. Use the lasso or box select tool.", empty_store, empty_metrics
        
        debug_text += f"Processing selection from Graph 3.<br>"
        
        # Check if graph3_subset and graph3_umap_coords are available
        if 'graph3_subset' not in combined_data_json or combined_data_json['graph3_subset'] == "{}":
            debug_text += "Graph 3 subset not found in data store.<br>"
            return empty_fig, "Graph 3 subset data not found. Please re-run Graph 3 first.", empty_store, empty_metrics
            
        if 'graph3_umap_coords' not in combined_data_json or combined_data_json['graph3_umap_coords'] == "{}":
            debug_text += "Graph 3 UMAP coordinates not found.<br>"
            return empty_fig, "Graph 3 UMAP coordinates not found. Please re-run Graph 3 first.", empty_store, empty_metrics
        
        # Load the Graph 3 subset data and UMAP coordinates
        graph3_subset_df = pd.read_json(combined_data_json['graph3_subset'], orient='split')
        graph3_umap_coords = pd.read_json(combined_data_json['graph3_umap_coords'], orient='split')
        
        debug_text += f"Found Graph 3 subset with {len(graph3_subset_df)} rows.<br>"
        debug_text += f"Found Graph 3 UMAP coordinates with {len(graph3_umap_coords)} points.<br>"
        
        # Process the selection using geometric operations - similar to Graph 1
        indices = []
        
        # Handle box selection
        if 'range' in graph3_selection:
            x_range = graph3_selection['range']['x']
            y_range = graph3_selection['range']['y']
            debug_text += f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]<br>"
            
            # Find points inside the box
            selected_mask = (
                (graph3_umap_coords['UMAP1'] >= x_range[0]) & 
                (graph3_umap_coords['UMAP1'] <= x_range[1]) & 
                (graph3_umap_coords['UMAP2'] >= y_range[0]) & 
                (graph3_umap_coords['UMAP2'] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()
            debug_text += f"Found {len(indices)} points in box selection.<br>"
        
        # Handle lasso selection
        elif 'lassoPoints' in graph3_selection:
            debug_text += "Lasso selection detected.<br>"
            
            # Extract lasso polygon coordinates
            lasso_x = graph3_selection['lassoPoints']['x']
            lasso_y = graph3_selection['lassoPoints']['y']
            
            from matplotlib.path import Path
            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            
            # Check which points are within the lasso path
            points_array = np.column_stack([graph3_umap_coords['UMAP1'], graph3_umap_coords['UMAP2']])
            inside_lasso = lasso_path.contains_points(points_array)
            
            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()
            debug_text += f"Found {len(indices)} points in lasso selection.<br>"
        
        # As a fallback, try to use points directly if available
        elif 'points' in graph3_selection and graph3_selection['points']:
            debug_text += "Direct point selection detected.<br>"
            
            # Try to map points to indices more intelligently
            all_points = graph3_selection['points']
            debug_text += f"Selection contains {len(all_points)} points.<br>"
            
            # Here we'll use curve-based indexing with a bit of validation
            for point in all_points:
                curve_num = point.get('curveNumber', -1)
                point_idx = point.get('pointIndex', -1)
                
                # Only proceed if we have valid values
                if curve_num >= 0 and point_idx >= 0:
                    # Find the corresponding index in our coords dataframe
                    # This is tricky because pointIndex is relative to each curve
                    # We'll need to reconstruct the mapping
                    
                    # First get the label for this curve
                    curve_label = None
                    if graph3_figure and 'data' in graph3_figure and curve_num < len(graph3_figure['data']):
                        trace = graph3_figure['data'][curve_num]
                        if 'name' in trace:
                            curve_label = trace['name']
                            if ' (' in curve_label:
                                curve_label = curve_label.split(' (')[0]
                    
                    if curve_label is not None:
                        # Find matching rows in our coords dataframe
                        matching_rows = graph3_umap_coords[graph3_umap_coords['file_label'] == curve_label]
                        
                        # Verify point_idx is valid for this subset
                        if 0 <= point_idx < len(matching_rows):
                            # Get the actual index in the full dataframe
                            actual_idx = matching_rows.iloc[point_idx].name
                            indices.append(actual_idx)
                        else:
                            debug_text += f"Warning: pointIndex {point_idx} out of range for curve {curve_num} with label {curve_label}.<br>"
            
            debug_text += f"Mapped {len(indices)} points from direct selection.<br>"
        
        if not indices:
            return empty_fig, "No valid points found in the selection region.", empty_store, empty_metrics
        
        # Get counts by label before extraction
        label_counts_before = graph3_umap_coords.iloc[indices]['file_label'].value_counts().to_dict()
        debug_text += "Label distribution in selection:<br>"
        for label, count in sorted(label_counts_before.items()):
            debug_text += f"- {label}: {count} points<br>"
        
        # Verify indices are valid
        valid_indices = [i for i in indices if 0 <= i < len(graph3_subset_df)]
        if len(valid_indices) != len(indices):
            debug_text += f"Warning: {len(indices) - len(valid_indices)} invalid indices were removed.<br>"
            indices = valid_indices
        
        if not indices:
            return empty_fig, "No valid indices found in the selection.", empty_store, empty_metrics
        
        # Extract the subset of data for selected points
        selected_df = graph3_subset_df.iloc[indices].copy()
        debug_text += f"Created dataframe with {len(selected_df)} rows.<br>"
        
        # Store the original labels to ensure consistency
        original_labels = selected_df['file_label'].values
        
        # Collect selected features for UMAP
        all_selected_features = []
        for features in selected_features_list:
            if features:  # Only add non-empty lists
                all_selected_features.extend(features)
        
        # Use selected features for UMAP if available
        if all_selected_features:
            feature_cols = [col for col in selected_df.columns if col in all_selected_features and col != 'file_label']
            if feature_cols:
                debug_text += f"Using {len(feature_cols)} selected features for UMAP.<br>"
                X = selected_df[feature_cols].to_numpy()
            else:
                # Fallback to momentum columns
                momentum_cols = [col for col in selected_df.columns if col.startswith('particle_') and col != 'file_label']
                debug_text += f"No valid selected features. Using {len(momentum_cols)} momentum columns.<br>"
                X = selected_df[momentum_cols].to_numpy()
        else:
            # Use momentum columns
            momentum_cols = [col for col in selected_df.columns if col.startswith('particle_') and col != 'file_label']
            debug_text += f"No features selected. Using {len(momentum_cols)} momentum columns.<br>"
            X = selected_df[momentum_cols].to_numpy()
        
        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Run UMAP
        try:
            debug_text += "Running UMAP...<br>"
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(int(num_neighbors), len(X)-1),  # Ensure n_neighbors is valid
                min_dist=float(min_dist),
                metric='euclidean',
                random_state=42
            )
            
            # Fit UMAP
            umap_result = reducer.fit_transform(X)
            debug_text += "UMAP transformation completed successfully.<br>"
        except Exception as e:
            debug_text += f"Error running UMAP: {str(e)}<br>"
            return empty_fig, f"Error running UMAP: {str(e)}<br>Debug info: {debug_text}", empty_store, empty_metrics
        
        # Create DataFrame for visualization with original labels
        result_df = pd.DataFrame({
            'UMAP1': umap_result[:, 0],
            'UMAP2': umap_result[:, 1],
            'file_label': original_labels,  # Use original labels to maintain correspondence
            'original_index': indices
        })
        
        # Extract color information from Graph 3 figure
        color_map = {}
        if graph3_figure and 'data' in graph3_figure:
            for trace in graph3_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    # Clean the label if it contains point count
                    clean_name = trace['name']
                    if ' (' in clean_name:
                        clean_name = clean_name.split(' (')[0]
                    color_map[clean_name] = trace['marker']['color']
        
        # Create figure with the same colors as Graph 3
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Create one trace per label for clean visualization
        for label in result_df['file_label'].unique():
            mask = result_df['file_label'] == label
            df_subset = result_df[mask]
            
            # Get color from Graph 3 or use default
            color = color_map.get(label, None)
            
            # Add trace for this label
            fig.add_trace(go.Scatter(
                x=df_subset['UMAP1'],
                y=df_subset['UMAP2'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                name=f"{label} ({len(df_subset)} pts)"
            ))
        
        # Update figure properties
        fig.update_layout(
            height=600,
            title=f"UMAP on Graph 3 Selected Points (n_neighbors={num_neighbors}, min_dist={min_dist})",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data File"
        )
        
        # Store the result for potential future use
        graph3_selection_umap_store = {
            "umap_coords": result_df.to_json(date_format='iso', orient='split'),
            "feature_data": selected_df.to_json(date_format='iso', orient='split')
        }

        # Calculate clustering metrics WITH SMART CONFIDENCE
        metrics_children = []
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            # Get UMAP coordinates for clustering
            X_umap = result_df[['UMAP1', 'UMAP2']].to_numpy()
            
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
            
            # Collect metrics for confidence calculation
            metrics = {}
            unique_clusters = set(cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = np.sum(cluster_labels == -1)
            noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            
            metrics['noise_ratio'] = noise_ratio
            
            # Only calculate metrics if we have at least 2 clusters
            if n_clusters >= 2:
                # For metrics, we need to exclude noise points (-1)
                mask = cluster_labels != -1
                non_noise_points = np.sum(mask)
                non_noise_clusters = len(set(cluster_labels[mask]))
                
                if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                    if 'silhouette' in selected_metrics:
                        metrics["silhouette"] = silhouette_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    if 'davies_bouldin' in selected_metrics:
                        metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    if 'calinski_harabasz' in selected_metrics:
                        metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    # Add new metrics based on selection
                    if 'hopkins' in selected_metrics:
                        h_stat = hopkins_statistic(X_umap_scaled)
                        metrics["hopkins"] = h_stat
                    
                    if 'stability' in selected_metrics:
                        stability = cluster_stability(X_umap_scaled, best_eps, 5, n_iterations=3)
                        metrics["stability"] = stability
                    
                    # Add physics consistency if selected
                    if 'physics_consistency' in selected_metrics and selected_df is not None and not selected_df.empty:
                        physics_metrics = physics_cluster_consistency(selected_df, cluster_labels)
                        metrics.update(physics_metrics)
            
            # SMART CONFIDENCE CALCULATION
            confidence_data = calculate_adaptive_confidence_score(
                metrics, 
                clustering_method='dbscan'
            )
            
            # Create the smart confidence UI
            metrics_children = [create_smart_confidence_ui(confidence_data)]
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            metrics_children = [html.Div(f"Error calculating confidence: {str(e)}")]
        
        return fig, debug_text, graph3_selection_umap_store, metrics_children
        
    except Exception as e:
        print(f"Error in Graph 3 selection UMAP: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error running UMAP on Graph 3 selection: {str(e)}"
        return {}, error_msg, [], []

# Callback to update the file selector checklist
@app.callback(
    Output('umap-file-selector', 'options'),
    Output('umap-file-selector', 'value'),
    Input('stored-files', 'data')
)
def update_umap_selector(stored_files):
    """Update the file selector dropdown based on uploaded files."""
    if not stored_files:
        return [], []
    
    options = [{'label': f['filename'], 'value': f['id']} for f in stored_files]
    values = [f['id'] for f in stored_files]  # Select all files by default
    
    return options, values
    

# Consolidated callback for status updates - updated function signature and implementation
@app.callback(
    Output('run-umap-status', 'children'),
    Output('show-selected-status', 'children'),
    Output('run-umap-selected-run-status', 'children'),
    Output('plot-custom-features-status', 'children'),
    Output('save-selection-status', 'children'),
    Output('show-selected-run-status', 'children'),
    Output('save-selection-run-status', 'children'),
    Output('run-umap-graph3-selection-status', 'children'),
    Output('save-selection-graph3-selection-status', 'children'),
    Output('train-autoencoder-status', 'children'),
    Output('run-umap-latent-status', 'children'),
    Output('save-latent-features-status', 'children'),
    Output('genetic-features-status', 'children'),
    Output('run-umap-genetic-status', 'children', allow_duplicate=True),
    Output('save-genetic-features-status', 'children'),
    Output('mi-features-status', 'children'),
    Output('run-umap-mi-status', 'children'),
    Output('save-mi-features-status', 'children'),
    Output('generate-plot-graph15-status', 'children'),
    Output('save-selection-graph15-status', 'children'),
    Output('show-selected-graph15-status', 'children'),
    Output('save-selection-graph25-status', 'children'),
    Input('run-umap', 'n_clicks'),
    Input('umap-graph', 'figure'),
    Input('show-selected', 'n_clicks'),
    Input('umap-graph-selected-only', 'figure'),
    Input('run-umap-selected-run', 'n_clicks'),
    Input('umap-graph-selected-run', 'figure'),
    Input('plot-custom-features', 'n_clicks'),
    Input('custom-feature-plot', 'figure'),
    Input('save-selection-btn', 'n_clicks'),
    Input('download-selection', 'data'),
    Input('show-selected-run', 'n_clicks'),
    Input('save-selection-run-btn', 'n_clicks'),
    Input('download-selection-run', 'data'),
    Input('run-umap-graph3-selection', 'n_clicks'),
    Input('umap-graph-graph3-selection', 'figure'),
    Input('save-selection-graph3-selection-btn', 'n_clicks'),
    Input('download-selection-graph3-selection', 'data'),
    Input('train-autoencoder', 'n_clicks'),
    Input('autoencoder-umap-graph', 'figure'),
    Input('run-umap-latent', 'n_clicks'),
    Input('save-latent-features-btn', 'n_clicks'),
    Input('download-latent-features', 'data'),
    Input('run-genetic-features', 'n_clicks'),
    Input('genetic-features-graph', 'figure'),
    Input('run-umap-genetic', 'n_clicks'),
    Input('save-genetic-features-btn', 'n_clicks'),
    Input('download-genetic-features', 'data'),
    Input('run-mi-features', 'n_clicks'),
    Input('mi-features-graph', 'figure'),
    Input('run-umap-mi', 'n_clicks'),
    Input('save-mi-features-btn', 'n_clicks'),
    Input('download-mi-features', 'data'),
    Input('generate-plot-graph15', 'n_clicks'),
    Input('scatter-graph15', 'figure'),
    Input('save-selection-graph15-btn', 'n_clicks'),
    Input('download-selection-graph15', 'data'),
    Input('show-selected-graph15', 'n_clicks'),
    Input('graph25', 'figure'),
    Input('save-selection-graph25-btn', 'n_clicks'),
    Input('download-selection-graph25', 'data'),
    prevent_initial_call=True
)
def update_all_status(run_umap_clicks, umap_fig, show_selected_clicks, selected_fig, 
                     run_umap_sel_clicks, sel_run_fig, plot_features_clicks, custom_plot_fig,
                     save_selection_clicks, download_data, show_selected_run_clicks,
                     save_selection_run_clicks, download_run_data,
                     run_umap_graph3_sel_clicks, graph3_sel_fig,
                     save_selection_graph3_sel_clicks, download_graph3_sel_data,
                     train_autoencoder_clicks, autoencoder_fig,
                     run_umap_latent_clicks, save_latent_clicks, download_latent_data,
                     run_genetic_features_clicks, genetic_features_fig,
                     run_umap_genetic_clicks, save_genetic_features_clicks, download_genetic_features_data,
                     run_mi_features_clicks, mi_features_fig,
                     run_umap_mi_clicks, save_mi_features_clicks, download_mi_features_data,
                     generate_plot_graph15_clicks, scatter_graph15_fig,
                     save_selection_graph15_clicks, download_selection_graph15_data,
                     show_selected_graph15_clicks, graph25_figure,
                     save_selection_graph25_clicks, download_selection_graph25_data):
    """Consolidated callback to update all status messages."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Initialize all status values to dash.no_update
    statuses = [dash.no_update] * 22  # Total number of status outputs
    
    # Handle run-umap status
    if trigger_id == 'run-umap':
        statuses[0] = "Running UMAP analysis..."
    elif trigger_id == 'umap-graph':
        statuses[0] = "UMAP analysis complete!"
    
    # Handle show-selected status
    elif trigger_id == 'show-selected':
        statuses[1] = "Processing selection..."
    elif trigger_id == 'umap-graph-selected-only':
        statuses[1] = "Selection displayed!"
    
    # Handle run-umap-selected-run status
    elif trigger_id == 'run-umap-selected-run':
        statuses[2] = "Re-running UMAP on selection..."
    elif trigger_id == 'umap-graph-selected-run':
        statuses[2] = "UMAP re-run complete!"
    
    # Handle plot-custom-features status
    elif trigger_id == 'plot-custom-features':
        statuses[3] = "Creating custom feature plot..."
    elif trigger_id == 'custom-feature-plot':
        statuses[3] = "Custom feature plot complete!"
    
    # Handle save-selection status
    elif trigger_id == 'save-selection-btn':
        statuses[4] = "Preparing selection for download..."
    elif trigger_id == 'download-selection':
        statuses[4] = "Selection saved successfully!"
    
    # Handle show-selected-run status
    elif trigger_id == 'show-selected-run':
        statuses[5] = "Processing Graph 3 selection..."
    
    # Handle save-selection-run status
    elif trigger_id == 'save-selection-run-btn':
        statuses[6] = "Preparing Graph 3 selection for download..."
    elif trigger_id == 'download-selection-run':
        statuses[6] = "Graph 3 selection saved successfully!"
    
    # Handle run-umap-graph3-selection status
    elif trigger_id == 'run-umap-graph3-selection':
        statuses[7] = "Running UMAP on Graph 3 selection..."
    elif trigger_id == 'umap-graph-graph3-selection':
        statuses[7] = "UMAP on Graph 3 selection complete!"
    
    # Handle save-selection-graph3-selection status
    elif trigger_id == 'save-selection-graph3-selection-btn':
        statuses[8] = "Preparing Graph 3 selection UMAP for download..."
    elif trigger_id == 'download-selection-graph3-selection':
        statuses[8] = "Graph 3 selection UMAP saved successfully!"

    # Handle autoencoder statuses
    elif trigger_id == 'train-autoencoder':
        statuses[9] = "Training autoencoder... This may take a while."
    elif trigger_id == 'autoencoder-umap-graph':
        statuses[9] = "Autoencoder training complete!"
    
    # Handle run-umap-latent status
    elif trigger_id == 'run-umap-latent':
        statuses[10] = "Running UMAP on latent space..."
    
    # Handle save-latent-features status
    elif trigger_id == 'save-latent-features-btn':
        statuses[11] = "Preparing latent features for download..."
    elif trigger_id == 'download-latent-features':
        statuses[11] = "Latent features saved successfully!"
    
    # Handle genetic feature statuses
    elif trigger_id == 'run-genetic-features':
        statuses[12] = "Running genetic feature discovery... This may take a while."
    elif trigger_id == 'genetic-features-graph':
        statuses[12] = "Genetic feature discovery complete!"
    
    # Handle run-umap-genetic status
    elif trigger_id == 'run-umap-genetic':
        statuses[13] = "Running UMAP on genetic features..."
    
    # Handle save-genetic-features status
    elif trigger_id == 'save-genetic-features-btn':
        statuses[14] = "Preparing genetic features for download..."
    elif trigger_id == 'download-genetic-features':
        statuses[14] = "Genetic features saved successfully!"
    
    # Handle MI feature statuses
    elif trigger_id == 'run-mi-features':
        statuses[15] = "Running mutual information feature selection... This may take a while."
    elif trigger_id == 'mi-features-graph':
        statuses[15] = "Mutual information feature selection complete!"
    
    # Handle run-umap-mi status
    elif trigger_id == 'run-umap-mi':
        statuses[16] = "Running UMAP on MI-selected features..."
    
    # Handle save-mi-features status
    elif trigger_id == 'save-mi-features-btn':
        statuses[17] = "Preparing MI features for download..."
    elif trigger_id == 'download-mi-features':
        statuses[17] = "MI features saved successfully!"
    
    # Handle Graph 1.5 status updates
    elif trigger_id == 'generate-plot-graph15':
        statuses[18] = "Generating custom scatter plot..."
    elif trigger_id == 'scatter-graph15':
        statuses[18] = "Custom scatter plot generated!"
    elif trigger_id == 'save-selection-graph15-btn':
        statuses[19] = "Preparing Graph 1.5 selection for download..."
    elif trigger_id == 'download-selection-graph15':
        statuses[19] = "Graph 1.5 selection saved successfully!"
    
    # Handle Graph 2.5 status updates
    elif trigger_id == 'show-selected-graph15':
        statuses[20] = "Processing Graph 1.5 selection..."
    elif trigger_id == 'graph25':
        statuses[20] = "Selection displayed!"
    elif trigger_id == 'save-selection-graph25-btn':
        statuses[21] = "Preparing Graph 2.5 selection for download..."
    elif trigger_id == 'download-selection-graph25':
        statuses[21] = "Graph 2.5 selection saved successfully!"
    
    return tuple(statuses)

# Callback for Graph 1: Original UMAP Embedding with selected features
@app.callback(
    Output('umap-graph', 'figure'),
    Output('debug-output', 'children'),
    Output('combined-data-store', 'data'),
    Output('umap-quality-metrics', 'children'),
    Input('run-umap', 'n_clicks'),
    State('stored-files', 'data'),
    State('umap-file-selector', 'value'),
    State('num-neighbors', 'value'),
    State('min-dist', 'value'),
    State('sample-frac', 'value'),
    State({'type': 'feature-selector-graph1', 'category': ALL}, 'value'),
    State('metric-selector', 'value'),
    State('point-opacity', 'value'),
    State('color-mode', 'value'),
    State('visualization-type', 'value'),
    State('heatmap-bandwidth', 'value'),
    State('heatmap-colorscale', 'value'),
    State('show-points-overlay', 'value'),
    prevent_initial_call=True
)
def update_umap(n_clicks, stored_files, selected_ids, num_neighbors, min_dist, sample_frac, 
                selected_features_list, selected_metrics, point_opacity, color_mode, visualization_type,
                heatmap_bandwidth, heatmap_colorscale, show_points_overlay):
    """Compute UMAP embedding on selected files using selected features."""
    if not stored_files:
        return {}, "No files uploaded.", {}, [html.Div("No files uploaded.")]
    
    if not selected_ids:
        return {}, "No files selected for UMAP.", {}, [html.Div("No files selected.")]
    
    try:
        # Collect all selected features
        all_selected_features = []
        for features in selected_features_list:
            if features:  # Only add non-empty lists
                all_selected_features.extend(features)
        
        sampled_dfs = []
        debug_str = ""
        selection_dfs = []  # Separate list for selection files to handle differently
        
        # Process each selected file
        for f in stored_files:
            if f['id'] in selected_ids:
                try:
                    df = pd.read_json(f['data'], orient='split')
                    is_selection = f.get('is_selection', False)
                    
                    if is_selection:
                        # This is a saved selection file
                        debug_str += f"{f['filename']}: Selection file with {len(df)} events.<br>"
                        
                        # Make sure it has required columns for visualization
                        if 'UMAP1' in df.columns and 'UMAP2' in df.columns and 'file_label' in df.columns:
                            selection_dfs.append(df)
                        else:
                            debug_str += f"Warning: Selection file {f['filename']} is missing required columns.<br>"
                    else:
                        # Regular COLTRIMS file
                        df['file_label'] = f['filename']  # Add file name as a label
                        
                        # Calculate physics features for this file's data
                        df_with_features = calculate_physics_features(df)
                        
                        # Sample the data to reduce processing time
                        sample_size = max(int(len(df_with_features) * sample_frac), 100)  # Ensure at least 100 points
                        if len(df_with_features) > sample_size:
                            sampled = df_with_features.sample(n=sample_size, random_state=42)
                        else:
                            sampled = df_with_features
                        
                        debug_str += f"{f['filename']}: {len(df)} events, sampled {len(sampled)}.<br>"
                        sampled_dfs.append(sampled)
                except Exception as e:
                    debug_str += f"Error processing {f['filename']}: {str(e)}.<br>"
        
        # Process regular COLTRIMS files (if any)
        combined_df = None
        umap_df = pd.DataFrame(columns=["UMAP1", "UMAP2", "file_label"])
        
        if len(sampled_dfs) > 0:
            # Combine all selected datasets
            combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
            debug_str += f"Combined data shape: {combined_df.shape}.<br>"
            
            # Use selected features for UMAP
            if all_selected_features and len(all_selected_features) > 0:
                feature_cols = [col for col in combined_df.columns if col in all_selected_features]
                if feature_cols:
                    debug_str += f"Using selected features for UMAP: {', '.join(feature_cols)}<br>"
                    X = combined_df[feature_cols].to_numpy()
                else:
                    # Fallback to original momentum columns
                    original_cols = [col for col in combined_df.columns if col.startswith('particle_')]
                    X = combined_df[original_cols].to_numpy()
                    debug_str += "No valid features selected, using original momentum components.<br>"
            else:
                # Use original momentum columns
                original_cols = [col for col in combined_df.columns if col.startswith('particle_')]
                X = combined_df[original_cols].to_numpy()
                debug_str += "No features selected, using original momentum components.<br>"
            
            # Run UMAP
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=int(num_neighbors),
                min_dist=float(min_dist),
                metric='euclidean',
                random_state=42
            )
            
            # Handle NaN/inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Fit UMAP
            umap_data = reducer.fit_transform(X)
            
            # Create DataFrame for visualization
            umap_df = pd.DataFrame(umap_data, columns=["UMAP1", "UMAP2"])
            umap_df['file_label'] = combined_df['file_label']
        else:
            combined_df = pd.DataFrame()
        
        # Add any selection files directly to the visualization
        for sel_df in selection_dfs:
            # Add the selection data to the UMAP visualization data
            # Use only the necessary columns for visualization
            selection_viz_df = sel_df[['UMAP1', 'UMAP2', 'file_label']].copy()
            
            # Append to the UMAP dataframe
            umap_df = pd.concat([umap_df, selection_viz_df], ignore_index=True)
        
        # Calculate clustering for DBSCAN coloring (do this regardless of color mode)
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Get UMAP coordinates for clustering
        X_umap = umap_df[['UMAP1', 'UMAP2']].to_numpy()
        
        # Scale the data for better DBSCAN performance
        scaler = StandardScaler()
        X_umap_scaled = scaler.fit_transform(X_umap)
        
        # Find a reasonable epsilon
        eps_candidates = np.linspace(0.1, 1.0, 10)
        best_eps = 0.5  # Default
        max_clusters = 0
        
        # Try different eps values and pick the one that gives a reasonable number of clusters
        for eps in eps_candidates:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_umap_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # We want to maximize the number of clusters but avoid too many noise points
            noise_count = np.sum(labels == -1)
            noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0
            
            # Good balance: enough clusters but not too many noise points
            if n_clusters >= 2 and noise_ratio < 0.5 and n_clusters > max_clusters:
                max_clusters = n_clusters
                best_eps = eps
        
        # Run DBSCAN with the best eps
        dbscan = DBSCAN(eps=best_eps, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_umap_scaled)
        
        # Create color mappings for both file and cluster modes
        # Color mapping for file labels
        unique_labels = umap_df['file_label'].unique()
        colorscale = px.colors.qualitative.Plotly  # Use Plotly's default colorscale
        color_map = {label: colorscale[i % len(colorscale)] for i, label in enumerate(unique_labels)}
        
        # Color mapping for clusters
        # Special handling for noise points (-1 label)
        unique_clusters = sorted(set(cluster_labels))
        if -1 in unique_clusters:
            # Move noise to the end
            unique_clusters.remove(-1)
            unique_clusters.append(-1)
        
        # Use a colorscale that works well for clusters
        if len(unique_clusters) <= 10:
            cluster_colorscale = px.colors.qualitative.D3  # Good for distinct clusters
        else:
            # For many clusters, use a continuous colorscale
            cluster_colorscale = px.colors.sequential.Viridis
        
        # Create color mapping for clusters
        cluster_colors = {}
        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:  # Noise points
                cluster_colors[cluster] = 'rgba(150,150,150,0.5)'  # Gray for noise
            else:
                # Regular clusters
                if len(unique_clusters) - (1 if -1 in unique_clusters else 0) <= 10:
                    colorscale_idx = i % len(cluster_colorscale)
                    cluster_colors[cluster] = cluster_colorscale[colorscale_idx]
                else:
                    # For many clusters, distribute colors evenly
                    n_real_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                    idx = i / (n_real_clusters - 1) if n_real_clusters > 1 else 0
                    idx = min(0.99, max(0, idx))  # Ensure it's between 0 and 1
                    color_idx = int(idx * (len(cluster_colorscale) - 1))
                    cluster_colors[cluster] = cluster_colorscale[color_idx]
        
        # Initialize our figure
        import plotly.graph_objects as go
        
        # Heatmap visualization
        if visualization_type == 'heatmap':
            from scipy.stats import gaussian_kde
            
            fig = go.Figure()
            
            # Get UMAP coordinates
            umap_data = umap_df[['UMAP1', 'UMAP2']].to_numpy()
            
            # Create the grid for the heatmap
            x_min, x_max = umap_data[:, 0].min() - 0.5, umap_data[:, 0].max() + 0.5
            y_min, y_max = umap_data[:, 1].min() - 0.5, umap_data[:, 1].max() + 0.5
            
            # Create a meshgrid
            grid_size = 200
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            xx, yy = np.meshgrid(x_grid, y_grid)
            grid_points = np.column_stack([xx.flatten(), yy.flatten()])
            
            # Compute KDE (Kernel Density Estimation)
            kde = gaussian_kde(umap_data.T, bw_method=heatmap_bandwidth)
            densities = kde(grid_points.T).reshape(grid_size, grid_size)
            
            # Add heatmap
            fig.add_trace(go.Heatmap(
                x=x_grid,
                y=y_grid,
                z=densities,
                colorscale=heatmap_colorscale,
                showscale=True,
                colorbar=dict(title='Density'),
                hoverinfo='none'
            ))
            
            # Optionally, overlay scatter points with reduced opacity for context
            if show_points_overlay == 'yes':
                if color_mode == 'file':
                    # Color by file source with reduced opacity
                    for label in umap_df['file_label'].unique():
                        mask = umap_df['file_label'] == label
                        df_subset = umap_df[mask]
                        
                        fig.add_trace(go.Scatter(
                            x=df_subset["UMAP1"],
                            y=df_subset["UMAP2"],
                            mode='markers',
                            marker=dict(
                                size=4,  # Smaller points
                                color=color_map[label],
                                opacity=0.3,  # Reduced opacity
                                line=dict(width=0)
                            ),
                            name=f"{label} ({len(df_subset)} pts)"
                        ))
                elif color_mode == 'cluster':
                    # Color by DBSCAN cluster with reduced opacity
                    for cluster in unique_clusters:
                        mask = cluster_labels == cluster
                        cluster_points = umap_df.iloc[mask]
                        
                        # For noise points, make them smaller and more transparent
                        marker_size = 3 if cluster == -1 else 4
                        marker_opacity = 0.2 if cluster == -1 else 0.3
                        
                        fig.add_trace(go.Scatter(
                            x=cluster_points["UMAP1"],
                            y=cluster_points["UMAP2"],
                            mode='markers',
                            marker=dict(
                                size=marker_size,
                                color=cluster_colors[cluster],
                                opacity=marker_opacity,
                                line=dict(width=0)
                            ),
                            name=f"Cluster {cluster if cluster != -1 else 'Noise'} ({len(cluster_points)} pts)"
                        ))
        else:
            # Original scatter plot visualization
            fig = go.Figure()
            
            # Visualization based on color mode
            if color_mode == 'file':
                # Color by file source (original coloring)
                # Add traces for each file label
                for label in unique_labels:
                    mask = umap_df['file_label'] == label
                    df_subset = umap_df[mask]
                    
                    fig.add_trace(go.Scatter(
                        x=df_subset["UMAP1"],
                        y=df_subset["UMAP2"],
                        mode='markers',
                        marker=dict(
                            size=7,
                            color=color_map[label],
                            opacity=point_opacity,
                            line=dict(width=0)
                        ),
                        name=f"{label} ({len(df_subset)} pts)"
                    ))
                    
            elif color_mode == 'cluster':
                # Color by DBSCAN cluster
                # Add points for each cluster
                for cluster in unique_clusters:
                    mask = cluster_labels == cluster
                    
                    # Get points for this cluster
                    cluster_points = umap_df.iloc[mask]
                    
                    # For noise points, make them smaller and more transparent
                    marker_size = 5 if cluster == -1 else 7
                    marker_opacity = point_opacity * 0.7 if cluster == -1 else point_opacity
                    
                    # Add trace for this cluster
                    fig.add_trace(go.Scatter(
                        x=cluster_points["UMAP1"],
                        y=cluster_points["UMAP2"],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            color=cluster_colors[cluster],
                            opacity=marker_opacity,
                            line=dict(width=0)
                        ),
                        name=f"Cluster {cluster if cluster != -1 else 'Noise'} ({len(cluster_points)} pts)"
                    ))
        
        # Update figure properties
        title_suffix = ""
        if color_mode == 'cluster':
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = np.sum(cluster_labels == -1)
            title_suffix = f" - {n_clusters} clusters detected"
        
        # Adjust legend position based on number of traces
        legend_y_position = -0.5 if len(fig.data) > 12 else -0.4 if len(fig.data) > 8 else -0.3
        
        # Create legend configuration
        legend_config = dict(
            orientation="h",
            yanchor="top",     # Anchor to the top of the legend box
            y=legend_y_position,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",  # More opaque background for readability
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10),  # Smaller font for many items
            itemwidth=30,       # Smaller item width for more compact, richer color symbols
            itemsizing="constant",
            tracegroupgap=5     # Reduced gap between legend groups
        )
        
        # Adjust figure height based on legend size - more space for many clusters
        figure_height = 600
        if len(fig.data) > 15:
            figure_height = 750
        elif len(fig.data) > 10:
            figure_height = 700
        elif len(fig.data) > 6:
            figure_height = 650
        
        # Apply the layout settings
        if visualization_type == 'heatmap':
            title = f"UMAP Density Heatmap (n_neighbors={num_neighbors}, min_dist={min_dist}){title_suffix}"
        else:
            title = f"UMAP Embedding (n_neighbors={num_neighbors}, min_dist={min_dist}){title_suffix}"
            
        fig.update_layout(
            height=figure_height,
            title=title,
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title=f"{'Clusters' if color_mode == 'cluster' else 'Data File'}",
            dragmode='lasso',  
            legend=legend_config,
            modebar=dict(add=['lasso2d', 'select2d']),
            margin=dict(l=50, r=50, t=50, b=100)  # Increased bottom margin for legend
        )
 
        # Store data for other callbacks
        combined_data_json = {
            "combined_df": combined_df.to_json(date_format='iso', orient='split') if not combined_df.empty else "{}",
            "umap_coords": umap_df.to_json(date_format='iso', orient='split'),
            "selected_features_graph1": all_selected_features,
            "cluster_labels": cluster_labels.tolist() if len(cluster_labels) > 0 else []
        }

        
         # Calculate clustering metrics WITH SMART CONFIDENCE
        metrics_children = []
        if not umap_df.empty and 'UMAP1' in umap_df.columns and 'UMAP2' in umap_df.columns:
            try:
                from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
                
                # Get UMAP coordinates for clustering
                X_umap = umap_df[['UMAP1', 'UMAP2']].to_numpy()
                
                # Scale the data for better DBSCAN performance
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
                    noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0
                    
                    if n_clusters >= 2 and noise_ratio < 0.5 and n_clusters > max_clusters:
                        max_clusters = n_clusters
                        best_eps = eps
                
                # Run DBSCAN with best parameters
                dbscan = DBSCAN(eps=best_eps, min_samples=5)
                cluster_labels = dbscan.fit_predict(X_umap_scaled)
                
                # Collect metrics for confidence calculation
                metrics = {}
                unique_clusters = set(cluster_labels)
                n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                n_noise = np.sum(cluster_labels == -1)
                noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
                
                metrics['noise_ratio'] = noise_ratio
                
                # Calculate selected metrics
                if n_clusters >= 2:
                    mask = cluster_labels != -1
                    non_noise_points = np.sum(mask)
                    non_noise_clusters = len(set(cluster_labels[mask]))
                    
                    if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                        if 'silhouette' in selected_metrics:
                            metrics["silhouette"] = silhouette_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        if 'davies_bouldin' in selected_metrics:
                            metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        if 'calinski_harabasz' in selected_metrics:
                            metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        if 'hopkins' in selected_metrics:
                            metrics["hopkins"] = hopkins_statistic(X_umap_scaled)
                        
                        if 'stability' in selected_metrics:
                            metrics["stability"] = cluster_stability(X_umap_scaled, best_eps, 5, n_iterations=3)
                        
                        if 'physics_consistency' in selected_metrics and combined_df is not None and not combined_df.empty:
                            physics_metrics = physics_cluster_consistency(combined_df, cluster_labels)
                            metrics.update(physics_metrics)
                
                # SMART CONFIDENCE CALCULATION
                confidence_data = calculate_adaptive_confidence_score(
                    metrics, 
                    clustering_method='dbscan'
                )
                
                # Create the smart confidence UI
                metrics_children = [create_smart_confidence_ui(confidence_data)]
                
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                metrics_children = [html.Div(f"Error calculating confidence: {str(e)}")]
        else:
            metrics_children = [html.Div("Run UMAP to see reliability assessment")]
        
        return fig, debug_str, combined_data_json, metrics_children
    
    except Exception as e:
        print(f"Error in update_umap: {e}")
        import traceback
        traceback.print_exc()
        return {}, f"Error computing UMAP: {str(e)}", {}, [html.Div(f"Error computing UMAP: {str(e)}")]
        
# Callback for Graph 2: Selected Points (Stored Coordinates)
@app.callback(
    Output('umap-graph-selected-only', 'figure'),
    Output('debug-output-selected-only', 'children'),
    Input('show-selected', 'n_clicks'),
    State('selected-points-store', 'data'),  # Use stored selection data
    State('combined-data-store', 'data'),
    State('umap-graph', 'figure'),  # Add original figure as input
    prevent_initial_call=True
)
def update_umap_selected_only(n_clicks, selectedData, combined_data_json, original_figure):
    """Display the selected points from Graph 1."""
    try:
        if not combined_data_json:
            return {}, "No UMAP data available. Run UMAP first."
        
        # Load the UMAP coordinates
        umap_df_all = pd.read_json(combined_data_json["umap_coords"], orient='split').reset_index(drop=True)
        
        if not selectedData:
            return {}, "No points selected. Use the lasso or box select tool on Graph 1."
        
        # Initialize variables to hold indices
        indices = []
        debug_text = ""
        
        # Handle box selection
        if 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            debug_text = f"Box selection: x: [{x_range[0]:.2f}, {x_range[1]:.2f}], y: [{y_range[0]:.2f}, {y_range[1]:.2f}]<br>"
            
            selected_mask = (
                (umap_df_all['UMAP1'] >= x_range[0]) & 
                (umap_df_all['UMAP1'] <= x_range[1]) & 
                (umap_df_all['UMAP2'] >= y_range[0]) & 
                (umap_df_all['UMAP2'] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()
        
        # Handle lasso selection
        elif 'lassoPoints' in selectedData:
            debug_text = "Lasso selection<br>"
            # Instead of trying to extract indices from points, we'll use the coordinates
            # from the lassoPoints and find the points within the polygon
            
            # Extract lasso polygon coordinates
            lasso_x = selectedData['lassoPoints']['x']
            lasso_y = selectedData['lassoPoints']['y']
            
            from matplotlib.path import Path
            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            
            # Check which points are within the lasso path
            points_array = np.column_stack([umap_df_all['UMAP1'], umap_df_all['UMAP2']])
            inside_lasso = lasso_path.contains_points(points_array)
            
            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()
        
        # Handle direct point selection (fallback)
        elif 'points' in selectedData:
            indices = [pt['pointIndex'] for pt in selectedData['points']]
            debug_text = f"Direct point selection: {len(indices)} points selected<br>"
        
        if not indices:
            return {}, "No valid selection or no points found in selection area."
        
        # Extract the selected points
        selected_umap_df = umap_df_all.iloc[indices].reset_index(drop=True)
        
        # Extract color information from original figure
        import plotly.graph_objects as go
        
        # Create a color map from the original figure
        color_map = {}
        if original_figure and 'data' in original_figure:
            for trace in original_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    color_map[trace['name']] = trace['marker']['color']
        
        # Create figure with consistent colors
        fig = go.Figure()
        
        # Create a temporary plotly express figure to get the layout settings
        import plotly.express as px
        temp_fig = px.scatter(
            selected_umap_df,
            x="UMAP1", y="UMAP2",
            color="file_label",
            title=f"Selected Points ({len(indices)} events)",
            labels={"file_label": "Data File"},
            opacity=0.7
        )
        
        # Use the layout from temp_fig
        fig.update_layout(temp_fig.layout)
        
        # Add traces for each file label
        for label in selected_umap_df['file_label'].unique():
            mask = selected_umap_df['file_label'] == label
            df_subset = selected_umap_df[mask]
            
            marker_color = color_map.get(label, None)  # Get color from map or None
            
            if marker_color:
                # Use color from original figure
                fig.add_trace(go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=marker_color,
                        opacity=0.7
                    ),
                    name=label
                ))
            else:
                # Fallback to auto-assigned color
                fig.add_trace(go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode='markers',
                    marker=dict(size=8, opacity=0.7),
                    name=label
                ))
        
        fig.update_layout(height=600)
        
        # Count points by file
        file_counts = selected_umap_df['file_label'].value_counts().to_dict()
        count_str = "<br>".join([f"{file}: {count} events" for file, count in file_counts.items()])
        debug_text += f"<br>Total selected: {len(selected_umap_df)} events<br>{count_str}"
        
        return fig, debug_text
    
    except Exception as e:
        print(f"Error in Graph 2 callback: {e}")
        import traceback
        traceback.print_exc()
        return {}, f"Error processing selection: {str(e)}"

# Modified to store the subset data used for Graph 3
@app.callback(
    Output('umap-graph-selected-run', 'figure'),
    Output('debug-output-selected-run', 'children'),
    Output('combined-data-store', 'data', allow_duplicate=True),
    Output('umap-quality-metrics-graph3', 'children'),
    Input('run-umap-selected-run', 'n_clicks'),
    State('selected-points-store', 'data'),
    State('combined-data-store', 'data'),
    State('num-neighbors-selected-run', 'value'),
    State('min-dist-selected-run', 'value'),
    State({'type': 'feature-selector-graph3', 'category': ALL}, 'value'),
    State('umap-graph', 'figure'),
    State('metric-selector-graph3', 'value'),
    prevent_initial_call=True
)
def update_umap_selected_run(n_clicks, selectedData, combined_data_json, num_neighbors_sel, min_dist_sel, 
                             selected_features_list_graph3, original_figure, selected_metrics):
    """Re-run UMAP on only the selected points from Graph 1 using Graph 3's feature selection."""
    try:
        # Initialize default return values
        empty_fig = {}
        default_data = combined_data_json or {}
        empty_metrics = []
        
        if not combined_data_json:
            return empty_fig, "No UMAP data available. Run UMAP first.", default_data, empty_metrics
        
        # Load the combined dataframe
        combined_df = pd.read_json(combined_data_json["combined_df"], orient='split').reset_index(drop=True)
        
        # Get features selected for Graph 3
        all_selected_features_graph3 = []
        for features in selected_features_list_graph3:
            if features:  # Only add non-empty lists
                all_selected_features_graph3.extend(features)
        
        if not selectedData:
            return empty_fig, "No points selected. Use the lasso or box select tool on Graph 1.", default_data, empty_metrics
        
        # Initialize variables to hold indices
        indices = []
        debug_text = ""
        
        # Load the UMAP coordinates for reference when finding points
        umap_coords = pd.read_json(combined_data_json["umap_coords"], orient='split').reset_index(drop=True)
        
        # Handle box selection
        if 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            debug_text = f"Box selection for re-run: x range: [{x_range[0]:.2f}, {x_range[1]:.2f}], y range: [{y_range[0]:.2f}, {y_range[1]:.2f}]<br>"
            
            selected_mask = (
                (umap_coords['UMAP1'] >= x_range[0]) & 
                (umap_coords['UMAP1'] <= x_range[1]) & 
                (umap_coords['UMAP2'] >= y_range[0]) & 
                (umap_coords['UMAP2'] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()
        
        # Handle lasso selection
        elif 'lassoPoints' in selectedData:
            debug_text = "Lasso selection for re-run<br>"
            
            # Extract lasso polygon coordinates
            lasso_x = selectedData['lassoPoints']['x']
            lasso_y = selectedData['lassoPoints']['y']
            
            from matplotlib.path import Path
            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            
            # Check which points are within the lasso path
            points_array = np.column_stack([umap_coords['UMAP1'], umap_coords['UMAP2']])
            inside_lasso = lasso_path.contains_points(points_array)
            
            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()
        
        # Handle direct point selection (fallback)
        elif 'points' in selectedData:
            indices = [pt['pointIndex'] for pt in selectedData['points']]
            debug_text = f"Direct point selection for re-run: {len(indices)} points selected<br>"
        
        if not indices:
            return empty_fig, "No valid selection or no points found in selection area.", default_data, empty_metrics
        
        # Extract selected points from the combined dataframe
        selected_df = combined_df.iloc[indices].reset_index(drop=True)
        debug_text += f"Selected data shape: {selected_df.shape}<br>"
        
        # IMPORTANT: Store the selected subset for Graph 3 in the combined_data_json
        # This is the key change to make the Graph 3 re-run work properly
        combined_data_json["graph3_subset"] = selected_df.to_json(date_format='iso', orient='split')
        combined_data_json["graph3_indices"] = indices  # Store the original indices as well
        
        # Use selected features for UMAP if available (from Graph 3 selection)
        if all_selected_features_graph3 and len(all_selected_features_graph3) > 0:
            feature_cols = [col for col in selected_df.columns if col in all_selected_features_graph3]
            if feature_cols:
                debug_text += f"Using selected features for UMAP re-run: {', '.join(feature_cols)}<br>"
                X_selected = selected_df[feature_cols].to_numpy()
            else:
                # Fallback to original momentum columns
                original_cols = [col for col in selected_df.columns if col.startswith('particle_')]
                X_selected = selected_df[original_cols].to_numpy()
                debug_text += "No valid features selected for re-run, using original momentum components.<br>"
        else:
            # Use original momentum columns
            original_cols = [col for col in selected_df.columns if col.startswith('particle_')]
            X_selected = selected_df[original_cols].to_numpy()
            debug_text += "No features selected for re-run, using original momentum components.<br>"
        
        # Count points by file
        file_counts = selected_df['file_label'].value_counts().to_dict()
        count_str = "<br>".join([f"{file}: {count} events" for file, count in file_counts.items()])
        debug_text += f"<br>{count_str}<br>"
        
        # Re-run UMAP on the selected subset
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(num_neighbors_sel),
            min_dist=float(min_dist_sel),
            metric='euclidean',
            random_state=42
        )
        
        # Handle NaN/inf values
        X_selected = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit UMAP
        umap_data_sel = reducer.fit_transform(X_selected)
        
        # Create DataFrame for visualization
        umap_df_sel = pd.DataFrame(umap_data_sel, columns=["UMAP1", "UMAP2"])
        umap_df_sel['file_label'] = selected_df['file_label']
        
        # Also store the Graph 3 UMAP coordinates
        combined_data_json["graph3_umap_coords"] = umap_df_sel.to_json(date_format='iso', orient='split')
        
        # Create the new figure with original colors
        # Extract color information from original figure
        import plotly.graph_objects as go
        
        # Create a color map from the original figure
        color_map = {}
        if original_figure and 'data' in original_figure:
            for trace in original_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    color_map[trace['name']] = trace['marker']['color']
        
        # Create figure with consistent colors
        fig = go.Figure()
        
        # Create a temporary plotly express figure to get the layout settings
        import plotly.express as px
        temp_fig = px.scatter(
            umap_df_sel, 
            x="UMAP1", y="UMAP2", 
            color="file_label",
            title=f"Re-run UMAP on Selected Points (n_neighbors={num_neighbors_sel}, min_dist={min_dist_sel})",
            labels={"file_label": "Data File"},
            opacity=0.7
        )
        
        # Use the layout from temp_fig
        fig.update_layout(temp_fig.layout)
        
        # Add traces for each file label
        for label in umap_df_sel['file_label'].unique():
            mask = umap_df_sel['file_label'] == label
            df_subset = umap_df_sel[mask]
            
            marker_color = color_map.get(label, None)  # Get color from map or None
            
            if marker_color:
                # Use color from original figure
                fig.add_trace(go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=marker_color,
                        opacity=0.7
                    ),
                    name=label
                ))
            else:
                # Fallback to auto-assigned color
                fig.add_trace(go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode='markers',
                    marker=dict(size=8, opacity=0.7),
                    name=label
                ))
        
        fig.update_layout(height=600)
        
        debug_text += f"Re-run UMAP completed on {len(umap_df_sel)} events with colors preserved from original."
        
        # Calculate clustering metrics WITH SMART CONFIDENCE
        metrics_children = []
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            # Get UMAP coordinates for clustering
            X_umap = umap_df_sel[['UMAP1', 'UMAP2']].to_numpy()
            
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
            
            # Collect metrics for confidence calculation
            metrics = {}
            unique_clusters = set(cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = np.sum(cluster_labels == -1)
            noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            
            metrics['noise_ratio'] = noise_ratio
            
            # Only calculate metrics if we have at least 2 clusters
            if n_clusters >= 2:
                # For metrics, we need to exclude noise points (-1)
                mask = cluster_labels != -1
                non_noise_points = np.sum(mask)
                non_noise_clusters = len(set(cluster_labels[mask]))
                
                if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                    if 'silhouette' in selected_metrics:
                        metrics["silhouette"] = silhouette_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    if 'davies_bouldin' in selected_metrics:
                        metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    if 'calinski_harabasz' in selected_metrics:
                        metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    # Add new metrics based on selection
                    if 'hopkins' in selected_metrics:
                        h_stat = hopkins_statistic(X_umap_scaled)
                        metrics["hopkins"] = h_stat
                    
                    if 'stability' in selected_metrics:
                        stability = cluster_stability(X_umap_scaled, best_eps, 5, n_iterations=3)
                        metrics["stability"] = stability
                    
                    # Add physics consistency if selected
                    if 'physics_consistency' in selected_metrics:
                        # Match the cluster labels to the original dataset
                        physics_metrics = physics_cluster_consistency(selected_df, cluster_labels)
                        metrics.update(physics_metrics)
            
            # SMART CONFIDENCE CALCULATION
            confidence_data = calculate_adaptive_confidence_score(
                metrics, 
                clustering_method='dbscan'
            )
            
            # Create the smart confidence UI
            metrics_children = [create_smart_confidence_ui(confidence_data)]
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            metrics_children = [html.Div(f"Error calculating confidence: {str(e)}")]
        
        return fig, debug_text, combined_data_json, metrics_children
    
    except Exception as e:
        print(f"Error in Graph 3 callback: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error re-running UMAP: {str(e)}"
        return {}, error_msg, combined_data_json or {}, []

# Callback to handle the download of selected points from the new graph
@app.callback(
    Output('download-selection-graph3-selection', 'data'),
    Input('save-selection-graph3-selection-btn', 'n_clicks'),
    State('selection-graph3-selection-filename', 'value'),
    State('graph3-selection-umap-store', 'data'),
    prevent_initial_call=True
)
def download_graph3_selection_points(n_clicks, filename, graph3_selection_umap_store):
    """Generate CSV file of points from the Graph 3 selection UMAP for download."""
    if not n_clicks or not filename or not graph3_selection_umap_store:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Load the feature data from the store
        feature_data = pd.read_json(graph3_selection_umap_store["feature_data"], orient='split')
        
        if feature_data.empty:
            raise dash.exceptions.PreventUpdate
        
        # Process the data for saving
        # Remove UMAP coordinates if they exist
        if 'UMAP1' in feature_data.columns:
            feature_data = feature_data.drop(columns=['UMAP1'])
        if 'UMAP2' in feature_data.columns:
            feature_data = feature_data.drop(columns=['UMAP2'])
        
        # The file_label column would have been added during processing, so remove it
        if 'file_label' in feature_data.columns:
            feature_data = feature_data.drop(columns=['file_label'])
        
        # Get only the particle momentum columns (original data format)
        momentum_columns = [col for col in feature_data.columns if col.startswith('particle_')]
        
        # If we have the momentum columns, convert back to original format
        if momentum_columns and len(momentum_columns) == 15:  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                'Px_ion1', 'Py_ion1', 'Pz_ion1',
                'Px_ion2', 'Py_ion2', 'Pz_ion2',
                'Px_neutral', 'Py_neutral', 'Pz_neutral',
                'Px_electron1', 'Py_electron1', 'Pz_electron1',
                'Px_electron2', 'Py_electron2', 'Pz_electron2'
            ]
            
            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = feature_data[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = feature_data
        
        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, 
            f"{filename}.csv",
            index=False
        )
        
    except Exception as e:
        print(f"Error saving Graph 3 selection UMAP data: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate

# Store selected points when selection changes in Graph 3
@app.callback(
    Output('selected-points-run-store', 'data'),
    Output('selected-points-run-info', 'children'),
    Input('umap-graph-selected-run', 'selectedData'),
    prevent_initial_call=True
)
def store_selected_points_run(selectedData):
    """Store the selected points from Graph 3."""
    if not selectedData:
        return [], "No points selected."
    
    selection_type = ""
    num_points = 0
    
    # Handle box selection
    if 'range' in selectedData:
        x_range = selectedData['range']['x']
        y_range = selectedData['range']['y']
        selection_type = f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"
    
    # Handle lasso selection
    elif 'lassoPoints' in selectedData:
        selection_type = "Lasso selection"
    
    # Handle individual point selection
    if 'points' in selectedData:
        num_points = len(selectedData['points'])
    
    info_text = [
        html.Div(f"Selection type: {selection_type}"),
        html.Div(f"Number of points: {num_points}")
    ]
    
    return selectedData, info_text

# Callback to handle the download of selected points from Graph 3
@app.callback(
    Output('download-selection-run', 'data'),
    Input('save-selection-run-btn', 'n_clicks'),
    State('selection-run-filename', 'value'),
    State('selected-points-run-store', 'data'),
    State('combined-data-store', 'data'),
    prevent_initial_call=True
)
def download_selected_points_run(n_clicks, filename, selectedData, combined_data_json):
    """Generate CSV file of selected points from Graph 3 for download in original format."""
    if not n_clicks or not filename or not selectedData or not combined_data_json:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Load the combined dataframe and UMAP coordinates
        combined_df = pd.DataFrame()
        if combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
        
        if combined_df.empty:
            raise dash.exceptions.PreventUpdate
        
        # Get indices of selected points
        indices = []
        
        # Handle box selection
        if 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            
            # We can't directly use combined_data_store because the points in Graph 3
            # have different UMAP coordinates after re-running UMAP
            # Instead, we'll extract the points directly from the selection
            if 'points' in selectedData:
                indices = [pt['pointIndex'] for pt in selectedData['points']]
        
        # Handle lasso selection
        elif 'lassoPoints' in selectedData:
            # Extract directly from points for Graph 3
            if 'points' in selectedData:
                indices = [pt['pointIndex'] for pt in selectedData['points']]
        
        # Handle direct point selection
        elif 'points' in selectedData:
            indices = [pt['pointIndex'] for pt in selectedData['points']]
        
        if not indices:
            raise dash.exceptions.PreventUpdate
        
        # Extract only the selected points from the combined dataframe
        # For Graph 3, we need to be careful because the indices in the re-run
        # correspond to points that were already selected from Graph 1
        # We need to get the original indices from the first selection
        
        # If we have fewer points in combined_df than our indices, it means
        # we're working with a subset already
        if len(indices) <= len(combined_df):
            selected_df = combined_df.iloc[indices].reset_index(drop=True)
        else:
            # This is a fallback if something is wrong with the indices
            # Just take the first N points where N is the number of indices
            selected_df = combined_df.head(min(len(indices), len(combined_df)))
        
        # Remove UMAP coordinates if they exist
        if 'UMAP1' in selected_df.columns:
            selected_df = selected_df.drop(columns=['UMAP1'])
        if 'UMAP2' in selected_df.columns:
            selected_df = selected_df.drop(columns=['UMAP2'])
        
        # Convert back to original format with standardized column names
        # The file_label column would have been added during processing, so remove it
        if 'file_label' in selected_df.columns:
            selected_df = selected_df.drop(columns=['file_label'])
        
        # Get only the particle momentum columns (original data format)
        momentum_columns = [col for col in selected_df.columns if col.startswith('particle_')]
        
        # If we have the momentum columns, convert back to original format
        if momentum_columns and len(momentum_columns) == 15:  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                'Px_ion1', 'Py_ion1', 'Pz_ion1',
                'Px_ion2', 'Py_ion2', 'Pz_ion2',
                'Px_neutral', 'Py_neutral', 'Pz_neutral',
                'Px_electron1', 'Py_electron1', 'Pz_electron1',
                'Px_electron2', 'Py_electron2', 'Pz_electron2'
            ]
            
            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = selected_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = selected_df
        
        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, 
            f"{filename}.csv",
            index=False
        )
        
    except Exception as e:
        print(f"Error saving Graph 3 selection: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate

# Callback for Graph 3 Selected Points
@app.callback(
    Output('umap-graph-selected-run-only', 'figure'),
    Output('debug-output-selected-run-only', 'children'),
    Output('graph3-selection-info-viz', 'children'),
    Input('show-selected-run', 'n_clicks'),
    State('selected-points-run-store', 'data'),
    State('umap-graph-selected-run', 'figure'),  # Graph 3 figure for reference
    State('umap-graph', 'figure'),  # Graph 1 figure for color consistency
    State('combined-data-store', 'data'),  # Add this to access graph3_umap_coords
    prevent_initial_call=True
)
def update_umap_selected_run_only(n_clicks, selectedData, graph3_figure, graph1_figure, combined_data_json):
    """Display the selected points from Graph 3 using geometric selection approach."""
    try:
        debug_text = ""
        
        # Validate inputs
        if not selectedData:
            return {}, "No points selected. Use the lasso or box select tool on Graph 3.", "No selection made yet."
        
        # Check if we have Graph 3 UMAP coordinates in the data store
        if 'graph3_umap_coords' not in combined_data_json or combined_data_json['graph3_umap_coords'] == "{}":
            return {}, "Graph 3 UMAP coordinates not found. Please re-run Graph 3 first.", "No Graph 3 data available."
        
        # Load the Graph 3 UMAP coordinates
        graph3_umap_coords = pd.read_json(combined_data_json['graph3_umap_coords'], orient='split')
        debug_text += f"Found Graph 3 UMAP coordinates with {len(graph3_umap_coords)} points.<br>"
        
        # Process the selection using geometric operations - similar to Graph 1
        indices = []
        
        # Handle box selection
        if 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            debug_text += f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]<br>"
            
            # Find points inside the box
            selected_mask = (
                (graph3_umap_coords['UMAP1'] >= x_range[0]) & 
                (graph3_umap_coords['UMAP1'] <= x_range[1]) & 
                (graph3_umap_coords['UMAP2'] >= y_range[0]) & 
                (graph3_umap_coords['UMAP2'] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()
            debug_text += f"Found {len(indices)} points in box selection.<br>"
        
        # Handle lasso selection
        elif 'lassoPoints' in selectedData:
            debug_text += "Lasso selection detected.<br>"
            
            # Extract lasso polygon coordinates
            lasso_x = selectedData['lassoPoints']['x']
            lasso_y = selectedData['lassoPoints']['y']
            
            from matplotlib.path import Path
            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            
            # Check which points are within the lasso path
            points_array = np.column_stack([graph3_umap_coords['UMAP1'], graph3_umap_coords['UMAP2']])
            inside_lasso = lasso_path.contains_points(points_array)
            
            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()
            debug_text += f"Found {len(indices)} points in lasso selection.<br>"
        
        # As a fallback, try to use points directly if available
        elif 'points' in selectedData and selectedData['points']:
            debug_text += "Direct point selection detected.<br>"
            
            # For direct point selection, we'll create a temporary dataframe with the points from the selection
            points = selectedData['points']
            debug_text += f"Found {len(points)} points in direct selection.<br>"
            
            # Extract point data directly
            # This approach ensures we display exactly what was selected visually
            selected_data = []
            for point in points:
                x = point.get('x', 0)
                y = point.get('y', 0)
                
                # Try to get the label from different sources
                label = "Unknown"
                
                # From curve number
                curve_num = point.get('curveNumber', -1)
                if curve_num >= 0 and graph3_figure and 'data' in graph3_figure and curve_num < len(graph3_figure['data']):
                    curve = graph3_figure['data'][curve_num]
                    if 'name' in curve:
                        label = curve['name']
                        if ' (' in label:
                            label = label.split(' (')[0]
                
                # From customdata if available
                if 'customdata' in point and point['customdata']:
                    if isinstance(point['customdata'], list) and len(point['customdata']) > 0:
                        # If we have customdata with label information
                        if isinstance(point['customdata'][0], str):
                            label = point['customdata'][0]
                
                selected_data.append({
                    'UMAP1': x,
                    'UMAP2': y,
                    'file_label': label
                })
            
            # Create a temporary selection dataframe
            temp_df = pd.DataFrame(selected_data)
            debug_text += f"Created temporary dataframe with {len(temp_df)} points.<br>"
            
            # For consistency with other selection methods, we'll still try to find indices
            # in graph3_umap_coords that match these points as closely as possible
            for i, row in temp_df.iterrows():
                # Find the closest point in graph3_umap_coords
                distances = ((graph3_umap_coords['UMAP1'] - row['UMAP1'])**2 + 
                            (graph3_umap_coords['UMAP2'] - row['UMAP2'])**2)
                closest_idx = distances.idxmin()
                indices.append(closest_idx)
            
            debug_text += f"Mapped {len(indices)} points to Graph 3 UMAP coordinates.<br>"
        
        if not indices:
            return {}, "No valid points found in the selection.", "No valid selection."
        
        # Make sure indices are valid
        valid_indices = [i for i in indices if 0 <= i < len(graph3_umap_coords)]
        if len(valid_indices) < len(indices):
            debug_text += f"Warning: {len(indices) - len(valid_indices)} invalid indices removed.<br>"
            indices = valid_indices
        
        if not indices:
            return {}, "No valid indices after validation.", "No valid selection."
        
        # Get the selected points from the Graph 3 UMAP coordinates
        selected_df = graph3_umap_coords.iloc[indices].copy().reset_index(drop=True)
        debug_text += f"Selected {len(selected_df)} points from Graph 3 UMAP coordinates.<br>"
        
        # Extract color information from both figures for consistent visualization
        color_map = {}
        
        # Get colors from Graph 1 figure
        if graph1_figure and 'data' in graph1_figure:
            for trace in graph1_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    name = trace['name']
                    if ' (' in name:  # Clean label if it contains count
                        name = name.split(' (')[0]
                    color_map[name] = trace['marker']['color']
        
        # Get colors from Graph 3 figure (prioritizing these if there's overlap)
        if graph3_figure and 'data' in graph3_figure:
            for trace in graph3_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    name = trace['name']
                    if ' (' in name:  # Clean label if it contains count
                        name = name.split(' (')[0]
                    color_map[name] = trace['marker']['color']
        
        # Create the visualization
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Add traces for each label
        for label in selected_df['file_label'].unique():
            mask = selected_df['file_label'] == label
            df_subset = selected_df[mask]
            
            # Get color for this label
            color = color_map.get(label, None)
            
            fig.add_trace(go.Scatter(
                x=df_subset['UMAP1'],
                y=df_subset['UMAP2'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                name=f"{label} ({len(df_subset)} pts)"
            ))
        
        # Update figure properties
        fig.update_layout(
            height=600,
            title=f"Selected Points from Graph 3 ({len(selected_df)} events)",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data File"
        )
        
        # Count points by file for information panel
        file_counts = selected_df['file_label'].value_counts().to_dict()
        
        # Create info text
        info_text = [
            html.Div(f"Total selected points: {len(selected_df)}"),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div([html.Div(f"{file}: {count} events") for file, count in file_counts.items()], 
                    style={"marginLeft": "10px"})
        ]
        
        return fig, debug_text, info_text
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return {}, f"Error processing Graph 3 selection: {str(e)}<br><pre>{trace}</pre>", f"Error: {str(e)}"

@app.callback(
    Output('custom-feature-plot', 'figure'),
    Output('debug-output-custom-plot', 'children'),
    Input('plot-custom-features', 'n_clicks'),
    State('x-axis-feature', 'value'),
    State('y-axis-feature', 'value'),
    State('selection-source', 'value'),
    State('selected-points-store', 'data'),  # Graph 1 selection
    State('selected-points-run-store', 'data'),  # Graph 3 selection
    State('combined-data-store', 'data'),
    State('umap-graph', 'figure'),
    State('umap-graph-selected-run', 'figure'),
    prevent_initial_call=True
)
def update_custom_feature_plot(n_clicks, x_feature, y_feature, selection_source, 
                               graph1_selection, graph3_selection, combined_data_json, 
                               graph1_figure, graph3_figure):
    """Create a custom scatter plot with proper selection handling for both Graph 1 and Graph 3."""
    try:
        # Initialize debug info
        debug_text = []
        debug_text.append(f"X-axis feature: {x_feature}")
        debug_text.append(f"Y-axis feature: {y_feature}")
        debug_text.append(f"Selection source: {selection_source}")
        
        if not x_feature or not y_feature:
            return {}, "Please select both X and Y axis features."
        
        # Step 1: Load datasets
        # ------------------------------------------
        
        # Load combined dataset (original data)
        combined_df = None
        if combined_data_json and "combined_df" in combined_data_json and combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
            debug_text.append(f"Loaded combined dataset with {len(combined_df)} rows")
        else:
            return {}, "No combined dataset available. Please run UMAP first."
        
        # Load Graph 3 subset
        graph3_subset_df = None
        if combined_data_json and "graph3_subset" in combined_data_json and combined_data_json["graph3_subset"] != "{}":
            graph3_subset_df = pd.read_json(combined_data_json["graph3_subset"], orient='split')
            debug_text.append(f"Loaded Graph 3 subset with {len(graph3_subset_df)} rows")
        
        # Load UMAP coordinates
        umap_coords = None
        if "umap_coords" in combined_data_json and combined_data_json["umap_coords"] != "{}":
            umap_coords = pd.read_json(combined_data_json["umap_coords"], orient='split')
        
        # Load Graph 3 UMAP coordinates
        graph3_umap_coords = None
        if "graph3_umap_coords" in combined_data_json and combined_data_json["graph3_umap_coords"] != "{}":
            graph3_umap_coords = pd.read_json(combined_data_json["graph3_umap_coords"], orient='split')
        
        # Verify features exist in datasets
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return {}, f"Features {x_feature} or {y_feature} not found in dataset."
        
        # Step 2: Process Graph 1 Selection
        # ------------------------------------------
        df_to_plot_graph1 = pd.DataFrame(columns=['x', 'y', 'label', 'source'])
        
        if selection_source in ['graph2', 'both'] and graph1_selection and umap_coords is not None:
            debug_text.append("\nProcessing Graph 1 selection...")
            
            # Extract indices
            indices = []
            
            # Handle different selection types (box, lasso, direct)
            if 'range' in graph1_selection:
                # Box selection
                x_range = graph1_selection['range']['x']
                y_range = graph1_selection['range']['y']
                
                # Find points inside the box
                selected_mask = (
                    (umap_coords['UMAP1'] >= x_range[0]) & 
                    (umap_coords['UMAP1'] <= x_range[1]) & 
                    (umap_coords['UMAP2'] >= y_range[0]) & 
                    (umap_coords['UMAP2'] <= y_range[1])
                )
                indices = np.where(selected_mask)[0].tolist()
                
                debug_text.append(f"Box selection with {len(indices)} points")
                
            elif 'lassoPoints' in graph1_selection:
                # Lasso selection
                from matplotlib.path import Path
                
                # Create path from lasso points
                lasso_x = graph1_selection['lassoPoints']['x']
                lasso_y = graph1_selection['lassoPoints']['y']
                lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
                
                # Check which points are inside the lasso
                points_array = np.column_stack([umap_coords['UMAP1'], umap_coords['UMAP2']])
                inside_lasso = lasso_path.contains_points(points_array)
                indices = np.where(inside_lasso)[0].tolist()
                
                debug_text.append(f"Lasso selection with {len(indices)} points")
                
            elif 'points' in graph1_selection:
                # Direct point selection
                indices = [p.get('pointIndex', -1) for p in graph1_selection['points']]
                indices = [i for i in indices if i >= 0]
                debug_text.append(f"Direct selection with {len(indices)} points")
            
            # Extract selected rows from combined dataset
            if indices and len(indices) > 0:
                # Ensure indices are valid
                valid_indices = [i for i in indices if 0 <= i < len(combined_df)]
                
                if valid_indices:
                    # Extract feature values
                    selected_rows = combined_df.iloc[valid_indices]
                    
                    # Get label distribution for debugging
                    label_counts = selected_rows['file_label'].value_counts().to_dict()
                    debug_text.append("Graph 1 selection label distribution:")
                    for label, count in sorted(label_counts.items()):
                        debug_text.append(f"- {label}: {count} points")
                    
                    # Create dataframe for plotting
                    df_to_plot_graph1 = pd.DataFrame({
                        'x': selected_rows[x_feature],
                        'y': selected_rows[y_feature],
                        'label': selected_rows['file_label'],
                        'source': 'Graph 1'
                    })
                    
                    debug_text.append(f"Extracted {len(df_to_plot_graph1)} points from Graph 1 selection")
                else:
                    debug_text.append("No valid indices found in Graph 1 selection")
        
        # Step 3: Process Graph 3 Selection
        # ------------------------------------------
        df_to_plot_graph3 = pd.DataFrame(columns=['x', 'y', 'label', 'source'])
        
        if selection_source in ['graph3', 'both'] and graph3_selection and graph3_subset_df is not None and graph3_umap_coords is not None:
            debug_text.append("\nProcessing Graph 3 selection...")
            
            # Extract indices using the same approach as in update_umap_graph3_selection
            indices = []
            
            # Handle box selection
            if 'range' in graph3_selection:
                x_range = graph3_selection['range']['x']
                y_range = graph3_selection['range']['y']
                debug_text.append(f"Box selection in Graph 3: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]")
                
                # Find points inside the box
                selected_mask = (
                    (graph3_umap_coords['UMAP1'] >= x_range[0]) & 
                    (graph3_umap_coords['UMAP1'] <= x_range[1]) & 
                    (graph3_umap_coords['UMAP2'] >= y_range[0]) & 
                    (graph3_umap_coords['UMAP2'] <= y_range[1])
                )
                indices = np.where(selected_mask)[0].tolist()
                debug_text.append(f"Found {len(indices)} points in box selection from Graph 3")
            
            # Handle lasso selection
            elif 'lassoPoints' in graph3_selection:
                debug_text.append("Lasso selection in Graph 3")
                
                # Extract lasso polygon coordinates
                lasso_x = graph3_selection['lassoPoints']['x']
                lasso_y = graph3_selection['lassoPoints']['y']
                
                from matplotlib.path import Path
                # Create a Path object from the lasso points
                lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
                
                # Check which points are within the lasso path
                points_array = np.column_stack([graph3_umap_coords['UMAP1'], graph3_umap_coords['UMAP2']])
                inside_lasso = lasso_path.contains_points(points_array)
                
                # Get indices of points inside the lasso
                indices = np.where(inside_lasso)[0].tolist()
                debug_text.append(f"Found {len(indices)} points in lasso selection from Graph 3")
            
            # Handle direct point selection
            elif 'points' in graph3_selection and graph3_selection['points']:
                points = graph3_selection['points']
                debug_text.append(f"Found {len(points)} points in Graph 3 selection")
                
                # For direct point selection, extract indices directly
                for point in points:
                    idx = point.get('pointIndex', -1)
                    if 0 <= idx < len(graph3_subset_df):
                        indices.append(idx)
                
                debug_text.append(f"Extracted {len(indices)} valid indices from Graph 3 points")
            
            if indices and len(indices) > 0:
                # Ensure indices are valid
                valid_indices = [i for i in indices if 0 <= i < len(graph3_subset_df)]
                
                if valid_indices:
                    # Extract feature values
                    selected_rows = graph3_subset_df.iloc[valid_indices]
                    
                    # Get label distribution for debugging
                    label_counts = selected_rows['file_label'].value_counts().to_dict()
                    debug_text.append("Graph 3 selection label distribution:")
                    for label, count in sorted(label_counts.items()):
                        debug_text.append(f"- {label}: {count} points")
                    
                    # Create dataframe for plotting
                    df_to_plot_graph3 = pd.DataFrame({
                        'x': selected_rows[x_feature],
                        'y': selected_rows[y_feature],
                        'label': selected_rows['file_label'],
                        'source': 'Graph 3'
                    })
                    
                    debug_text.append(f"Extracted {len(df_to_plot_graph3)} points from Graph 3 selection")
                else:
                    debug_text.append("No valid indices found in Graph 3 selection")
        
        # Step 4: Combine data sources based on selection_source
        # ------------------------------------------------------------
        df_to_plot = pd.DataFrame(columns=['x', 'y', 'label', 'source'])
        
        if selection_source == 'graph2':
            df_to_plot = df_to_plot_graph1
        elif selection_source == 'graph3':
            df_to_plot = df_to_plot_graph3
        elif selection_source == 'both':
            df_to_plot = pd.concat([df_to_plot_graph1, df_to_plot_graph3], ignore_index=True)
        
        if df_to_plot.empty:
            return {}, "No points to plot with the selected criteria."
        
        # Step 5: Create color map from both figures for consistency
        # ------------------------------------------------------------
        color_map = {}
        
        # Get colors from Graph 1 figure
        if graph1_figure and 'data' in graph1_figure:
            for trace in graph1_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    name = trace['name']
                    if ' (' in name:  # Clean label if it contains count
                        name = name.split(' (')[0]
                    color_map[name] = trace['marker']['color']
        
        # Get additional colors from Graph 3 figure
        if graph3_figure and 'data' in graph3_figure:
            for trace in graph3_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    name = trace['name']
                    if ' (' in name:  # Clean label if it contains count
                        name = name.split(' (')[0]
                    if name not in color_map:  # Don't overwrite existing colors
                        color_map[name] = trace['marker']['color']
        
        # Step 6: Create the plot
        # ------------------------------------------------------------
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Use different symbols for different sources if plotting both
        if selection_source == 'both':
            symbols = {'Graph 1': 'circle', 'Graph 3': 'square'}
            
            # Add traces by source and label
            for source in df_to_plot['source'].unique():
                for label in df_to_plot[df_to_plot['source'] == source]['label'].unique():
                    mask = (df_to_plot['source'] == source) & (df_to_plot['label'] == label)
                    points = df_to_plot[mask]
                    
                    fig.add_trace(go.Scatter(
                        x=points['x'],
                        y=points['y'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_map.get(label),
                            symbol=symbols.get(source, 'circle'),
                            opacity=0.7
                        ),
                        name=f"{label} ({source}, {len(points)} pts)"
                    ))
        else:
            # Add one trace per label
            for label in df_to_plot['label'].unique():
                points = df_to_plot[df_to_plot['label'] == label]
                
                fig.add_trace(go.Scatter(
                    x=points['x'],
                    y=points['y'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color_map.get(label),
                        opacity=0.7
                    ),
                    name=f"{label} ({len(points)} pts)"
                ))
        
        # Update layout
        fig.update_layout(
            height=600,
            title=f"Custom Feature Plot: {x_feature} vs {y_feature}",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title="Data Source"
        )
        
        return fig, "<br>".join(debug_text)
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return {}, f"Error: {str(e)}<br><pre>{trace}</pre>"

# Callback for training autoencoder and running UMAP on latent space
@app.callback(
    Output('autoencoder-umap-graph', 'figure'),
    Output('autoencoder-debug-output', 'children'),
    Output('autoencoder-latent-store', 'data'),
    Output('umap-quality-metrics-autoencoder', 'children'),
    Output('feature-importance-container', 'children'),
    Input('train-autoencoder', 'n_clicks'),
    Input('run-umap-latent', 'n_clicks'),
    State('autoencoder-latent-dim', 'value'),
    State('autoencoder-epochs', 'value'),
    State('autoencoder-batch-size', 'value'),
    State('autoencoder-learning-rate', 'value'),
    State('autoencoder-data-source', 'value'),
    State('selected-points-store', 'data'),
    State('selected-points-run-store', 'data'),
    State('combined-data-store', 'data'),
    State('autoencoder-umap-neighbors', 'value'),
    State('autoencoder-umap-min-dist', 'value'),
    State({'type': 'feature-selector-autoencoder', 'category': ALL}, 'value'),
    State('umap-graph', 'figure'),
    State('autoencoder-latent-store', 'data'),
    State('metric-selector-autoencoder', 'value'),
    prevent_initial_call=True
)
def train_autoencoder_and_run_umap(train_clicks, umap_clicks, latent_dim, epochs, batch_size, 
                                  learning_rate, data_source, graph1_selection, graph3_selection,
                                  combined_data_json, n_neighbors, min_dist, selected_features_list,
                                  original_figure, latent_store, selected_metrics):
    """Train autoencoder and run UMAP on latent space."""
    try:
        # Import required libraries
        import numpy as np
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return {}, "No action triggered.", {}, [], []
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Initialize debug info
        debug_text = ["Starting callback execution..."]
        
        # Initialize feature importance container
        feature_importance_container = []
        
        # Load the combined dataframe - do this regardless of training or not
        if combined_data_json and "combined_df" in combined_data_json and combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
            debug_text.append(f"Loaded combined dataset with {len(combined_df)} rows")
        else:
            return {}, "No combined dataset available. Please run UMAP first.", {}, [], []
        
        # Check if we need to train the autoencoder or just run UMAP on existing latent space
        if trigger_id == 'run-umap-latent' and latent_store:
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
            umap_coords = None
            if "umap_coords" in combined_data_json and combined_data_json["umap_coords"] != "{}":
                umap_coords = pd.read_json(combined_data_json["umap_coords"], orient='split')
            
            # Collect all selected features for autoencoder
            all_selected_features = []
            for features in selected_features_list:
                if features:  # Only add non-empty lists
                    all_selected_features.extend(features)
            
            if not all_selected_features:
                # Use particle momentum as default if nothing is selected
                all_selected_features = [col for col in combined_df.columns if col.startswith('particle_')]
                debug_text.append(f"No features selected, using {len(all_selected_features)} default momentum features")
            else:
                debug_text.append(f"Using {len(all_selected_features)} selected features")
            
            # Default to all data
            df_for_training = combined_df.copy()
            labels = combined_df['file_label'].copy()
            debug_text.append(f"Using all {len(df_for_training)} rows for training")
            
            # Extract only the selected features
            feature_cols = [col for col in df_for_training.columns if col in all_selected_features]
            
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
            model = DeepAutoencoder(input_dim=input_dim, latent_dim=int(latent_dim)).to(device)
            
            # Initialize optimizer and loss function
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))
            
            # Training loop
            num_epochs = int(epochs)
            losses = []
            
            debug_text.append(f"Starting training for {num_epochs} epochs...")
            app.current_epoch = 0
            app.total_epochs = num_epochs

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

                app.current_epoch = epoch + 1
                
                # Only log some epochs to avoid overcrowding
                if epoch == 0 or epoch == num_epochs-1 or (epoch+1) % max(1, num_epochs//5) == 0:
                    debug_text.append(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            debug_text.append("Training complete!")
            
            # Extract latent representations
            model.eval()
            with torch.no_grad():
                latent_features = model.encoder(feature_tensor.to(device)).cpu().numpy()
            
            # Create dataframe with latent features
            latent_df = pd.DataFrame(latent_features, columns=[f"Latent_{i}" for i in range(latent_dim)])
            latent_df['file_label'] = labels.values
            
            # Store the latent features for future use
            latent_store = {
                'latent_features': latent_df.to_json(date_format='iso', orient='split'),
                'feature_cols': feature_cols,
                'latent_dim': latent_dim
            }
            
            debug_text.append(f"Extracted {len(latent_df)} latent representations with dimension {latent_dim}")
            
            # If training completed but we don't want to run UMAP yet
            if trigger_id == 'train-autoencoder':
                placeholder_fig = {
                    'data': [],
                    'layout': {
                        'title': 'Training complete! Click "Run UMAP on Latent Space" to visualize',
                        'xaxis': {'title': 'UMAP1'},
                        'yaxis': {'title': 'UMAP2'},
                        'height': 600
                    }
                }
                return placeholder_fig, "<br>".join(debug_text), latent_store, [], []
        
        else:
            # Load latent features from store
            try:
                latent_df = pd.read_json(latent_store['latent_features'], orient='split')
                feature_cols = latent_store['feature_cols']
                latent_dim = latent_store['latent_dim']
                debug_text.append(f"Loaded {len(latent_df)} latent representations with dimension {latent_dim}")
            except Exception as e:
                return {}, f"Error loading latent features: {str(e)}", {}, [], []
        
        # Run UMAP on the latent space
        debug_text.append(f"Running UMAP on latent space (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        
        # Prepare data for UMAP
        X_latent = latent_df[[f"Latent_{i}" for i in range(latent_dim)]].to_numpy()
        
        # Run UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric='euclidean',
            random_state=42
        )
        
        # Fit UMAP
        umap_data = reducer.fit_transform(X_latent)
        
        # Create DataFrame for visualization
        umap_df = pd.DataFrame(umap_data, columns=["UMAP1", "UMAP2"])
        umap_df['file_label'] = latent_df['file_label'].values
        
        debug_text.append(f"UMAP transformation complete with {len(umap_df)} points")
        
        # Extract color information from original figure
        import plotly.graph_objects as go
        
        # Create a color map from the original figure
        color_map = {}
        if original_figure and 'data' in original_figure:
            for trace in original_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    # Clean the label if it contains point count
                    clean_name = trace['name']
                    if ' (' in clean_name:
                        clean_name = clean_name.split(' (')[0]
                    color_map[clean_name] = trace['marker']['color']
        
        # Create figure with consistent colors
        fig = go.Figure()
        
        # Add traces for each file label
        for label in umap_df['file_label'].unique():
            mask = umap_df['file_label'] == label
            df_subset = umap_df[mask]
            
            # Get color from original figure if available
            color = color_map.get(label, None)
            
            fig.add_trace(go.Scatter(
                x=df_subset["UMAP1"],
                y=df_subset["UMAP2"],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                name=f"{label} ({len(df_subset)} pts)"
            ))
        
        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"UMAP of Autoencoder Latent Space (dim={latent_dim}, n_neighbors={n_neighbors}, min_dist={min_dist})",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data File"
        )
        
        # Add latent UMAP coordinates to the latent store
        latent_store['umap_coords'] = umap_df.to_json(date_format='iso', orient='split')
        
        # Calculate mutual information between original features and UMAP dimensions
        debug_text.append("Calculating mutual information between original features and UMAP dimensions...")
        
        # We need to get back to the original features that were used
        try:
            # Get the original feature columns that were used to train the autoencoder
            feature_cols = latent_store.get('feature_cols', [])
            
            if feature_cols and len(feature_cols) > 0:
                # Check if the combined_df has these feature columns
                missing_cols = [col for col in feature_cols if col not in combined_df.columns]
                if missing_cols:
                    debug_text.append(f"Warning: Some feature columns are missing from the data: {missing_cols}")
                    # Only use the columns that exist
                    feature_cols = [col for col in feature_cols if col in combined_df.columns]
                
                if not feature_cols:
                    raise ValueError("No valid feature columns found for MI calculation")
                
                # Get the data source that was used - use all data
                original_features_df = combined_df[feature_cols]
                
                # Handle NaN/inf values in the original features
                original_features_np = np.nan_to_num(original_features_df.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get the UMAP coordinates for mutual information calculation
                umap_coords_np = umap_df[['UMAP1', 'UMAP2']].to_numpy()
                
                # Ensure original_features_np and umap_coords_np have the same length
                if len(original_features_np) != len(umap_coords_np):
                    debug_text.append(f"Warning: Feature matrix ({len(original_features_np)} rows) and UMAP coordinates ({len(umap_coords_np)} rows) have different lengths")
                    min_length = min(len(original_features_np), len(umap_coords_np))
                    original_features_np = original_features_np[:min_length]
                    umap_coords_np = umap_coords_np[:min_length]
                
                # Compute mutual information between original features and each UMAP dimension
                umap_mi_scores = {}
                for i, dim in enumerate(["UMAP1", "UMAP2"]):
                    umap_mi_scores[dim] = mutual_info_regression(original_features_np, umap_coords_np[:, i])
                
                # Average MI scores across both UMAP dimensions
                avg_mi_scores = np.mean(list(umap_mi_scores.values()), axis=0)
                
                # Create a dictionary mapping each feature name to its average MI score
                mi_scores_dict = dict(zip(feature_cols, avg_mi_scores))
                
                # Sort features by MI score (highest first)
                sorted_features = sorted(mi_scores_dict, key=mi_scores_dict.get, reverse=True)
                
                # Add top contributing features to debug text
                debug_text.append("Top Features Contributing to Latent Space Clustering:")
                for feature in sorted_features[:10]:
                    debug_text.append(f"{feature}: {mi_scores_dict[feature]:.4f}")
                
                # Also calculate correlation between original features and UMAP dimensions
                # This provides a simpler linear relationship measure to compare with MI
                corr_scores = {}
                for i, dim in enumerate(["UMAP1", "UMAP2"]):
                    corr_scores[dim] = []
                    for j, feature in enumerate(feature_cols):
                        corr = np.corrcoef(original_features_np[:, j], umap_coords_np[:, i])[0, 1]
                        corr_scores[dim].append(corr)
                
                # Calculate average absolute correlation across dimensions
                avg_corr_scores = np.mean([np.abs(corr_scores["UMAP1"]), np.abs(corr_scores["UMAP2"])], axis=0)
                corr_scores_dict = dict(zip(feature_cols, avg_corr_scores))
                
                # Sort features by correlation score (highest first)
                sorted_features_corr = sorted(corr_scores_dict, key=corr_scores_dict.get, reverse=True)
                
                # Add top contributing features by correlation to debug text
                debug_text.append("Top Features by Correlation with UMAP Dimensions:")
                for feature in sorted_features_corr[:10]:
                    debug_text.append(f"{feature}: {corr_scores_dict[feature]:.4f}")
                    
                # Store MI and correlation scores in the latent store for potential future use
                latent_store['mi_scores'] = mi_scores_dict
                latent_store['corr_scores'] = corr_scores_dict
                
                # Create feature importance UI
                mi_ui = html.Div([
                    html.H4("Feature Importance Analysis", style={"fontSize": "14px", "color": "#2e7d32"}),
                    
                    # Summary section - top features
                    html.Div([
                        html.Div([
                            html.H5("Top Features by Mutual Information:", style={"fontSize": "13px", "marginBottom": "5px"}),
                            html.Div([
                                html.Div([
                                    html.Span(f"{i+1}. {feature}: ", style={"fontWeight": "bold"}),
                                    html.Span(f"{mi_scores_dict[feature]:.4f}")
                                ]) for i, feature in enumerate(sorted_features[:10])
                            ], style={"marginLeft": "10px", "fontSize": "12px"})
                        ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
                        
                        html.Div([
                            html.H5("Top Features by Correlation:", style={"fontSize": "13px", "marginBottom": "5px"}),
                            html.Div([
                                html.Div([
                                    html.Span(f"{i+1}. {feature}: ", style={"fontWeight": "bold"}),
                                    html.Span(f"{corr_scores_dict[feature]:.4f}")
                                ]) for i, feature in enumerate(sorted_features_corr[:10])
                            ], style={"marginLeft": "10px", "fontSize": "12px"})
                        ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"})
                    ], style={"display": "flex", "flexWrap": "wrap"}),
                    
                    # Explanation section
                    html.Div([
                        html.Details([
                            html.Summary("What does this mean?", style={"cursor": "pointer", "color": "#2e7d32"}),
                            html.Div([
                                html.P("• Mutual Information (MI) measures nonlinear relationships between features and UMAP dimensions"),
                                html.P("• Correlation measures linear relationships only"),
                                html.P("• Higher values indicate features that strongly influence the latent space clustering"),
                                html.P("• Features appearing high in both metrics have strong overall influence"),
                                html.P("• Features high in MI but low in correlation have primarily nonlinear influence")
                            ], style={"fontSize": "11px", "paddingLeft": "10px"})
                        ])
                    ], style={"marginTop": "10px"})
                ], style={"backgroundColor": "#f1f8e9", "padding": "10px", "borderRadius": "5px", "marginTop": "15px"})
                
                # Add the feature importance UI to the container
                feature_importance_container = [mi_ui]
                
            else:
                debug_text.append("No feature columns available for mutual information calculation")
                feature_importance_container = [html.Div("No feature columns available for mutual information calculation")]
        
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            debug_text.append(f"Error calculating mutual information: {str(e)}")
            print(f"Error calculating mutual information: {str(e)}\n{trace}")
            feature_importance_container = [html.Div(f"Error calculating feature importance: {str(e)}")]
            
        # Calculate clustering metrics
        metrics_children = []
        try:
            # Get UMAP coordinates for clustering
            X_umap = umap_df[['UMAP1', 'UMAP2']].to_numpy()
            
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
            noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            
            metrics = {}
            
            # Only calculate metrics if we have at least 2 clusters
            if n_clusters >= 2:
                # For metrics, we need to exclude noise points (-1)
                mask = cluster_labels != -1
                non_noise_points = np.sum(mask)
                non_noise_clusters = len(set(cluster_labels[mask]))
                
                if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                    if 'silhouette' in selected_metrics:
                        metrics["silhouette"] = silhouette_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    if 'davies_bouldin' in selected_metrics:
                        metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    if 'calinski_harabasz' in selected_metrics:
                        metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled[mask], cluster_labels[mask])
                    
                    # Add new metrics based on selection
                    if 'hopkins' in selected_metrics:
                        h_stat = hopkins_statistic(X_umap_scaled)
                        metrics["hopkins"] = h_stat
                    
                    if 'stability' in selected_metrics:
                        stability = cluster_stability(X_umap_scaled, best_eps, 5, n_iterations=3)
                        metrics["stability"] = stability
                    
                    metrics["note"] = "Metrics calculated excluding noise points"
                else:
                    metrics["note"] = "Not enough valid points for metrics"
            else:
                # Try KMeans as fallback
                from sklearn.cluster import KMeans
                
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                fallback_labels = kmeans.fit_predict(X_umap_scaled)
                
                if 'silhouette' in selected_metrics:
                    metrics["silhouette"] = silhouette_score(X_umap_scaled, fallback_labels)
                
                if 'davies_bouldin' in selected_metrics:
                    metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled, fallback_labels)
                
                if 'calinski_harabasz' in selected_metrics:
                    metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled, fallback_labels)
                
                if 'hopkins' in selected_metrics:
                    h_stat = hopkins_statistic(X_umap_scaled)
                    metrics["hopkins"] = h_stat
                
                metrics["note"] = "DBSCAN found no clusters, metrics based on KMeans fallback"
            
            # Create UI elements for the metrics
            metrics_children = [
                html.H4("Clustering Quality Metrics (DBSCAN)", style={"fontSize": "14px", "marginBottom": "5px"}),
                html.Div([
                    # Existing metrics
                    html.Div([
                        html.Span("Estimated Clusters: ", style={"fontWeight": "bold"}),
                        html.Span(f"{n_clusters}")
                    ]),
                    html.Div([
                        html.Span("Noise Points: ", style={"fontWeight": "bold"}),
                        html.Span(f"{n_noise} ({noise_ratio:.1%})")
                    ]),
                    html.Div([
                        html.Span("DBSCAN eps: ", style={"fontWeight": "bold"}),
                        html.Span(f"{best_eps:.3f}")
                    ]),
                    
                    # Basic metrics (existing)
                    html.Div([
                        html.Span("Silhouette Score: ", style={"fontWeight": "bold"}),
                        html.Span(f"{metrics.get('silhouette', 'N/A'):.4f}", 
                                style={"color": "green" if metrics.get('silhouette', 0) > 0.5 else 
                                       "orange" if metrics.get('silhouette', 0) > 0.25 else "red"})
                    ]) if 'silhouette' in metrics else None,
                    
                    html.Div([
                        html.Span("Davies-Bouldin Index: ", style={"fontWeight": "bold"}),
                        html.Span(f"{metrics.get('davies_bouldin', 'N/A'):.4f}",
                                style={"color": "green" if metrics.get('davies_bouldin', float('inf')) < 0.8 else 
                                       "orange" if metrics.get('davies_bouldin', float('inf')) < 1.5 else "red"})
                    ]) if 'davies_bouldin' in metrics else None,
                    
                    html.Div([
                        html.Span("Calinski-Harabasz Index: ", style={"fontWeight": "bold"}),
                        html.Span(f"{metrics.get('calinski_harabasz', 'N/A'):.1f}",
                                style={"color": "green" if metrics.get('calinski_harabasz', 0) > 100 else 
                                       "orange" if metrics.get('calinski_harabasz', 0) > 50 else "red"})
                    ]) if 'calinski_harabasz' in metrics else None,
                    
                    # New metrics
                    html.Div([
                        html.Span("Hopkins Statistic: ", style={"fontWeight": "bold"}),
                        html.Span(f"{metrics.get('hopkins', 'N/A'):.4f}",
                                 style={"color": "green" if metrics.get('hopkins', 0) > 0.75 else 
                                        "orange" if metrics.get('hopkins', 0) > 0.6 else "red"})
                    ]) if 'hopkins' in metrics else None,
                    
                    html.Div([
                        html.Span("Cluster Stability: ", style={"fontWeight": "bold"}),
                        html.Span(f"{metrics.get('stability', 'N/A'):.4f}",
                                 style={"color": "green" if metrics.get('stability', 0) > 0.8 else 
                                        "orange" if metrics.get('stability', 0) > 0.6 else "red"})
                    ]) if 'stability' in metrics else None,
                    
                    html.Div(metrics.get('note', ''), style={"fontSize": "11px", "fontStyle": "italic", "marginTop": "3px"})
                ])
            ]
            
            # Add a tooltip about the metrics
            metrics_children.append(
                html.Div([
                    html.Hr(style={"marginTop": "10px", "marginBottom": "10px"}),
                    html.Details([
                        html.Summary("What do these metrics mean?", style={"cursor": "pointer"}),
                        html.Div([
                            html.P("• Silhouette Score: Measures how well-separated clusters are (higher is better, range: -1 to 1)"),
                            html.P("• Davies-Bouldin Index: Measures average similarity between clusters (lower is better, range: 0 to ∞)"),
                            html.P("• Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion (higher is better, range: 0 to ∞)"),
                            html.P("• Hopkins Statistic: Measures clusterability of the data (>0.75 indicates good clustering, range: 0 to 1)"),
                            html.P("• Cluster Stability: How stable clusters are with small perturbations (higher is better, range: 0 to 1)"),
                            html.P("• Physics Consistency: How well clusters align with physical parameters (higher is better, range: 0 to 1)")
                        ], style={"fontSize": "11px", "paddingLeft": "10px"})
                    ])
                ], style={"marginTop": "10px"})
            )
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            metrics_children = [html.Div(f"Error calculating metrics: {str(e)}")]
        
        return fig, "<br>".join(debug_text), latent_store, metrics_children, feature_importance_container
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        error_message = f"Error in callback: {str(e)}\n{trace}"
        print(error_message)  # Print to console for debugging
        
        # Return minimal valid outputs for all return values
        return {}, f"Error: {str(e)}", {}, [], []
# Callback to save latent features
@app.callback(
    Output('download-latent-features', 'data'),
    Input('save-latent-features-btn', 'n_clicks'),
    State('latent-features-filename', 'value'),
    State('autoencoder-latent-store', 'data'),
    prevent_initial_call=True
)
def download_latent_features(n_clicks, filename, latent_store):
    """Generate CSV file of latent features for download."""
    if not n_clicks or not filename or not latent_store or 'latent_features' not in latent_store:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Load the latent features
        latent_df = pd.read_json(latent_store['latent_features'], orient='split')
        
        if latent_df.empty:
            raise dash.exceptions.PreventUpdate
        
        # Prepare for saving - include both latent features and file labels
        latent_dim = latent_store.get('latent_dim', 7)  # Default to 7 if not specified
        
        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            latent_df.to_csv, 
            f"{filename}.csv",
            index=False
        )
        
    except Exception as e:
        print(f"Error saving latent features: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate
        
# Callback to handle the download of selected points
@app.callback(
    Output('download-selection', 'data'),
    Input('save-selection-btn', 'n_clicks'),
    State('selection-filename', 'value'),
    State('selected-points-store', 'data'),
    State('combined-data-store', 'data'),
    prevent_initial_call=True
)
def download_selected_points(n_clicks, filename, selectedData, combined_data_json):
    """Generate CSV file of selected points for download in original format."""
    if not n_clicks or not filename or not selectedData or not combined_data_json:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Load the combined dataframe and UMAP coordinates
        umap_coords = pd.read_json(combined_data_json["umap_coords"], orient='split')
        combined_df = pd.DataFrame()
        if combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
        
        if combined_df.empty:
            raise dash.exceptions.PreventUpdate
        
        # Get indices of selected points
        indices = []
        
        # Handle box selection
        if 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            
            selected_mask = (
                (umap_coords['UMAP1'] >= x_range[0]) & 
                (umap_coords['UMAP1'] <= x_range[1]) & 
                (umap_coords['UMAP2'] >= y_range[0]) & 
                (umap_coords['UMAP2'] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()
        
        # Handle lasso selection
        elif 'lassoPoints' in selectedData:
            # Extract lasso polygon coordinates
            lasso_x = selectedData['lassoPoints']['x']
            lasso_y = selectedData['lassoPoints']['y']
            
            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            
            # Check which points are within the lasso path
            points_array = np.column_stack([umap_coords['UMAP1'], umap_coords['UMAP2']])
            inside_lasso = lasso_path.contains_points(points_array)
            
            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()
        
        # Handle direct point selection
        elif 'points' in selectedData:
            indices = [pt['pointIndex'] for pt in selectedData['points']]
        
        if not indices:
            raise dash.exceptions.PreventUpdate
        
        # Extract only the selected points from the combined dataframe
        selected_df = combined_df.iloc[indices].reset_index(drop=True)
        
        # Remove UMAP coordinates if they exist
        if 'UMAP1' in selected_df.columns:
            selected_df = selected_df.drop(columns=['UMAP1'])
        if 'UMAP2' in selected_df.columns:
            selected_df = selected_df.drop(columns=['UMAP2'])
        
        # Convert back to original format with standardized column names
        # The file_label column would have been added during processing, so remove it
        if 'file_label' in selected_df.columns:
            selected_df = selected_df.drop(columns=['file_label'])
        
        # Get only the particle momentum columns (original data format)
        momentum_columns = [col for col in selected_df.columns if col.startswith('particle_')]
        
        # If we have the momentum columns, convert back to original format
        if momentum_columns and len(momentum_columns) == 15:  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                'Px_ion1', 'Py_ion1', 'Pz_ion1',
                'Px_ion2', 'Py_ion2', 'Pz_ion2',
                'Px_neutral', 'Py_neutral', 'Pz_neutral',
                'Px_electron1', 'Py_electron1', 'Pz_electron1',
                'Px_electron2', 'Py_electron2', 'Pz_electron2'
            ]
            
            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = selected_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = selected_df
        
        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, 
            f"{filename}.csv",
            index=False
        )
        
    except Exception as e:
        print(f"Error saving selection: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate

# Add this new callback:
@app.callback(
    Output('training-interval', 'disabled'),
    Input('train-autoencoder', 'n_clicks'),
    Input('autoencoder-umap-graph', 'figure'),
    prevent_initial_call=True
)
def toggle_interval(n_clicks, figure):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'train-autoencoder':
        return False  # Enable interval
    else:
        return True   # Disable interval

@app.callback(
    Output('training-progress', 'children'),
    Input('training-interval', 'n_intervals'),
)
def update_progress(n_intervals):
    if hasattr(app, 'current_epoch') and hasattr(app, 'total_epochs'):
        if app.total_epochs > 0:
            return f"Training progress: Epoch {app.current_epoch}/{app.total_epochs}"
    return ""

@app.callback(
    Output('feature-selection-ui-genetic', 'children'),
    Input('features-data-store', 'data'),
    Input('genetic-features-store', 'data'),  # Add this to update UI when genetic features are created
    prevent_initial_call=True
)
def update_genetic_feature_ui(features_data, genetic_features_store):
    """Update the feature selection UI for genetic programming."""
    feature_columns = []
    
    # First get standard features
    if features_data and 'column_names' in features_data:
        feature_columns = features_data['column_names']
    
    # Then check if we have discovered genetic features to add
    if genetic_features_store and 'feature_names' in genetic_features_store:
        gp_features = genetic_features_store['feature_names']
        expressions = genetic_features_store.get('expressions', [])
        
        # Add a special category for discovered genetic features
        genetic_category = html.Div([
            html.Div("Discovered Genetic Features", className='feature-category-title', 
                    style={'color': '#d32f2f', 'fontWeight': 'bold'}),
            dcc.Checklist(
                id={'type': 'feature-selector-genetic', 'category': 'GeneticFeatures'},
                options=[{'label': f"{feat} ({expressions[i] if i < len(expressions) else ''})", 
                         'value': feat} for i, feat in enumerate(gp_features)],
                value=[],  # No default selection
                labelStyle={'display': 'block'}
            )
        ], className='feature-category', style={'backgroundColor': '#ffebee', 'padding': '10px', 'marginBottom': '15px'})
        
        # Create regular feature selection UI without the genetic features
        regular_ui = create_feature_categories_ui(feature_columns, 'genetic')
        
        # Combine genetic features and regular features
        if len(gp_features) > 0:
            return [genetic_category] + regular_ui
    
    # If no genetic features or standard features, use the default UI
    if not feature_columns:
        return [html.Div("Upload files to see available features", style={"color": "gray"})]
    else:
        return create_feature_categories_ui(feature_columns, 'genetic')

@app.callback(
    Output('feature-selection-ui-mi', 'children'),
    Input('features-data-store', 'data'),
    Input('mi-features-store', 'data'),
    prevent_initial_call=True
)
def update_mi_feature_ui(features_data, mi_features_store):
    """Update the feature selection UI for MI analysis."""
    if not features_data or 'column_names' not in features_data:
        return [html.Div("Upload files to see available features", style={"color": "gray"})]
    
    # Create feature selection UI
    return create_feature_categories_ui(features_data['column_names'], 'mi')

@app.callback(
    Output('dbscan-params', 'style'),
    Output('kmeans-params', 'style'),
    Output('agglomerative-params', 'style'),
    Input('clustering-method', 'value')
)
def toggle_clustering_params(method):
    """Show/hide clustering parameters based on selected method."""
    dbscan_style = {'display': 'block'} if method == 'dbscan' else {'display': 'none'}
    kmeans_style = {'display': 'block'} if method == 'kmeans' else {'display': 'none'}
    agglomerative_style = {'display': 'block'} if method == 'agglomerative' else {'display': 'none'}
    return dbscan_style, kmeans_style, agglomerative_style

@app.callback(
    Output('run-umap-genetic-status', 'children'),
    Input('run-umap-genetic', 'n_clicks'),
    Input('genetic-features-graph', 'figure'),
    prevent_initial_call=True
)
def update_genetic_umap_status(n_clicks, figure):
    """Update status for UMAP on genetic features."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'run-umap-genetic':
        return "Running UMAP on selected genetic features..."
    elif trigger_id == 'genetic-features-graph':
        # This will be overwritten by the main callback
        return dash.no_update
    
    return dash.no_update

def extract_selection_indices(selection_data, coords_df):
    """Extract indices from selection data."""
    indices = []
    
    # Handle box selection
    if 'range' in selection_data:
        x_range = selection_data['range']['x']
        y_range = selection_data['range']['y']
        
        selected_mask = (
            (coords_df['UMAP1'] >= x_range[0]) & 
            (coords_df['UMAP1'] <= x_range[1]) & 
            (coords_df['UMAP2'] >= y_range[0]) & 
            (coords_df['UMAP2'] <= y_range[1])
        )
        indices = np.where(selected_mask)[0].tolist()
    
    # Handle lasso selection
    elif 'lassoPoints' in selection_data:
        # Extract lasso polygon coordinates
        lasso_x = selection_data['lassoPoints']['x']
        lasso_y = selection_data['lassoPoints']['y']
        
        # Create a Path object from the lasso points
        lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
        
        # Check which points are within the lasso path
        points_array = np.column_stack([coords_df['UMAP1'], coords_df['UMAP2']])
        inside_lasso = lasso_path.contains_points(points_array)
        
        # Get indices of points inside the lasso
        indices = np.where(inside_lasso)[0].tolist()
    
    # Handle direct point selection
    elif 'points' in selection_data:
        indices = [pt['pointIndex'] for pt in selection_data['points']]
    
    return indices

@app.callback(
    Output('download-genetic-features', 'data'),
    Input('save-genetic-features-btn', 'n_clicks'),
    State('genetic-features-filename', 'value'),
    State('genetic-features-store', 'data'),
    prevent_initial_call=True
)
def download_genetic_features(n_clicks, filename, genetic_features_store):
    """Generate CSV file of genetic features for download."""
    if not n_clicks or not filename or not genetic_features_store or 'genetic_features' not in genetic_features_store:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Load the genetic features
        gp_df = pd.read_json(genetic_features_store['genetic_features'], orient='split')
        
        if gp_df.empty:
            raise dash.exceptions.PreventUpdate
        
        # Create expressions text as header comment
        expressions = genetic_features_store.get('expressions', [])
        expressions_header = "# Generated features and their expressions:\n"
        for i, expr in enumerate(expressions):
            expressions_header += f"# GP_Feature_{i+1}: {expr}\n"
        
        # Write to string buffer with the expressions as header comment
        import io
        buffer = io.StringIO()
        buffer.write(expressions_header)
        gp_df.to_csv(buffer, index=False)
        
        # Return as download
        return dict(
            content=buffer.getvalue(),
            filename=f"{filename}.csv"
        )
        
    except Exception as e:
        print(f"Error saving genetic features: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output('download-mi-features', 'data'),
    Input('save-mi-features-btn', 'n_clicks'),
    State('mi-features-filename', 'value'),
    State('mi-features-store', 'data'),
    prevent_initial_call=True
)
def download_mi_features(n_clicks, filename, mi_features_store):
    """Generate CSV file of MI-selected features and latent representations for download."""
    if not n_clicks or not filename or not mi_features_store or 'latent_features' not in mi_features_store:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Load the latent features
        latent_df = pd.read_json(mi_features_store['latent_features'], orient='split')
        
        if latent_df.empty:
            raise dash.exceptions.PreventUpdate
        
        # Get selected features for header information
        selected_features = mi_features_store.get('selected_features', [])
        mi_scores = mi_features_store.get('mi_scores', {})
        
        # Create header with MI information
        mi_header = "# Mutual Information Feature Selection Results\n"
        mi_header += "# Selected Features and their MI scores:\n"
        
        for feature in selected_features:
            score = mi_scores.get(feature, 0.0)
            mi_header += f"# {feature}: {score:.6f}\n"
        
        mi_header += "#\n# Latent Features (from Autoencoder):\n"
        latent_dim = mi_features_store.get('latent_dim', 7)
        for i in range(latent_dim):
            mi_header += f"# Latent_{i}\n"
        
        # Write to string buffer with the MI information as header
        import io
        buffer = io.StringIO()
        buffer.write(mi_header)
        latent_df.to_csv(buffer, index=False)
        
        # Return as download
        return dict(
            content=buffer.getvalue(),
            filename=f"{filename}.csv"
        )
        
    except Exception as e:
        print(f"Error saving MI features: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate

@app.callback(
    Output('genetic-features-graph', 'figure'),
    Output('genetic-features-debug-output', 'children'),
    Output('genetic-features-store', 'data'),
    Output('run-umap-genetic-status', 'children', allow_duplicate=True),
    Output('umap-quality-metrics-genetic', 'children'),  # Add this output
    Input('run-genetic-features', 'n_clicks'),
    Input('run-umap-genetic', 'n_clicks'),
    State('genetic-data-source', 'value'),
    State('clustering-method', 'value'),
    State('dbscan-eps', 'value'),
    State('dbscan-min-samples', 'value'),
    State('kmeans-n-clusters', 'value'),
    State('agglomerative-n-clusters', 'value'),
    State('agglomerative-linkage', 'value'),
    State('gp-generations', 'value'),
    State('gp-population-size', 'value'),
    State('gp-n-components', 'value'),
    State('gp-functions', 'value'),
    State({'type': 'feature-selector-genetic', 'category': ALL}, 'value'),
    State('selected-points-store', 'data'),
    State('selected-points-run-store', 'data'),
    State('autoencoder-latent-store', 'data'),
    State('combined-data-store', 'data'),
    State('umap-graph', 'figure'),
    State('genetic-features-store', 'data'),
    State('metric-selector-genetic', 'value'),  # Add this state
    prevent_initial_call=True
)
def run_genetic_feature_discovery_and_umap(run_gp_clicks, run_umap_clicks, data_source, clustering_method, 
                                 dbscan_eps, dbscan_min_samples, 
                                 kmeans_n_clusters, agglo_n_clusters, agglo_linkage,
                                 gp_generations, gp_population_size, gp_n_components, 
                                 gp_functions, selected_features_list,
                                 graph1_selection, graph3_selection, 
                                 autoencoder_latent, combined_data_json, original_figure,
                                 genetic_features_store, selected_metrics):
    """Run genetic programming to discover features that explain clustering patterns,
    or run UMAP on previously discovered genetic features."""
    
    # Import clustering modules at the beginning of the function
    from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
    
    # Initialize debug info and check which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, "No action triggered.", {}, "", []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    debug_text = []
    
    # If the UMAP button was clicked and we have stored genetic features, skip the GP part
    if trigger_id == 'run-umap-genetic' and genetic_features_store and 'genetic_features' in genetic_features_store:
        debug_text.append("Running UMAP visualization on previously discovered genetic features...")
        
        try:
            # Load the genetic features from the store
            gp_df = pd.read_json(genetic_features_store['genetic_features'], orient='split')
            feature_names = genetic_features_store.get('feature_names', [])
            expressions = genetic_features_store.get('expressions', [])
            
            if gp_df.empty or not feature_names:
                return {}, "No genetic features found. Run Genetic Feature Discovery first.", genetic_features_store, "Error: No features to visualize", []
            
            # Get user-selected features for UMAP
            all_selected_features = []
            for features in selected_features_list:
                if features:  # Only add non-empty lists
                    all_selected_features.extend(features)
            
            # Filter to only include genetic features (GP_Feature_X)
            gp_selected_features = [f for f in all_selected_features if f.startswith('GP_Feature_')]
            
            if not gp_selected_features:
                # If no GP features selected, use all available GP features
                gp_selected_features = feature_names
                debug_text.append(f"No genetic features specifically selected, using all {len(gp_selected_features)} available genetic features")
            else:
                debug_text.append(f"Using {len(gp_selected_features)} selected genetic features for UMAP")
            
            # Extract the genetic feature data
            X_gp = gp_df[gp_selected_features].to_numpy()
            
            # Handle NaN/inf values
            X_gp = np.nan_to_num(X_gp, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Run UMAP on the selected genetic features
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
            
            umap_result = reducer.fit_transform(X_gp)
            
            # Create DataFrame for UMAP visualization
            umap_df = pd.DataFrame({
                'UMAP1': umap_result[:, 0],
                'UMAP2': umap_result[:, 1],
                'Cluster': gp_df['Cluster'].values,
                'file_label': gp_df['file_label'].values
            })
            
            # Create visualization
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Extract color information from original figure
            color_map = {}
            if original_figure and 'data' in original_figure:
                for trace in original_figure['data']:
                    if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                        # Clean the label if it contains point count
                        clean_name = trace['name']
                        if ' (' in clean_name:
                            clean_name = clean_name.split(' (')[0]
                        color_map[clean_name] = trace['marker']['color']
            
            # Add traces for each file label with consistent colors
            for label in umap_df['file_label'].unique():
                mask = umap_df['file_label'] == label
                df_subset = umap_df[mask]
                
                # Get color from original figure if available
                color = color_map.get(label, None)
                
                fig.add_trace(go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=0.7
                    ),
                    name=f"{label} ({len(df_subset)} pts)"
                ))
            
            # Update figure layout
            fig.update_layout(
                height=600,
                title=f"UMAP of Selected Genetic Features ({len(gp_selected_features)} features)",
                xaxis_title="UMAP1",
                yaxis_title="UMAP2",
                legend_title="Data File"
            )
            
            # Add information about which features were used
            debug_text.append("Features used for UMAP visualization:")
            for feature in gp_selected_features:
                idx = int(feature.split('_')[-1]) - 1  # Extract feature number and adjust for 0-indexing
                if idx < len(expressions):
                    debug_text.append(f"{feature}: {expressions[idx]}")
            
            # Calculate clustering metrics
            metrics_children = []
            try:
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
                
                # Get UMAP coordinates for clustering
                X_umap = umap_df[['UMAP1', 'UMAP2']].to_numpy()
                
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
                noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
                
                metrics = {}
                
                # Only calculate metrics if we have at least 2 clusters
                if n_clusters >= 2:
                    # For metrics, we need to exclude noise points (-1)
                    mask = cluster_labels != -1
                    non_noise_points = np.sum(mask)
                    non_noise_clusters = len(set(cluster_labels[mask]))
                    
                    if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                        if 'silhouette' in selected_metrics:
                            metrics["silhouette"] = silhouette_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        if 'davies_bouldin' in selected_metrics:
                            metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        if 'calinski_harabasz' in selected_metrics:
                            metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        # Add new metrics based on selection
                        if 'hopkins' in selected_metrics:
                            h_stat = hopkins_statistic(X_umap_scaled)
                            metrics["hopkins"] = h_stat
                        
                        if 'stability' in selected_metrics:
                            stability = cluster_stability(X_umap_scaled, best_eps, 5, n_iterations=3)
                            metrics["stability"] = stability
                        
                        metrics["note"] = "Metrics calculated excluding noise points"
                    else:
                        metrics["note"] = "Not enough valid points for metrics"
                else:
                    # Try KMeans as fallback
                    from sklearn.cluster import KMeans
                    
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    fallback_labels = kmeans.fit_predict(X_umap_scaled)
                    
                    if 'silhouette' in selected_metrics:
                        metrics["silhouette"] = silhouette_score(X_umap_scaled, fallback_labels)
                    
                    if 'davies_bouldin' in selected_metrics:
                        metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled, fallback_labels)
                    
                    if 'calinski_harabasz' in selected_metrics:
                        metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled, fallback_labels)
                    
                    if 'hopkins' in selected_metrics:
                        h_stat = hopkins_statistic(X_umap_scaled)
                        metrics["hopkins"] = h_stat
                    
                    metrics["note"] = "DBSCAN found no clusters, metrics based on KMeans fallback"
                
                # Create UI elements for the metrics
                metrics_children = [
                    html.H4("Clustering Quality Metrics (DBSCAN)", style={"fontSize": "14px", "marginBottom": "5px"}),
                    html.Div([
                        # Existing metrics
                        html.Div([
                            html.Span("Estimated Clusters: ", style={"fontWeight": "bold"}),
                            html.Span(f"{n_clusters}")
                        ]),
                        html.Div([
                            html.Span("Noise Points: ", style={"fontWeight": "bold"}),
                            html.Span(f"{n_noise} ({noise_ratio:.1%})")
                        ]),
                        html.Div([
                            html.Span("DBSCAN eps: ", style={"fontWeight": "bold"}),
                            html.Span(f"{best_eps:.3f}")
                        ]),
                        
                        # Basic metrics (existing)
                        html.Div([
                            html.Span("Silhouette Score: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('silhouette', 'N/A'):.4f}", 
                                    style={"color": "green" if metrics.get('silhouette', 0) > 0.5 else 
                                           "orange" if metrics.get('silhouette', 0) > 0.25 else "red"})
                        ]) if 'silhouette' in metrics else None,
                        
                        html.Div([
                            html.Span("Davies-Bouldin Index: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('davies_bouldin', 'N/A'):.4f}",
                                    style={"color": "green" if metrics.get('davies_bouldin', float('inf')) < 0.8 else 
                                           "orange" if metrics.get('davies_bouldin', float('inf')) < 1.5 else "red"})
                        ]) if 'davies_bouldin' in metrics else None,
                        
                        html.Div([
                            html.Span("Calinski-Harabasz Index: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('calinski_harabasz', 'N/A'):.1f}",
                                    style={"color": "green" if metrics.get('calinski_harabasz', 0) > 100 else 
                                           "orange" if metrics.get('calinski_harabasz', 0) > 50 else "red"})
                        ]) if 'calinski_harabasz' in metrics else None,
                        
                        # New metrics
                        html.Div([
                            html.Span("Hopkins Statistic: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('hopkins', 'N/A'):.4f}",
                                     style={"color": "green" if metrics.get('hopkins', 0) > 0.75 else 
                                            "orange" if metrics.get('hopkins', 0) > 0.6 else "red"})
                        ]) if 'hopkins' in metrics else None,
                        
                        html.Div([
                            html.Span("Cluster Stability: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('stability', 'N/A'):.4f}",
                                     style={"color": "green" if metrics.get('stability', 0) > 0.8 else 
                                            "orange" if metrics.get('stability', 0) > 0.6 else "red"})
                        ]) if 'stability' in metrics else None,
                        
                        html.Div(metrics.get('note', ''), style={"fontSize": "11px", "fontStyle": "italic", "marginTop": "3px"})
                    ])
                ]
                
                # Add a tooltip about the metrics
                metrics_children.append(
                    html.Div([
                        html.Hr(style={"marginTop": "10px", "marginBottom": "10px"}),
                        html.Details([
                            html.Summary("What do these metrics mean?", style={"cursor": "pointer"}),
                            html.Div([
                                html.P("• Silhouette Score: Measures how well-separated clusters are (higher is better, range: -1 to 1)"),
                                html.P("• Davies-Bouldin Index: Measures average similarity between clusters (lower is better, range: 0 to ∞)"),
                                html.P("• Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion (higher is better, range: 0 to ∞)"),
                                html.P("• Hopkins Statistic: Measures clusterability of the data (>0.75 indicates good clustering, range: 0 to 1)"),
                                html.P("• Cluster Stability: How stable clusters are with small perturbations (higher is better, range: 0 to 1)"),
                                html.P("• Physics Consistency: How well clusters align with physical parameters (higher is better, range: 0 to 1)")
                            ], style={"fontSize": "11px", "paddingLeft": "10px"})
                        ])
                    ], style={"marginTop": "10px"})
                )
            
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                metrics_children = [html.Div(f"Error calculating metrics: {str(e)}")]
            
            return fig, "<br>".join(debug_text), genetic_features_store, "UMAP visualization complete!", metrics_children
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            error_message = f"Error running UMAP on genetic features: {str(e)}<br>{trace}"
            return {}, error_message, genetic_features_store, "Error running UMAP", []
    
    # If we're here, we're running the full genetic programming discovery
    if trigger_id == 'run-genetic-features':
        debug_text.append(f"Data source: {data_source}")
        debug_text.append(f"Clustering method: {clustering_method}")
        
        # Define a custom exponential function with overflow protection
        def custom_exp(x):
            return np.where(x < 50, np.exp(x), np.exp(50))  # Prevents overflow
        
        exp_function = make_function(function=custom_exp, name="exp", arity=1)
        
        # Build function set based on user selection
        function_set = []
        if 'basic' in gp_functions:
            function_set.extend(['add', 'sub', 'mul', 'div'])
        if 'trig' in gp_functions:
            function_set.extend(['sin', 'cos', 'tan'])
        if 'exp_log' in gp_functions:
            function_set.extend(['log', exp_function])
        if 'sqrt_pow' in gp_functions:
            function_set.extend(['sqrt'])
        if 'special' in gp_functions:
            function_set.extend(['abs', 'inv'])
        
        if not function_set:
            function_set = ['add', 'sub', 'mul', 'div']  # Default to basic functions
        
        debug_text.append(f"Function set: {', '.join([str(f) for f in function_set])}")
        debug_text.append(f"Generations: {gp_generations}, Population: {gp_population_size}, Features: {gp_n_components}")
        
        try:
            # Step 1: Prepare data based on selected source
            # -----------------------------------------
            
            # Load the combined dataframe
            if combined_data_json and "combined_df" in combined_data_json and combined_data_json["combined_df"] != "{}":
                combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
                debug_text.append(f"Loaded combined dataset with {len(combined_df)} rows")
            else:
                return {}, "No combined dataset available. Please run UMAP first.", {}, "", []
            
            # Load UMAP coordinates
            umap_coords = None
            if "umap_coords" in combined_data_json and combined_data_json["umap_coords"] != "{}":
                umap_coords = pd.read_json(combined_data_json["umap_coords"], orient='split')
            
            # Load Graph 3 subset
            graph3_subset_df = None
            if combined_data_json and "graph3_subset" in combined_data_json and combined_data_json["graph3_subset"] != "{}":
                graph3_subset_df = pd.read_json(combined_data_json["graph3_subset"], orient='split')
                debug_text.append(f"Loaded Graph 3 subset with {len(graph3_subset_df)} rows")
            
            # Collect all selected features for genetic programming
            all_selected_features = []
            for features in selected_features_list:
                if features:  # Only add non-empty lists
                    all_selected_features.extend(features)
            
            if not all_selected_features:
                # Use particle momentum as default if nothing is selected
                all_selected_features = [col for col in combined_df.columns if col.startswith('particle_')]
                debug_text.append(f"No features selected, using {len(all_selected_features)} default momentum features")
            else:
                debug_text.append(f"Using {len(all_selected_features)} selected features")
            
            # Prepare data based on data source
            if data_source == 'all':
                # Use all data
                df_for_analysis = combined_df.copy()
                labels = combined_df['file_label'].copy()
                debug_text.append(f"Using all {len(df_for_analysis)} rows for analysis")
            
            elif data_source == 'graph1-selection' and graph1_selection and umap_coords is not None:
                # Use selection from Graph 1
                indices = extract_selection_indices(graph1_selection, umap_coords)
                if not indices:
                    return {}, "No valid points found in Graph 1 selection.", {}, "", []
                
                df_for_analysis = combined_df.iloc[indices].copy()
                labels = df_for_analysis['file_label'].copy()
                debug_text.append(f"Selected {len(df_for_analysis)} rows from Graph 1 selection")
            
            elif data_source == 'graph3-selection' and graph3_selection and graph3_subset_df is not None:
                # Use selection from Graph 3
                df_for_analysis = graph3_subset_df.copy()
                labels = df_for_analysis['file_label'].copy()
                debug_text.append(f"Using Graph 3 selection with {len(df_for_analysis)} rows")
            
            elif data_source == 'autoencoder-latent' and autoencoder_latent:
                # Use autoencoder latent space
                try:
                    latent_df = pd.read_json(autoencoder_latent['latent_features'], orient='split')
                    # Keep only the latent features, not the labels
                    latent_cols = [col for col in latent_df.columns if col.startswith('Latent_')]
                    df_for_analysis = latent_df[latent_cols].copy()
                    # We still need the labels for visualization
                    labels = latent_df['file_label'].copy()
                    debug_text.append(f"Using autoencoder latent space with {len(df_for_analysis)} rows and {len(latent_cols)} dimensions")
                    # Override selected features with latent features
                    all_selected_features = latent_cols
                except Exception as e:
                    return {}, f"Error loading autoencoder latent space: {str(e)}", {}, "", []
            
            else:
                # Default to all data
                df_for_analysis = combined_df.copy()
                labels = combined_df['file_label'].copy()
                debug_text.append(f"Defaulting to all {len(df_for_analysis)} rows")
            
            # Extract feature data
            feature_cols = [col for col in df_for_analysis.columns if col in all_selected_features]
            
            if not feature_cols:
                return {}, "No valid features selected for genetic programming.", {}, "", []
            
            # Extract feature data
            X = df_for_analysis[feature_cols].to_numpy()
            
            # Handle NaN/inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Step 2: Apply clustering to get a rough idea of structure (optional for visualization)
            # -----------------------------------------
            from sklearn.preprocessing import StandardScaler
            
            # Standardize data before clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            debug_text.append(f"Standardized data for analysis, shape: {X_scaled.shape}")
            
            # Apply the selected clustering method
            if clustering_method == 'dbscan':
                debug_text.append(f"Running DBSCAN with eps={dbscan_eps}, min_samples={dbscan_min_samples}")
                clusterer = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
                cluster_labels = clusterer.fit_predict(X_scaled)
            
            elif clustering_method == 'kmeans':
                debug_text.append(f"Running KMeans with n_clusters={kmeans_n_clusters}")
                clusterer = KMeans(n_clusters=int(kmeans_n_clusters), random_state=42)
                cluster_labels = clusterer.fit_predict(X_scaled)
            
            elif clustering_method == 'agglomerative':
                debug_text.append(f"Running Agglomerative clustering with n_clusters={agglo_n_clusters}, linkage={agglo_linkage}")
                clusterer = AgglomerativeClustering(n_clusters=int(agglo_n_clusters), linkage=agglo_linkage)
                cluster_labels = clusterer.fit_predict(X_scaled)
            
            # Handle case where all points are noise in DBSCAN
            if clustering_method == 'dbscan' and np.all(cluster_labels == -1):
                debug_text.append("All points were labeled as noise. Using KMeans clustering as fallback.")
                clusterer = KMeans(n_clusters=3, random_state=42)
                cluster_labels = clusterer.fit_predict(X_scaled)
            
            # Count clusters and noise points
            unique_labels = np.unique(cluster_labels)
            num_clusters = len([label for label in unique_labels if label != -1])
            num_noise = np.sum(cluster_labels == -1) if -1 in unique_labels else 0
            
            debug_text.append(f"Found {num_clusters} clusters and {num_noise} noise points")
            
            # Step 3: Run genetic programming with standard metric
            # -----------------------------------------
            debug_text.append("Training genetic program to discover features with high information content...")

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
                    metric='spearman',  # Use Spearman correlation as metric
                    random_state=42,
                    parsimony_coefficient=0.05,  # Adds penalty for complexity
                    n_jobs=-1
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
                feature_names = [f"GP_Feature_{i+1}" for i in range(genetic_features.shape[1])]
                gp_df = pd.DataFrame(genetic_features, columns=feature_names)
                
                # Add cluster labels and original labels
                gp_df['Cluster'] = cluster_labels
                gp_df['file_label'] = labels.values
                
                # Add the generated features to the combined data
                combined_df_copy = combined_df.copy()
                for i, col in enumerate(feature_names):
                    # Add features to the combined dataframe
                    combined_df_copy[col] = np.nan  # Initialize with NaN
                    
                    if data_source == 'all':
                        # If using all data, we can add features directly
                        combined_df_copy[col] = genetic_features[:, i]
                    elif data_source == 'graph1-selection' and graph1_selection and umap_coords is not None:
                        # For Graph 1 selection, add features to the selected rows
                        indices = extract_selection_indices(graph1_selection, umap_coords)
                        if indices:
                            for j, idx in enumerate(indices):
                                if idx < len(combined_df_copy):
                                    combined_df_copy.loc[idx, col] = genetic_features[j, i]
                    else:
                        # For other selections, just note that features are partial
                        debug_text.append(f"Note: Discovered feature {col} is only populated for the selected data")

                # Update the features store with the new features
                if 'features_data_store' in combined_data_json:
                    features_data = combined_data_json['features_data_store'].copy()
                    if 'column_names' in features_data:
                        if not any(name in features_data['column_names'] for name in feature_names):
                            features_data['column_names'].extend(feature_names)
                            debug_text.append(f"Added {len(feature_names)} new features to the feature list")
                else:
                    # Create a new features data store if one doesn't exist
                    features_data = {'column_names': list(combined_df.columns) + feature_names}
                
                # Now create the placeholder figure and store
                placeholder_fig = {
                    'data': [],
                    'layout': {
                        'title': 'Genetic features discovered! Select features and click "Run UMAP on Genetic Features" to visualize',
                        'xaxis': {'title': 'UMAP1'},
                        'yaxis': {'title': 'UMAP2'},
                        'height': 600
                    }
                }
                
                # Store the genetic features but don't run UMAP yet
                new_genetic_features_store = {
                    'genetic_features': gp_df.to_json(date_format='iso', orient='split'),
                    'expressions': expressions,
                    'feature_cols': feature_cols,
                    'feature_names': feature_names
                }
                
                # Add information about the discovered features to the debug text
                debug_text.append("Discovered genetic features:")
                for i, expr in enumerate(expressions):
                    debug_text.append(f"{feature_names[i]}: {expr}")
                
                debug_text.append("Select features of interest and click 'Run UMAP on Genetic Features' to visualize.")
                
                return placeholder_fig, "<br>".join(debug_text), new_genetic_features_store, "Genetic feature discovery complete!", []
                
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                error_message = f"Error in genetic programming: {str(e)}<br>{trace}"
                debug_text.append(error_message)
                return {}, "<br>".join(debug_text), {}, "Error occurred during genetic programming", []
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            error_message = f"Error in genetic feature discovery: {str(e)}<br>{trace}"
            return {}, error_message, {}, "Error occurred", []
    
    # If neither button was properly triggered, return empty states
    return {}, "Click 'Run Genetic Feature Discovery' to start.", {}, "", []


@app.callback(
    Output('feature-importance-table', 'data'),
    Input('feature-search-button', 'n_clicks'),
    Input('feature-sort-option', 'value'),
    State('feature-search-input', 'value'),
    State('autoencoder-latent-store', 'data'),
    prevent_initial_call=True
)
def update_feature_importance_table(n_clicks, sort_option, search_term, latent_store):
    """Update the feature importance table based on search and sort options."""
    ctx = dash.callback_context
    if not ctx.triggered or not latent_store:
        raise dash.exceptions.PreventUpdate
    
    # Get MI and correlation scores from the store
    try:
        mi_scores_dict = latent_store.get('mi_scores', {})
        corr_scores_dict = latent_store.get('corr_scores', {})
        
        if not mi_scores_dict or not corr_scores_dict:
            return []
        
        # Get features that are in both dictionaries
        common_features = list(set(mi_scores_dict.keys()).intersection(set(corr_scores_dict.keys())))
        
        if not common_features:
            return []
        
        # Create DataFrame with all features and scores
        feature_importance_df = pd.DataFrame({
            'Feature': common_features,
            'Mutual_Information': [mi_scores_dict.get(f, 0.0) for f in common_features],
            'Correlation': [abs(corr_scores_dict.get(f, 0.0)) for f in common_features]  # Use absolute correlation value
        })
        
        # Filter by search term if provided
        if search_term and len(search_term.strip()) > 0:
            pattern = re.compile(search_term, re.IGNORECASE)
            feature_importance_df = feature_importance_df[
                feature_importance_df['Feature'].apply(lambda x: bool(pattern.search(x)))
            ]
        
        # Sort based on selected option
        if sort_option == 'mi':
            feature_importance_df = feature_importance_df.sort_values('Mutual_Information', ascending=False)
        else:  # sort by correlation
            feature_importance_df = feature_importance_df.sort_values('Correlation', ascending=False)
        
        # Reset index for proper ranking
        feature_importance_df = feature_importance_df.reset_index(drop=True)
        
        # Prepare the table data
        table_data = []
        for i, row in feature_importance_df.iterrows():
            table_data.append({
                'Rank': i + 1,
                'Feature': row['Feature'],
                'MI Score': f"{row['Mutual_Information']:.4f}",
                'Correlation': f"{row['Correlation']:.4f}"  # Use absolute correlation value
            })
        
        return table_data
    
    except Exception as e:
        print(f"Error updating feature importance table: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


@app.callback(
    Output('mi-features-graph', 'figure'),
    Output('mi-features-debug-output', 'children'),
    Output('mi-features-store', 'data'),
    Output('run-umap-mi-status', 'children', allow_duplicate=True),
    Output('umap-quality-metrics-mi', 'children'),  # Add this output
    Input('run-mi-features', 'n_clicks'),
    Input('run-umap-mi', 'n_clicks'),
    State('mi-data-source', 'value'),
    State('mi-target-variables', 'value'),
    State('mi-redundancy-threshold', 'value'),
    State('mi-max-features', 'value'),
    State('autoencoder-latent-dim', 'value'),
    State('autoencoder-epochs', 'value'),
    State('autoencoder-batch-size', 'value'),
    State('autoencoder-learning-rate', 'value'),
    State({'type': 'feature-selector-mi', 'category': ALL}, 'value'),
    State('selected-points-store', 'data'),
    State('selected-points-run-store', 'data'),
    State('combined-data-store', 'data'),
    State('umap-graph', 'figure'),
    State('mi-features-store', 'data'),
    State('metric-selector-mi', 'value'),  # Add this state
    prevent_initial_call=True
)
def run_mi_feature_selection_and_umap(run_mi_clicks, run_umap_clicks, data_source, 
                                    target_variables, redundancy_threshold, max_features,
                                    latent_dim, epochs, batch_size, learning_rate,
                                    selected_features_list, graph1_selection, graph3_selection, 
                                    combined_data_json, original_figure, mi_features_store,
                                    selected_metrics):
    """Run mutual information feature selection followed by autoencoder for dimensionality reduction."""
    
    # Initialize debug info and check which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, "No action triggered.", {}, "", []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    debug_text = []
    
    # If UMAP button was clicked and we have stored features, skip the MI part
    if trigger_id == 'run-umap-mi' and mi_features_store and 'latent_features' in mi_features_store:
        debug_text.append("Running UMAP visualization on previously computed MI-based features...")
        
        try:
            # Load the latent features from the store
            latent_df = pd.read_json(mi_features_store['latent_features'], orient='split')
            selected_features = mi_features_store.get('selected_features', [])
            
            if latent_df.empty:
                return {}, "No MI-based features found. Run MI Feature Selection first.", mi_features_store, "", []
            
            # Run UMAP on the latent space
            latent_dim = mi_features_store.get('latent_dim', 7)
            X_latent = latent_df[[f"Latent_{i}" for i in range(latent_dim)]].to_numpy()
            
            # Run UMAP
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
            
            umap_result = reducer.fit_transform(X_latent)
            
            # Create DataFrame for UMAP visualization
            umap_df = pd.DataFrame({
                'UMAP1': umap_result[:, 0],
                'UMAP2': umap_result[:, 1],
                'file_label': latent_df['file_label'].values
            })
            
            # Create visualization
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Extract color information from original figure
            color_map = {}
            if original_figure and 'data' in original_figure:
                for trace in original_figure['data']:
                    if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                        # Clean the label if it contains point count
                        clean_name = trace['name']
                        if ' (' in clean_name:
                            clean_name = clean_name.split(' (')[0]
                        color_map[clean_name] = trace['marker']['color']
            
            # Add traces for each file label with consistent colors
            for label in umap_df['file_label'].unique():
                mask = umap_df['file_label'] == label
                df_subset = umap_df[mask]
                
                # Get color from original figure if available
                color = color_map.get(label, None)
                
                fig.add_trace(go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=0.7
                    ),
                    name=f"{label} ({len(df_subset)} pts)"
                ))
            
            # Update figure layout
            fig.update_layout(
                height=600,
                title=f"UMAP of MI-Selected Features and Autoencoder (latent dim={latent_dim})",
                xaxis_title="UMAP1",
                yaxis_title="UMAP2",
                legend_title="Data File"
            )
            
            # Add information about which features were selected
            debug_text.append(f"Selected {len(selected_features)} features using mutual information:")
            debug_text.append(", ".join(selected_features[:10]) + ("..." if len(selected_features) > 10 else ""))
            
            # Calculate clustering metrics
            metrics_children = []
            try:
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
                
                # Get UMAP coordinates for clustering
                X_umap = umap_df[['UMAP1', 'UMAP2']].to_numpy()
                
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
                noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
                
                metrics = {}
                
                # Only calculate metrics if we have at least 2 clusters
                if n_clusters >= 2:
                    # For metrics, we need to exclude noise points (-1)
                    mask = cluster_labels != -1
                    non_noise_points = np.sum(mask)
                    non_noise_clusters = len(set(cluster_labels[mask]))
                    
                    if non_noise_points > non_noise_clusters and non_noise_clusters > 1:
                        if 'silhouette' in selected_metrics:
                            metrics["silhouette"] = silhouette_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        if 'davies_bouldin' in selected_metrics:
                            metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        if 'calinski_harabasz' in selected_metrics:
                            metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled[mask], cluster_labels[mask])
                        
                        # Add new metrics based on selection
                        if 'hopkins' in selected_metrics:
                            h_stat = hopkins_statistic(X_umap_scaled)
                            metrics["hopkins"] = h_stat
                        
                        if 'stability' in selected_metrics:
                            stability = cluster_stability(X_umap_scaled, best_eps, 5, n_iterations=3)
                            metrics["stability"] = stability
                        
                        metrics["note"] = "Metrics calculated excluding noise points"
                    else:
                        metrics["note"] = "Not enough valid points for metrics"
                else:
                    # Try KMeans as fallback
                    from sklearn.cluster import KMeans
                    
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    fallback_labels = kmeans.fit_predict(X_umap_scaled)
                    
                    if 'silhouette' in selected_metrics:
                        metrics["silhouette"] = silhouette_score(X_umap_scaled, fallback_labels)
                    
                    if 'davies_bouldin' in selected_metrics:
                        metrics["davies_bouldin"] = davies_bouldin_score(X_umap_scaled, fallback_labels)
                    
                    if 'calinski_harabasz' in selected_metrics:
                        metrics["calinski_harabasz"] = calinski_harabasz_score(X_umap_scaled, fallback_labels)
                    
                    if 'hopkins' in selected_metrics:
                        h_stat = hopkins_statistic(X_umap_scaled)
                        metrics["hopkins"] = h_stat
                    
                    metrics["note"] = "DBSCAN found no clusters, metrics based on KMeans fallback"
                
                # Create UI elements for the metrics
                metrics_children = [
                    html.H4("Clustering Quality Metrics (DBSCAN)", style={"fontSize": "14px", "marginBottom": "5px"}),
                    html.Div([
                        # Existing metrics
                        html.Div([
                            html.Span("Estimated Clusters: ", style={"fontWeight": "bold"}),
                            html.Span(f"{n_clusters}")
                        ]),
                        html.Div([
                            html.Span("Noise Points: ", style={"fontWeight": "bold"}),
                            html.Span(f"{n_noise} ({noise_ratio:.1%})")
                        ]),
                        html.Div([
                            html.Span("DBSCAN eps: ", style={"fontWeight": "bold"}),
                            html.Span(f"{best_eps:.3f}")
                        ]),
                        
                        # Basic metrics (existing)
                        html.Div([
                            html.Span("Silhouette Score: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('silhouette', 'N/A'):.4f}", 
                                    style={"color": "green" if metrics.get('silhouette', 0) > 0.5 else 
                                           "orange" if metrics.get('silhouette', 0) > 0.25 else "red"})
                        ]) if 'silhouette' in metrics else None,
                        
                        html.Div([
                            html.Span("Davies-Bouldin Index: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('davies_bouldin', 'N/A'):.4f}",
                                    style={"color": "green" if metrics.get('davies_bouldin', float('inf')) < 0.8 else 
                                           "orange" if metrics.get('davies_bouldin', float('inf')) < 1.5 else "red"})
                        ]) if 'davies_bouldin' in metrics else None,
                        
                        html.Div([
                            html.Span("Calinski-Harabasz Index: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('calinski_harabasz', 'N/A'):.1f}",
                                    style={"color": "green" if metrics.get('calinski_harabasz', 0) > 100 else 
                                           "orange" if metrics.get('calinski_harabasz', 0) > 50 else "red"})
                        ]) if 'calinski_harabasz' in metrics else None,
                        
                        # New metrics
                        html.Div([
                            html.Span("Hopkins Statistic: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('hopkins', 'N/A'):.4f}",
                                     style={"color": "green" if metrics.get('hopkins', 0) > 0.75 else 
                                            "orange" if metrics.get('hopkins', 0) > 0.6 else "red"})
                        ]) if 'hopkins' in metrics else None,
                        
                        html.Div([
                            html.Span("Cluster Stability: ", style={"fontWeight": "bold"}),
                            html.Span(f"{metrics.get('stability', 'N/A'):.4f}",
                                     style={"color": "green" if metrics.get('stability', 0) > 0.8 else 
                                            "orange" if metrics.get('stability', 0) > 0.6 else "red"})
                        ]) if 'stability' in metrics else None,
                        
                        html.Div(metrics.get('note', ''), style={"fontSize": "11px", "fontStyle": "italic", "marginTop": "3px"})
                    ])
                ]
                
                # Add a tooltip about the metrics
                metrics_children.append(
                    html.Div([
                        html.Hr(style={"marginTop": "10px", "marginBottom": "10px"}),
                        html.Details([
                            html.Summary("What do these metrics mean?", style={"cursor": "pointer"}),
                            html.Div([
                                html.P("• Silhouette Score: Measures how well-separated clusters are (higher is better, range: -1 to 1)"),
                                html.P("• Davies-Bouldin Index: Measures average similarity between clusters (lower is better, range: 0 to ∞)"),
                                html.P("• Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion (higher is better, range: 0 to ∞)"),
                                html.P("• Hopkins Statistic: Measures clusterability of the data (>0.75 indicates good clustering, range: 0 to 1)"),
                                html.P("• Cluster Stability: How stable clusters are with small perturbations (higher is better, range: 0 to 1)"),
                                html.P("• Physics Consistency: How well clusters align with physical parameters (higher is better, range: 0 to 1)")
                            ], style={"fontSize": "11px", "paddingLeft": "10px"})
                        ])
                    ], style={"marginTop": "10px"})
                )
                
            except Exception as e:
                import traceback
                trace = traceback.format_exc()
                metrics_children = [html.Div(f"Error calculating metrics: {str(e)}")]
            
            return fig, "<br>".join(debug_text), mi_features_store, "UMAP visualization complete!", metrics_children
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            error_message = f"Error running UMAP on MI features: {str(e)}<br>{trace}"
            return {}, error_message, mi_features_store, "", []
    
    # If we're here, we're running the full MI feature selection
    if trigger_id == 'run-mi-features':
        debug_text.append(f"Data source: {data_source}")
        debug_text.append(f"Target variables: {target_variables}")
        debug_text.append(f"Redundancy threshold: {redundancy_threshold}")
        debug_text.append(f"Maximum features: {max_features}")
        
        try:
            # Step 1: Prepare data based on selected source
            # -----------------------------------------
            
            # Load the combined dataframe
            if combined_data_json and "combined_df" in combined_data_json and combined_data_json["combined_df"] != "{}":
                combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
                debug_text.append(f"Loaded combined dataset with {len(combined_df)} rows")
            else:
                return {}, "No combined dataset available. Please run UMAP first.", {}, "", []
            
            # Load UMAP coordinates
            umap_coords = None
            if "umap_coords" in combined_data_json and combined_data_json["umap_coords"] != "{}":
                umap_coords = pd.read_json(combined_data_json["umap_coords"], orient='split')
            
            # Load Graph 3 subset
            graph3_subset_df = None
            if combined_data_json and "graph3_subset" in combined_data_json and combined_data_json["graph3_subset"] != "{}":
                graph3_subset_df = pd.read_json(combined_data_json["graph3_subset"], orient='split')
                debug_text.append(f"Loaded Graph 3 subset with {len(graph3_subset_df)} rows")
            
            # Collect all selected features for MI analysis
            all_selected_features = []
            for features in selected_features_list:
                if features:  # Only add non-empty lists
                    all_selected_features.extend(features)
            
            if not all_selected_features:
                # Use particle momentum as default if nothing is selected
                all_selected_features = [col for col in combined_df.columns if col.startswith('particle_')]
                debug_text.append(f"No features selected, using {len(all_selected_features)} default momentum features")
            else:
                debug_text.append(f"Using {len(all_selected_features)} selected features")
            
            # Verify that target variables are valid
            if not target_variables or len(target_variables) == 0:
                # Default targets if none are specified
                target_variables = ["KER", "EESum", "TotalEnergy"]
                debug_text.append(f"Using default target variables: {', '.join(target_variables)}")
            
            # Check if targets exist in the data
            valid_targets = [t for t in target_variables if t in combined_df.columns]
            if not valid_targets:
                return {}, "No valid target variables found in the data. Please run feature extraction first.", {}, "", []
            
            debug_text.append(f"Using valid targets: {', '.join(valid_targets)}")
            
            # Prepare data based on data source
            if data_source == 'all':
                # Use all data
                df_for_analysis = combined_df.copy()
                labels = combined_df['file_label'].copy()
                debug_text.append(f"Using all {len(df_for_analysis)} rows for analysis")
            
            elif data_source == 'graph1-selection' and graph1_selection and umap_coords is not None:
                # Use selection from Graph 1
                indices = extract_selection_indices(graph1_selection, umap_coords)
                if not indices:
                    return {}, "No valid points found in Graph 1 selection.", {}, "", []
                
                df_for_analysis = combined_df.iloc[indices].copy()
                labels = df_for_analysis['file_label'].copy()
                debug_text.append(f"Selected {len(df_for_analysis)} rows from Graph 1 selection")
            
            elif data_source == 'graph3-selection' and graph3_selection and graph3_subset_df is not None:
                # Use selection from Graph 3
                df_for_analysis = graph3_subset_df.copy()
                labels = df_for_analysis['file_label'].copy()
                debug_text.append(f"Using Graph 3 selection with {len(df_for_analysis)} rows")
            
            else:
                # Default to all data
                df_for_analysis = combined_df.copy()
                labels = combined_df['file_label'].copy()
                debug_text.append(f"Defaulting to all {len(df_for_analysis)} rows")
            
            # Extract feature data - only include columns that are in all_selected_features
            feature_cols = [col for col in df_for_analysis.columns if col in all_selected_features]
            
            if not feature_cols:
                return {}, "No valid features selected for MI analysis.", {}, "", []
            
            # Create feature matrix for MI analysis
            feature_matrix = df_for_analysis[feature_cols].copy()
            
            # Handle NaN/inf values
            feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
            feature_matrix = feature_matrix.fillna(0)
            
            # Step 2: Mutual Information Feature Selection
            # -----------------------------------------
            from sklearn.feature_selection import mutual_info_regression
            from itertools import combinations
            
            debug_text.append("Computing mutual information with target variables...")
            
            # Compute MI with each target
            mi_scores = {}
            for target in valid_targets:
                target_values = df_for_analysis[target].values
                mi_scores[target] = mutual_info_regression(
                    feature_matrix, 
                    target_values,
                    random_state=42
                )
            
            # Average MI scores across all targets
            avg_mi_scores = np.mean([mi_scores[target] for target in valid_targets], axis=0)
            mi_scores_dict = dict(zip(feature_matrix.columns, avg_mi_scores))
            
            # Sort features by MI score (highest to lowest)
            sorted_features = sorted(mi_scores_dict, key=mi_scores_dict.get, reverse=True)
            
            debug_text.append(f"Top 5 features by mutual information:")
            for i, feature in enumerate(sorted_features[:5]):
                debug_text.append(f"{i+1}. {feature}: {mi_scores_dict[feature]:.4f}")
            
            # Compute pairwise MI between features (to remove redundancy)
            debug_text.append("Computing pairwise mutual information to reduce redundancy...")
            
            # Limit number of features to consider for pairwise MI to avoid excessive computation
            top_k_features = sorted_features[:min(100, len(sorted_features))]
            
            pairwise_mi = {}
            for f1, f2 in combinations(top_k_features, 2):
                pairwise_mi[(f1, f2)] = mutual_info_regression(
                    feature_matrix[[f1]], 
                    feature_matrix[f2].values,
                    random_state=42
                )[0]
            
            # Select features with maximum MI & minimum redundancy
            debug_text.append(f"Selecting non-redundant features (threshold={redundancy_threshold})...")
            
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
            
            debug_text.append(f"Selected {len(selected_features)} non-redundant features out of {len(feature_cols)}")
            
            # Create compressed feature matrix
            compressed_feature_matrix = feature_matrix[selected_features]
            
            # Step 3: Train autoencoder on selected features
            # -----------------------------------------
            debug_text.append(f"Training autoencoder with latent dimension {latent_dim}...")
            
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
            model = DeepAutoencoder(input_dim=num_features, latent_dim=int(latent_dim)).to(device)
            
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
                if epoch == 0 or epoch == num_epochs-1 or (epoch+1) % max(1, num_epochs//5) == 0:
                    debug_text.append(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            debug_text.append("Training complete!")
            
            # Extract latent representations
            model.eval()
            with torch.no_grad():
                latent_features = model.encoder(feature_tensor.to(device)).cpu().numpy()
            
            # Create dataframe with latent features
            latent_df = pd.DataFrame(latent_features, columns=[f"Latent_{i}" for i in range(latent_dim)])
            latent_df['file_label'] = labels.values
            
            # Store the latent features for future use
            mi_features_store = {
                'latent_features': latent_df.to_json(date_format='iso', orient='split'),
                'selected_features': selected_features,
                'feature_cols': feature_cols,
                'latent_dim': latent_dim,
                'mi_scores': mi_scores_dict
            }
            
            debug_text.append(f"Extracted {len(latent_df)} latent representations with dimension {latent_dim}")
            
            # Create a placeholder figure until UMAP is run
            placeholder_fig = {
                'data': [],
                'layout': {
                    'title': 'MI feature selection and autoencoder training complete! Click "Run UMAP on MI Features" to visualize',
                    'xaxis': {'title': 'UMAP1'},
                    'yaxis': {'title': 'UMAP2'},
                    'height': 600
                }
            }
            
            return placeholder_fig, "<br>".join(debug_text), mi_features_store, "MI feature selection and autoencoder training complete!", []
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            error_message = f"Error in MI feature selection: {str(e)}<br>{trace}"
            return {}, error_message, {}, "", []
    
    # If neither button was properly triggered, return empty states
    return {}, "Click 'Run MI Feature Selection' to start.", {}, "", []

@app.callback(
    Output('file-selector-graph15', 'options'),
    Output('file-selector-graph15', 'value'),
    Input('stored-files', 'data')
)
def update_file_selector_graph15(stored_files):
    """Update the file selector dropdown for Graph 1.5 based on uploaded files."""
    if not stored_files:
        return [], []
    
    options = [{'label': f['filename'], 'value': f['id']} for f in stored_files]
    values = [f['id'] for f in stored_files]  # Select all files by default
    
    return options, values

# Callback to update the feature dropdowns when data is available
@app.callback(
    Output('x-axis-feature-graph15', 'options'),
    Output('y-axis-feature-graph15', 'options'),
    Input('features-data-store', 'data'),
    prevent_initial_call=True
)
def update_feature_dropdowns_graph15(features_data):
    """Update the dropdown options for custom feature plot in Graph 1.5."""
    if not features_data or 'column_names' not in features_data:
        return [], []
    
    # Get all feature columns
    feature_columns = features_data['column_names']
    
    # Create dropdown options
    options = [{'label': col, 'value': col} for col in feature_columns]
    
    return options, options

# Callback to toggle visualization settings for Graph 1.5
@app.callback(
    [Output('heatmap-settings-graph15', 'style'),
     Output('scatter-settings-graph15', 'style')],
    [Input('visualization-type-graph15', 'value')]
)
def toggle_visualization_settings_graph15(visualization_type):
    """Show/hide appropriate settings based on visualization type for Graph 1.5."""
    if visualization_type == 'heatmap':
        return {'display': 'block', 'marginTop': '10px'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block', 'marginTop': '10px'}

# Callback to store selected points from Graph 1.5
@app.callback(
    Output('selected-points-store-graph15', 'data'),
    Output('selected-points-info-graph15', 'children'),
    Input('scatter-graph15', 'selectedData'),
    prevent_initial_call=True
)
def store_selected_points_graph15(selectedData):
    """Store the selected points from Graph 1.5."""
    if not selectedData:
        return [], "No points selected."
    
    selection_type = ""
    num_points = 0
    
    # Handle box selection
    if 'range' in selectedData:
        x_range = selectedData['range']['x']
        y_range = selectedData['range']['y']
        selection_type = f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"
    
    # Handle lasso selection
    elif 'lassoPoints' in selectedData:
        selection_type = "Lasso selection"
    
    # Handle individual point selection
    if 'points' in selectedData:
        num_points = len(selectedData['points'])
    
    info_text = [
        html.Div(f"Selection type: {selection_type}"),
        html.Div(f"Number of points: {num_points}")
    ]
    
    return selectedData, info_text

# Main callback for Graph 1.5 scatter plot
@app.callback(
    Output('scatter-graph15', 'figure'),
    Output('debug-output-graph15', 'children'),
    Output('quality-metrics-graph15', 'children'),
    Input('generate-plot-graph15', 'n_clicks'),
    State('stored-files', 'data'),
    State('file-selector-graph15', 'value'),
    State('x-axis-feature-graph15', 'value'),
    State('y-axis-feature-graph15', 'value'),
    State('sample-frac-graph15', 'value'),
    State('visualization-type-graph15', 'value'),
    State('point-opacity-graph15', 'value'),
    State('heatmap-bandwidth-graph15', 'value'),
    State('heatmap-colorscale-graph15', 'value'),
    State('show-points-overlay-graph15', 'value'),
    State('color-mode-graph15', 'value'),
    prevent_initial_call=True
)
def update_scatter_graph15(n_clicks, stored_files, selected_ids, x_feature, y_feature, 
                          sample_frac, visualization_type, point_opacity, 
                          heatmap_bandwidth, heatmap_colorscale, show_points_overlay, 
                          color_mode):
    """Generate custom scatter plot for Graph 1.5."""
    if not stored_files:
        return {}, "No files uploaded.", []
    
    if not selected_ids:
        return {}, "No files selected for plotting.", []
    
    if not x_feature or not y_feature:
        return {}, "Please select both X and Y axis features.", []
    
    try:
        debug_text = []
        debug_text.append(f"X-axis feature: {x_feature}")
        debug_text.append(f"Y-axis feature: {y_feature}")
        debug_text.append(f"Visualization type: {visualization_type}")
        
        # Process selected files
        sampled_dfs = []
        debug_str = ""
        
        for f in stored_files:
            if f['id'] in selected_ids:
                try:
                    df = pd.read_json(f['data'], orient='split')
                    is_selection = f.get('is_selection', False)
                    
                    if is_selection:
                        # This is a saved selection file
                        debug_str += f"{f['filename']}: Selection file with {len(df)} events.<br>"
                        df['file_label'] = f['filename']  # Make sure it has a file_label
                        sampled_dfs.append(df)
                    else:
                        # Regular COLTRIMS file
                        df['file_label'] = f['filename']  # Add file name as a label
                        
                        # Calculate physics features for this file's data
                        df_with_features = calculate_physics_features(df)
                        
                        # Sample the data to reduce processing time
                        sample_size = max(int(len(df_with_features) * sample_frac), 100)  # Ensure at least 100 points
                        if len(df_with_features) > sample_size:
                            sampled = df_with_features.sample(n=sample_size, random_state=42)
                        else:
                            sampled = df_with_features
                        
                        debug_str += f"{f['filename']}: {len(df)} events, sampled {len(sampled)}.<br>"
                        sampled_dfs.append(sampled)
                except Exception as e:
                    debug_str += f"Error processing {f['filename']}: {str(e)}.<br>"
        
        # Combine all selected datasets
        if not sampled_dfs:
            return {}, "No valid data to plot.", []
        
        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        debug_str += f"Combined data shape: {combined_df.shape}.<br>"
        
        # Verify features exist in dataset
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return {}, f"Features {x_feature} or {y_feature} not found in dataset.", []
        
        # Apply DBSCAN for cluster coloring (if needed)
        cluster_labels = None
        best_eps = 0.5  # Default value
        
        if color_mode == 'cluster':
            debug_text.append("Applying DBSCAN clustering for coloring")
            
            # Extract only the two features we're plotting
            X_features = combined_df[[x_feature, y_feature]].to_numpy()
            
            # Handle NaN/inf values
            X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Standardize the data
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import DBSCAN
            
            scaler = StandardScaler()
            X_features_scaled = scaler.fit_transform(X_features)
            
            # Find a reasonable epsilon for DBSCAN
            eps_candidates = np.linspace(0.1, 1.0, 10)
            best_eps = 0.5  # Default
            max_clusters = 0
            
            # Try different eps values and pick the one that gives a reasonable number of clusters
            for eps in eps_candidates:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(X_features_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                noise_count = np.sum(labels == -1)
                noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0
                
                if n_clusters >= 2 and noise_ratio < 0.5 and n_clusters > max_clusters:
                    max_clusters = n_clusters
                    best_eps = eps
            
            # Run DBSCAN with the best eps
            dbscan = DBSCAN(eps=best_eps, min_samples=5)
            cluster_labels = dbscan.fit_predict(X_features_scaled)
            
            debug_text.append(f"DBSCAN identified {max_clusters} clusters with eps={best_eps}")
        
        # Create color maps for file labels and clusters
        import plotly.graph_objects as go
        
        # Create color map based on file labels
        unique_labels = combined_df['file_label'].unique()
        colorscale = px.colors.qualitative.Plotly
        color_map = {label: colorscale[i % len(colorscale)] for i, label in enumerate(unique_labels)}
        
        # Create color map for clusters if needed
        cluster_colors = {}
        if color_mode == 'cluster' and cluster_labels is not None:
            # Special handling for noise points (-1 label)
            unique_clusters = sorted(set(cluster_labels))
            if -1 in unique_clusters:
                # Move noise to the end
                unique_clusters.remove(-1)
                unique_clusters.append(-1)
            
            # Choose color scale for clusters
            if len(unique_clusters) <= 10:
                cluster_colorscale = px.colors.qualitative.D3  # Good for distinct clusters
            else:
                # For many clusters, use a continuous colorscale
                cluster_colorscale = px.colors.sequential.Viridis
            
            # Create color mapping for clusters
            for i, cluster in enumerate(unique_clusters):
                if cluster == -1:  # Noise points
                    cluster_colors[cluster] = 'rgba(150,150,150,0.5)'  # Gray for noise
                else:
                    # Regular clusters
                    if len(unique_clusters) - (1 if -1 in unique_clusters else 0) <= 10:
                        colorscale_idx = i % len(cluster_colorscale)
                        cluster_colors[cluster] = cluster_colorscale[colorscale_idx]
                    else:
                        # For many clusters, distribute colors evenly
                        n_real_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                        idx = i / (n_real_clusters - 1) if n_real_clusters > 1 else 0
                        idx = min(0.99, max(0, idx))  # Ensure it's between 0 and 1
                        color_idx = int(idx * (len(cluster_colorscale) - 1))
                        cluster_colors[cluster] = cluster_colorscale[color_idx]
        
        # Initialize figure
        fig = go.Figure()
        
        # Heatmap visualization
        if visualization_type == 'heatmap':
            from scipy.stats import gaussian_kde
            
            # Get coordinates for heatmap
            x_data = combined_df[x_feature].values
            y_data = combined_df[y_feature].values
            
            # Create the grid for the heatmap
            x_min, x_max = np.min(x_data) - 0.05 * (np.max(x_data) - np.min(x_data)), np.max(x_data) + 0.05 * (np.max(x_data) - np.min(x_data))
            y_min, y_max = np.min(y_data) - 0.05 * (np.max(y_data) - np.min(y_data)), np.max(y_data) + 0.05 * (np.max(y_data) - np.min(y_data))
            
            # Create a meshgrid
            grid_size = 200
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            xx, yy = np.meshgrid(x_grid, y_grid)
            grid_points = np.column_stack([xx.flatten(), yy.flatten()])
            
            # Compute KDE (Kernel Density Estimation)
            data_points = np.column_stack([x_data, y_data])
            kde = gaussian_kde(data_points.T, bw_method=heatmap_bandwidth)
            densities = kde(grid_points.T).reshape(grid_size, grid_size)
            
            # Add heatmap
            fig.add_trace(go.Heatmap(
                x=x_grid,
                y=y_grid,
                z=densities,
                colorscale=heatmap_colorscale,
                showscale=True,
                colorbar=dict(title='Density'),
                hoverinfo='none'
            ))
            
            # Optionally, overlay scatter points with reduced opacity for context
            if show_points_overlay == 'yes':
                if color_mode == 'file':
                    # Color by file source with reduced opacity
                    for label in unique_labels:
                        mask = combined_df['file_label'] == label
                        df_subset = combined_df[mask]
                        
                        fig.add_trace(go.Scatter(
                            x=df_subset[x_feature],
                            y=df_subset[y_feature],
                            mode='markers',
                            marker=dict(
                                size=4,  # Smaller points
                                color=color_map[label],
                                opacity=0.3,  # Reduced opacity
                                line=dict(width=0)
                            ),
                            name=f"{label} ({len(df_subset)} pts)"
                        ))
                elif color_mode == 'cluster' and cluster_labels is not None:
                    # Color by DBSCAN cluster with reduced opacity
                    for cluster in sorted(set(cluster_labels)):
                        mask = cluster_labels == cluster
                        cluster_points = combined_df.iloc[mask]
                        
                        # For noise points, make them smaller and more transparent
                        marker_size = 3 if cluster == -1 else 4
                        marker_opacity = 0.2 if cluster == -1 else 0.3
                        
                        fig.add_trace(go.Scatter(
                            x=cluster_points[x_feature],
                            y=cluster_points[y_feature],
                            mode='markers',
                            marker=dict(
                                size=marker_size,
                                color=cluster_colors[cluster],
                                opacity=marker_opacity,
                                line=dict(width=0)
                            ),
                            name=f"Cluster {cluster if cluster != -1 else 'Noise'} ({len(cluster_points)} pts)"
                        ))
        else:
            # Scatter plot visualization
            if color_mode == 'file':
                # Color by file source
                for label in unique_labels:
                    mask = combined_df['file_label'] == label
                    df_subset = combined_df[mask]
                    
                    fig.add_trace(go.Scatter(
                        x=df_subset[x_feature],
                        y=df_subset[y_feature],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_map[label],
                            opacity=point_opacity,
                            line=dict(width=0)
                        ),
                        name=f"{label} ({len(df_subset)} pts)"
                    ))
            
            elif color_mode == 'cluster' and cluster_labels is not None:
                # Color by DBSCAN cluster
                for cluster in sorted(set(cluster_labels)):
                    mask = cluster_labels == cluster
                    cluster_points = combined_df.iloc[mask]
                    
                    # For noise points, make them smaller and more transparent
                    marker_size = 5 if cluster == -1 else 8
                    marker_opacity = point_opacity * 0.7 if cluster == -1 else point_opacity
                    
                    fig.add_trace(go.Scatter(
                        x=cluster_points[x_feature],
                        y=cluster_points[y_feature],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            color=cluster_colors[cluster],
                            opacity=marker_opacity,
                            line=dict(width=0)
                        ),
                        name=f"Cluster {cluster if cluster != -1 else 'Noise'} ({len(cluster_points)} pts)"
                    ))
        
        # Update figure layout
        title_suffix = ""
        if color_mode == 'cluster' and cluster_labels is not None:
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = np.sum(cluster_labels == -1)
            title_suffix = f" - {n_clusters} clusters detected"
        
        # Adjust legend position based on number of traces
        legend_y_position = -0.5 if len(fig.data) > 12 else -0.4 if len(fig.data) > 8 else -0.3
        
        # Create legend configuration
        legend_config = dict(
            orientation="h",
            yanchor="top",     # Anchor to the top of the legend box
            y=legend_y_position,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",  # More opaque background for readability
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=10),  # Smaller font for many items
            itemwidth=30,       # Smaller item width for more compact, richer color symbols
            itemsizing="constant",
            tracegroupgap=5     # Reduced gap between legend groups
        )
        
        # Adjust figure height based on legend size
        figure_height = 600
        if len(fig.data) > 15:
            figure_height = 750
        elif len(fig.data) > 10:
            figure_height = 700
        elif len(fig.data) > 6:
            figure_height = 650
        
        # Apply the layout settings
        if visualization_type == 'heatmap':
            title = f"Custom Feature Heatmap: {x_feature} vs {y_feature}{title_suffix}"
        else:
            title = f"Custom Feature Scatter Plot: {x_feature} vs {y_feature}{title_suffix}"
            
        fig.update_layout(
            height=figure_height,
            title=title,
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title=f"{'Clusters' if color_mode == 'cluster' else 'Data File'}",
            dragmode='lasso',  # Explicitly set lasso as default selection mode
            legend=legend_config,
            modebar=dict(add=['lasso2d', 'select2d']),  # Add these tools to the modebar
            margin=dict(l=50, r=50, t=50, b=100)  # Increased bottom margin for legend
        )
        
        # Simplified cluster information (no complex metrics)
        metrics_children = []
        
        if cluster_labels is not None and color_mode == 'cluster':
            # Count clusters and noise points
            unique_clusters = set(cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_noise = np.sum(cluster_labels == -1)
            noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            
            # Create simple cluster info UI
            metrics_children = [
                html.H4("Clustering Information", style={"fontSize": "14px", "marginBottom": "5px"}),
                html.Div([
                    html.Div([
                        html.Span("Clusters Detected: ", style={"fontWeight": "bold"}),
                        html.Span(f"{n_clusters}")
                    ]),
                    html.Div([
                        html.Span("Noise Points: ", style={"fontWeight": "bold"}),
                        html.Span(f"{n_noise} ({noise_ratio:.1%})")
                    ]),
                    html.Div([
                        html.Span("DBSCAN Epsilon: ", style={"fontWeight": "bold"}),
                        html.Span(f"{best_eps:.3f}")
                    ])
                ])
            ]
        
        return fig, debug_str, metrics_children
    
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return {}, f"Error: {str(e)}<br><pre>{trace}</pre>", []


# Callback to download selected points from Graph 1.5
@app.callback(
    Output('download-selection-graph15', 'data'),
    Input('save-selection-graph15-btn', 'n_clicks'),
    State('selection-filename-graph15', 'value'),
    State('selected-points-store-graph15', 'data'),
    State('x-axis-feature-graph15', 'value'),
    State('y-axis-feature-graph15', 'value'),
    State('scatter-graph15', 'figure'),  # Add the figure data
    State('file-selector-graph15', 'value'),
    State('stored-files', 'data'),
    prevent_initial_call=True
)
def download_selected_points_graph15(n_clicks, filename, selectedData, x_feature, y_feature, figure_data, selected_ids, stored_files):
    """Generate CSV file of selected points from Graph 1.5 for download."""
    if not n_clicks or not filename or not selectedData:
        raise dash.exceptions.PreventUpdate
    
    print(f"Save button clicked for Graph 1.5, filename: {filename}")
    
    try:
        # Process selected files to get the combined dataframe
        sampled_dfs = []
        
        # This part is different from Graph 1 - need to build combined dataset
        for f in stored_files:
            if f['id'] in selected_ids:
                try:
                    df = pd.read_json(f['data'], orient='split')
                    is_selection = f.get('is_selection', False)
                    
                    df['file_label'] = f['filename']  # Add file name as a label
                    
                    if not is_selection:
                        # Calculate physics features
                        df = calculate_physics_features(df)
                    
                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']} for download: {str(e)}")
        
        # Combine datasets
        if not sampled_dfs:
            print("No valid files selected")
            raise dash.exceptions.PreventUpdate
        
        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        print(f"Combined dataframe shape: {combined_df.shape}")
        
        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            print(f"Features {x_feature} or {y_feature} not found in data")
            raise dash.exceptions.PreventUpdate
        
        # Extract selected points using the coordinates - this is the key part
        selected_indices = []
        
        # Print debug info about selectedData
        print(f"Selected data type: {type(selectedData)}")
        print(f"Selected data content: {selectedData}")
        
        # Handle box selection
        if 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            
            print(f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]")
            
            # Use masked selection
            selected_mask = (
                (combined_df[x_feature] >= x_range[0]) & 
                (combined_df[x_feature] <= x_range[1]) & 
                (combined_df[y_feature] >= y_range[0]) & 
                (combined_df[y_feature] <= y_range[1])
            )
            selected_indices = np.where(selected_mask)[0].tolist()
            print(f"Found {len(selected_indices)} points in box selection")
        
        # Handle lasso selection
        elif 'lassoPoints' in selectedData:
            from matplotlib.path import Path
            # Extract lasso polygon coordinates
            lasso_x = selectedData['lassoPoints']['x']
            lasso_y = selectedData['lassoPoints']['y']
            
            print(f"Lasso selection with {len(lasso_x)} points")
            
            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            
            # Check which points are within the lasso path
            points_array = np.column_stack([combined_df[x_feature].values, combined_df[y_feature].values])
            inside_lasso = lasso_path.contains_points(points_array)
            
            # Get indices of points inside the lasso
            selected_indices = np.where(inside_lasso)[0].tolist()
            print(f"Found {len(selected_indices)} points in lasso selection")
        
        # Handle direct point selection
        elif 'points' in selectedData and selectedData['points']:
            print(f"Direct selection with {len(selectedData['points'])} points")
            
            # Try to extract points directly from the selection
            for point in selectedData['points']:
                print(f"Point data: {point}")
                # Try different ways to get index
                if 'pointIndex' in point:
                    selected_indices.append(point['pointIndex'])
                elif 'customdata' in point and point['customdata']:
                    # Some plots store index in customdata
                    selected_indices.append(point['customdata'])
                else:
                    # Find by coordinates
                    x_val = point.get('x')
                    y_val = point.get('y')
                    
                    if x_val is not None and y_val is not None:
                        # Find closest point
                        distances = ((combined_df[x_feature] - x_val)**2 + 
                                    (combined_df[y_feature] - y_val)**2)
                        closest_idx = distances.idxmin()
                        selected_indices.append(closest_idx)
        
        if not selected_indices:
            print("No valid indices found")
            raise dash.exceptions.PreventUpdate
        
        print(f"Processing {len(selected_indices)} selected points")
        
        # Extract the selected points
        selected_df = combined_df.iloc[selected_indices].reset_index(drop=True)
        
        # Remove UMAP coordinates if they exist
        if 'UMAP1' in selected_df.columns:
            selected_df = selected_df.drop(columns=['UMAP1'])
        if 'UMAP2' in selected_df.columns:
            selected_df = selected_df.drop(columns=['UMAP2'])
        
        # Remove file_label
        if 'file_label' in selected_df.columns:
            selected_df = selected_df.drop(columns=['file_label'])
        
        # Get only the particle momentum columns (original data format)
        momentum_columns = [col for col in selected_df.columns if col.startswith('particle_')]
        
        # Convert to original format
        if momentum_columns and len(momentum_columns) == 15:  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                'Px_ion1', 'Py_ion1', 'Pz_ion1',
                'Px_ion2', 'Py_ion2', 'Pz_ion2',
                'Px_neutral', 'Py_neutral', 'Pz_neutral',
                'Px_electron1', 'Py_electron1', 'Pz_electron1',
                'Px_electron2', 'Py_electron2', 'Pz_electron2'
            ]
            
            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = selected_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = selected_df
        
        print(f"Final export dataframe shape: {original_format_df.shape}")
        print(f"Final export dataframe columns: {original_format_df.columns.tolist()}")
        
        # Return the dataframe as a CSV for download
        # The key difference is we're only returning the download output here
        return dcc.send_data_frame(
            original_format_df.to_csv, 
            f"{filename}.csv",
            index=False
        )
        
    except Exception as e:
        print(f"Error saving Graph 1.5 selection: {e}")
        import traceback
        trace = traceback.format_exc()
        print(trace)
        raise dash.exceptions.PreventUpdate
@app.callback(
    Output('graph25', 'figure'),
    Output('debug-output-graph25', 'children'),
    Output('selected-points-info-graph25', 'children'),
    Input('show-selected-graph15', 'n_clicks'),
    State('selected-points-store-graph15', 'data'),
    State('x-axis-feature-graph15', 'value'),
    State('y-axis-feature-graph15', 'value'),
    State('scatter-graph15', 'figure'),
    State('file-selector-graph15', 'value'),
    State('stored-files', 'data'),
    prevent_initial_call=True
)
def update_graph25(n_clicks, selectedData, x_feature, y_feature, 
                  original_figure, selected_ids, stored_files):
    """Display the selected points from Graph 1.5."""
    if not n_clicks or not selectedData:
        return {}, "No points selected.", "No points selected."
    
    debug_text = []
    
    try:
        # Process selected files to get the combined dataframe
        sampled_dfs = []
        
        for f in stored_files:
            if f['id'] in selected_ids:
                try:
                    df = pd.read_json(f['data'], orient='split')
                    is_selection = f.get('is_selection', False)
                    
                    df['file_label'] = f['filename']  # Add file name as a label
                    
                    if not is_selection:
                        # Calculate physics features
                        df = calculate_physics_features(df)
                    
                    sampled_dfs.append(df)
                except Exception as e:
                    debug_text.append(f"Error processing {f['filename']}: {str(e)}")
        
        # Combine datasets
        if not sampled_dfs:
            return {}, "No valid files selected.", "No valid files selected."
        
        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        debug_text.append(f"Combined dataframe shape: {combined_df.shape}")
        
        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return {}, f"Features {x_feature} or {y_feature} not found in data.", "Features not found."
        
        # Extract selected points
        selected_indices = []
        
        # Handle box selection
        if 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            
            debug_text.append(f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]")
            
            selected_mask = (
                (combined_df[x_feature] >= x_range[0]) & 
                (combined_df[x_feature] <= x_range[1]) & 
                (combined_df[y_feature] >= y_range[0]) & 
                (combined_df[y_feature] <= y_range[1])
            )
            selected_indices = np.where(selected_mask)[0].tolist()
            debug_text.append(f"Found {len(selected_indices)} points in box selection")
        
        # Handle lasso selection
        elif 'lassoPoints' in selectedData:
            from matplotlib.path import Path
            # Extract lasso polygon coordinates
            lasso_x = selectedData['lassoPoints']['x']
            lasso_y = selectedData['lassoPoints']['y']
            
            debug_text.append(f"Lasso selection with {len(lasso_x)} points")
            
            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            
            # Check which points are within the lasso path
            points_array = np.column_stack([combined_df[x_feature].values, combined_df[y_feature].values])
            inside_lasso = lasso_path.contains_points(points_array)
            
            # Get indices of points inside the lasso
            selected_indices = np.where(inside_lasso)[0].tolist()
            debug_text.append(f"Found {len(selected_indices)} points in lasso selection")
        
        # Handle direct point selection
        elif 'points' in selectedData and selectedData['points']:
            debug_text.append(f"Direct selection with {len(selectedData['points'])} points")
            
            for point in selectedData['points']:
                x_val = point.get('x')
                y_val = point.get('y')
                
                if x_val is not None and y_val is not None:
                    # Find the closest point in the dataset
                    distances = (combined_df[x_feature] - x_val)**2 + (combined_df[y_feature] - y_val)**2
                    closest_idx = distances.idxmin()
                    selected_indices.append(closest_idx)
        
        if not selected_indices:
            return {}, "No points found in the selection area.", "No points found."
        
        # Extract the selected points
        selected_df = combined_df.iloc[selected_indices].reset_index(drop=True)
        debug_text.append(f"Selected {len(selected_df)} points")
        
        # Create visualization
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Extract color information from original figure
        color_map = {}
        if original_figure and 'data' in original_figure:
            for trace in original_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    # Clean the label if it contains point count
                    clean_name = trace['name']
                    if ' (' in clean_name:
                        clean_name = clean_name.split(' (')[0]
                    color_map[clean_name] = trace['marker']['color']
        
        # Add traces for each file label
        for label in selected_df['file_label'].unique():
            mask = selected_df['file_label'] == label
            df_subset = selected_df[mask]
            
            # Get color from original figure if available
            color = color_map.get(label, None)
            
            fig.add_trace(go.Scatter(
                x=df_subset[x_feature],
                y=df_subset[y_feature],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                name=f"{label} ({len(df_subset)} pts)"
            ))
        
        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"Selected Points from Graph 1.5: {x_feature} vs {y_feature}",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title="Data File"
        )
        
        # Count points by file for information panel
        file_counts = selected_df['file_label'].value_counts().to_dict()
        
        # Create info text
        info_text = [
            html.Div(f"Total selected points: {len(selected_df)}"),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div([html.Div(f"{file}: {count} events") for file, count in file_counts.items()], 
                    style={"marginLeft": "10px"})
        ]
        
        return fig, "<br>".join(debug_text), info_text
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return {}, f"Error: {str(e)}<br><pre>{trace}</pre>", f"Error: {str(e)}"

@app.callback(
    Output('download-selection-graph25', 'data'),
    Input('save-selection-graph25-btn', 'n_clicks'),
    State('selection-filename-graph25', 'value'),
    State('selected-points-store-graph15', 'data'),  # Use the data from Graph 1.5
    State('x-axis-feature-graph15', 'value'),
    State('y-axis-feature-graph15', 'value'),
    State('file-selector-graph15', 'value'),
    State('stored-files', 'data'),
    prevent_initial_call=True
)
def download_selected_points_graph25(n_clicks, filename, selectedData, x_feature, y_feature, selected_ids, stored_files):
    """Generate CSV file of selected points from Graph 2.5 for download."""
    if not n_clicks or not filename or not selectedData:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Process selected files to get the combined dataframe
        sampled_dfs = []
        
        for f in stored_files:
            if f['id'] in selected_ids:
                try:
                    df = pd.read_json(f['data'], orient='split')
                    is_selection = f.get('is_selection', False)
                    
                    df['file_label'] = f['filename']  # Add file name as a label
                    
                    if not is_selection:
                        # Calculate physics features
                        df = calculate_physics_features(df)
                    
                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']} for download: {str(e)}")
        
        # Combine datasets
        if not sampled_dfs:
            raise dash.exceptions.PreventUpdate
        
        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        
        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            raise dash.exceptions.PreventUpdate
        
        # Extract selected points
        selected_indices = []
        
        # Handle box selection
        if 'range' in selectedData:
            x_range = selectedData['range']['x']
            y_range = selectedData['range']['y']
            
            selected_mask = (
                (combined_df[x_feature] >= x_range[0]) & 
                (combined_df[x_feature] <= x_range[1]) & 
                (combined_df[y_feature] >= y_range[0]) & 
                (combined_df[y_feature] <= y_range[1])
            )
            selected_indices = np.where(selected_mask)[0].tolist()
        
        # Handle lasso selection
        elif 'lassoPoints' in selectedData:
            from matplotlib.path import Path
            # Extract lasso polygon coordinates
            lasso_x = selectedData['lassoPoints']['x']
            lasso_y = selectedData['lassoPoints']['y']
            
            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))
            
            # Check which points are within the lasso path
            points_array = np.column_stack([combined_df[x_feature].values, combined_df[y_feature].values])
            inside_lasso = lasso_path.contains_points(points_array)
            
            # Get indices of points inside the lasso
            selected_indices = np.where(inside_lasso)[0].tolist()
        
        # Handle direct point selection
        elif 'points' in selectedData and selectedData['points']:
            for point in selectedData['points']:
                x_val = point.get('x')
                y_val = point.get('y')
                
                if x_val is not None and y_val is not None:
                    # Find the closest point in the dataset
                    distances = (combined_df[x_feature] - x_val)**2 + (combined_df[y_feature] - y_val)**2
                    closest_idx = distances.idxmin()
                    selected_indices.append(closest_idx)
        
        if not selected_indices:
            raise dash.exceptions.PreventUpdate
        
        # Extract the selected points
        selected_df = combined_df.iloc[selected_indices].reset_index(drop=True)
        
        # Remove UMAP coordinates if they exist
        if 'UMAP1' in selected_df.columns:
            selected_df = selected_df.drop(columns=['UMAP1'])
        if 'UMAP2' in selected_df.columns:
            selected_df = selected_df.drop(columns=['UMAP2'])
        
        # Keep file_label for reference but prepare for export
        export_df = selected_df.copy()
        
        # Remove file_label which was added during processing
        if 'file_label' in export_df.columns:
            export_df = export_df.drop(columns=['file_label'])
        
        # Get only the particle momentum columns (original data format)
        momentum_columns = [col for col in export_df.columns if col.startswith('particle_')]
        
        # If we have the momentum columns, convert back to original format
        if momentum_columns and len(momentum_columns) == 15:  # Should be 5 particles x 3 dimensions
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                'Px_ion1', 'Py_ion1', 'Pz_ion1',
                'Px_ion2', 'Py_ion2', 'Pz_ion2',
                'Px_neutral', 'Py_neutral', 'Pz_neutral',
                'Px_electron1', 'Py_electron1', 'Pz_electron1',
                'Px_electron2', 'Py_electron2', 'Pz_electron2'
            ]
            
            # Extract and rename the momentum columns
            original_format_df = pd.DataFrame()
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = export_df[col]
                    
            # Also include other physics features if available
            for col in export_df.columns:
                if not col.startswith('particle_'):
                    original_format_df[col] = export_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = export_df
        
        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, 
            f"{filename}.csv",
            index=False
        )
        
    except Exception as e:
        print(f"Error saving Graph 2.5 selection: {e}")
        import traceback
        traceback.print_exc()
        raise dash.exceptions.PreventUpdate

# Callback to show/hide parameter filter controls
@app.callback(
    Output('parameter-filter-controls', 'style'),
    Output('parameter-range-slider', 'min'),
    Output('parameter-range-slider', 'max'),
    Output('parameter-range-slider', 'marks'),
    Output('parameter-range-slider', 'value'),
    Input('physics-parameter-dropdown', 'value'),
    State('scatter-graph15', 'figure'),
    State('file-selector-graph15', 'value'),
    State('stored-files', 'data')
)
def update_parameter_filter_controls(selected_parameter, figure, selected_ids, stored_files):
    """Update the parameter filter controls based on the selected parameter."""
    if not selected_parameter:
        return {'display': 'none'}, 0, 100, {0: '0', 100: '100'}, [0, 100]
    
    # Get dataset parameter range
    try:
        # Load data to determine parameter range
        sampled_dfs = []
        for f in stored_files:
            if f['id'] in selected_ids:
                df = pd.read_json(f['data'], orient='split')
                if selected_parameter not in df.columns:
                    df = calculate_physics_features(df)
                if selected_parameter in df.columns:
                    sampled_dfs.append(df)
        
        if not sampled_dfs:
            return {'display': 'block'}, 0, 100, {0: '0', 100: '100'}, [0, 100]
        
        combined_df = pd.concat(sampled_dfs, ignore_index=True)
        if selected_parameter not in combined_df.columns:
            return {'display': 'block'}, 0, 100, {0: '0', 100: '100'}, [0, 100]
        
        param_min = float(combined_df[selected_parameter].min())
        param_max = float(combined_df[selected_parameter].max())
        
        # Round the values for better display
        param_min = np.floor(param_min)
        param_max = np.ceil(param_max)
        
        # Create marks dictionary
        num_steps = 5
        step_size = (param_max - param_min) / (num_steps - 1)
        marks = {param_min + i * step_size: f'{param_min + i * step_size:.1f}' 
                for i in range(num_steps)}
        
        return {'display': 'block'}, param_min, param_max, marks, [param_min, param_max]
    
    except Exception as e:
        print(f"Error updating parameter filter controls: {e}")
        return {'display': 'block'}, 0, 100, {0: '0', 100: '100'}, [0, 100]

# Callback to apply density filter
@app.callback(
    Output('filtered-data-graph', 'figure', allow_duplicate=True),
    Output('density-filter-status', 'children'),
    Output('density-filter-info', 'children'),
    Output('filtered-data-store', 'data', allow_duplicate=True),
    Input('apply-density-filter', 'n_clicks'),
    State('density-bandwidth-slider', 'value'),
    State('density-threshold-slider', 'value'),
    State('x-axis-feature-graph15', 'value'),
    State('y-axis-feature-graph15', 'value'),
    State('file-selector-graph15', 'value'),
    State('stored-files', 'data'),
    State('sample-frac-graph15', 'value'),  # Add sampling control
    prevent_initial_call=True
)
def apply_density_filter(n_clicks, bandwidth, threshold_percentile, 
                         x_feature, y_feature, selected_ids, stored_files, sample_frac):
    ctx = dash.callback_context
    if not ctx.triggered or 'apply-density-filter' not in ctx.triggered[0]['prop_id']:
        raise dash.exceptions.PreventUpdate
    
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    print(f"Density filter called with bandwidth={bandwidth}, threshold={threshold_percentile}")
    
    try:
        # Process selected files - WITH SAMPLING to reduce processing time
        sampled_dfs = []
        total_rows = 0
        sampled_rows = 0
        
        for f in stored_files:
            if f['id'] in selected_ids:
                try:
                    df = pd.read_json(f['data'], orient='split')
                    total_rows += len(df)
                    
                    is_selection = f.get('is_selection', False)
                    df['file_label'] = f['filename']  # Add file name as a label
                    
                    if not is_selection:
                        # Calculate physics features
                        df = calculate_physics_features(df)
                        
                        # Apply sampling to reduce processing time
                        if len(df) > 1000:
                            # Use the sample_frac from the UI or a default value
                            actual_sample_frac = sample_frac if sample_frac is not None else 0.1
                            sample_size = max(int(len(df) * actual_sample_frac), 500)  # Ensure reasonable number
                            df = df.sample(n=sample_size, random_state=42)
                            sampled_rows += len(df)
                        else:
                            sampled_rows += len(df)
                    else:
                        sampled_rows += len(df)
                    
                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']}: {str(e)}")
        
        # Combine datasets
        if not sampled_dfs:
            return {}, "No valid files selected.", "No data available.", {}
        
        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        print(f"Combined dataframe: {sampled_rows} sampled rows from {total_rows} total rows")
        
        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return {}, f"Features {x_feature} or {y_feature} not found in data.", "Features not found.", {}
        
        # Extract feature data for density calculation
        feature_data = combined_df[[x_feature, y_feature]].to_numpy()
        
        # Handle NaN/inf values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # OPTIMIZED METHOD: Use grid-based density estimation with fewer bins for large datasets
        # Adjust bins based on data size
        num_bins = min(100, max(20, int(np.sqrt(len(feature_data)))))  # Scale bins with data size
        print(f"Using {num_bins} bins for density estimation on {len(feature_data)} points")
        
        x_min, x_max = np.min(feature_data[:,0]), np.max(feature_data[:,0])
        y_min, y_max = np.min(feature_data[:,1]), np.max(feature_data[:,1])
        
        # Add small padding to min/max
        padding = 0.01
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        
        # OPTIMIZATION: Use a smaller sigma for large datasets
        sigma_scale = max(0.5, min(10, 10.0 * np.sqrt(1000.0 / len(feature_data))))
        actual_sigma = bandwidth * sigma_scale
        print(f"Using sigma {actual_sigma} for smoothing (bandwidth={bandwidth})")
        
        # Create 2D histogram - faster approach
        H, xedges, yedges = np.histogram2d(
            feature_data[:,0], feature_data[:,1], 
            bins=num_bins, 
            range=[[x_min, x_max], [y_min, y_max]]
        )
        
        # Apply smoothing with optimized sigma
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H, sigma=actual_sigma)
        
        # Assign density values to each point - vectorized approach
        x_indices = np.clip(
            np.floor((feature_data[:,0] - x_min) / (x_max - x_min) * num_bins).astype(int),
            0, num_bins-1
        )
        y_indices = np.clip(
            np.floor((feature_data[:,1] - y_min) / (y_max - y_min) * num_bins).astype(int),
            0, num_bins-1
        )
        
        densities = H_smooth[x_indices, y_indices]
        
        # Calculate density threshold
        threshold = np.percentile(densities, threshold_percentile)
        
        # Filter by density
        high_density_mask = densities >= threshold
        filtered_df = combined_df.iloc[high_density_mask].copy()
        filtered_df['density'] = densities[high_density_mask]  # Store density for reference
        
        print(f"Kept {len(filtered_df)} high-density points ({len(filtered_df)/len(combined_df):.1%} of original)")
        
        # Check if we have any points left
        if len(filtered_df) == 0:
            return {}, "No points remain after filtering.", "Try lowering the threshold.", {}
        
        # Create visualization of filtered data - OPTIMIZED RENDERING
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Use a standard color palette
        color_map = px.colors.qualitative.Plotly
        
        # OPTIMIZATION: Limit points displayed for better performance
        # For background points, randomly sample to reduce visual clutter
        if len(combined_df) > 5000:
            background_idx = np.random.choice(len(combined_df), size=5000, replace=False)
            background_df = combined_df.iloc[background_idx]
        else:
            background_df = combined_df
            
        # Add original data as background with reduced opacity
        fig.add_trace(go.Scatter(
            x=background_df[x_feature],
            y=background_df[y_feature],
            mode='markers',
            marker=dict(
                size=3,  # Smaller points for background
                color='gray',
                opacity=0.05  # Less opacity
            ),
            name='Original data',
            showlegend=True
        ))
        
        # Add traces for each file label
        for i, label in enumerate(filtered_df['file_label'].unique()):
            mask = filtered_df['file_label'] == label
            df_subset = filtered_df[mask]
            
            # If subset is very large, sample it for display
            if len(df_subset) > 2000:
                subset_idx = np.random.choice(len(df_subset), size=2000, replace=False)
                df_subset = df_subset.iloc[subset_idx]
                display_name = f"{label} ({len(filtered_df[mask])} pts, showing 2000)"
            else:
                display_name = f"{label} ({len(df_subset)} pts)"
            
            # Use standard color palette
            color = color_map[i % len(color_map)]
            
            fig.add_trace(go.Scatter(
                x=df_subset[x_feature],
                y=df_subset[y_feature],
                mode='markers',
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.7
                ),
                name=display_name
            ))
        
        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"Density-Filtered Data: {x_feature} vs {y_feature} (Kept {len(filtered_df)} points)",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title="Data Source"
        )
        
        # Store filtered data
        filtered_data_store = {
            'filtered_df': filtered_df.to_json(date_format='iso', orient='split'),
            'x_feature': x_feature,
            'y_feature': y_feature,
            'filtering_method': 'density',
            'params': {
                'bandwidth': bandwidth,
                'threshold_percentile': threshold_percentile,
                'threshold_value': float(threshold)
            }
        }
        
        # Create info text
        file_counts = filtered_df['file_label'].value_counts().to_dict()
        info_text = [
            html.Div(f"Total points after filtering: {len(filtered_df)} ({len(filtered_df)/len(combined_df):.1%} of original)"),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div([html.Div(f"{file}: {count} events") for file, count in file_counts.items()], 
                     style={"marginLeft": "10px"})
        ]
        
        return fig, "Density filtering applied successfully!", info_text, filtered_data_store
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Density filter error: {str(e)}\n{trace}")
        return {}, f"Error applying density filter: {str(e)}", f"Error: {str(e)}", {}
# Callback to download filtered data
@app.callback(
    Output('download-filtered-data', 'data'),
    Output('save-filtered-data-status', 'children'),
    Input('save-filtered-data-btn', 'n_clicks'),
    State('filtered-data-filename', 'value'),
    State('filtered-data-store', 'data'),
    prevent_initial_call=True
)
def download_filtered_data(n_clicks, filename, filtered_data_store):
    """Generate CSV file of filtered data points for download."""
    if not n_clicks or not filename or not filtered_data_store or 'filtered_df' not in filtered_data_store:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Load the filtered data
        filtered_df = pd.read_json(filtered_data_store['filtered_df'], orient='split')
        
        if filtered_df.empty:
            raise dash.exceptions.PreventUpdate
        
        # Remove columns not needed for export
        export_df = filtered_df.copy()
        
        # Remove density column which was added during filtering
        if 'density' in export_df.columns:
            export_df = export_df.drop(columns=['density'])
        
        # Remove file_label which was added during processing
        if 'file_label' in export_df.columns:
            export_df = export_df.drop(columns=['file_label'])
        
        # Get only the particle momentum columns (original data format)
        momentum_columns = [col for col in export_df.columns if col.startswith('particle_')]
        
        # If we have the momentum columns, convert back to original format
        original_format_df = pd.DataFrame()
        if momentum_columns and len(momentum_columns) == 15:
            # Create the reverse mapping from standardized to original column names
            reverse_columns = [
                'Px_ion1', 'Py_ion1', 'Pz_ion1',
                'Px_ion2', 'Py_ion2', 'Pz_ion2',
                'Px_neutral', 'Py_neutral', 'Pz_neutral',
                'Px_electron1', 'Py_electron1', 'Pz_electron1',
                'Px_electron2', 'Py_electron2', 'Pz_electron2'
            ]
            
            # Extract and rename the momentum columns
            for i, col in enumerate(momentum_columns):
                if i < len(reverse_columns):
                    original_format_df[reverse_columns[i]] = export_df[col]
                    
            # Also include other physics features if available
            for col in export_df.columns:
                if not col.startswith('particle_'):
                    original_format_df[col] = export_df[col]
        else:
            # If we can't convert back exactly, just use what we have
            original_format_df = export_df
        
        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            original_format_df.to_csv, 
            f"{filename}.csv",
            index=False
        ), f"Successfully saved {len(original_format_df)} filtered points to {filename}.csv"
        
    except Exception as e:
        print(f"Error saving filtered data: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error saving filtered data: {str(e)}"

# Callback to apply parameter filter
@app.callback(
    Output('filtered-data-graph', 'figure', allow_duplicate=True),
    Output('parameter-filter-status', 'children'),
    Output('filtered-data-store', 'data', allow_duplicate=True),
    Output('filtered-data-info', 'children', allow_duplicate=True),
    Input('apply-parameter-filter', 'n_clicks'),
    State('physics-parameter-dropdown', 'value'),
    State('parameter-range-slider', 'value'),
    State('file-selector-graph15', 'value'),
    State('stored-files', 'data'),
    State('x-axis-feature-graph15', 'value'),
    State('y-axis-feature-graph15', 'value'),
    prevent_initial_call=True
)
def apply_parameter_filter(n_clicks, parameter, parameter_range, 
                          selected_ids, stored_files,
                          x_feature, y_feature):
    ctx = dash.callback_context
    if not ctx.triggered or 'apply-parameter-filter' not in ctx.triggered[0]['prop_id']:
        raise dash.exceptions.PreventUpdate
    
    print(f"Parameter filter called with {parameter}=[{parameter_range[0]}, {parameter_range[1]}]")
    
    try:
        # Process selected files to get the combined dataframe
        sampled_dfs = []
        
        for f in stored_files:
            if f['id'] in selected_ids:
                try:
                    df = pd.read_json(f['data'], orient='split')
                    is_selection = f.get('is_selection', False)
                    
                    df['file_label'] = f['filename']  # Add file name as a label
                    
                    if not is_selection:
                        # Calculate physics features
                        df = calculate_physics_features(df)
                    
                    sampled_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {f['filename']}: {str(e)}")
        
        # Combine datasets
        if not sampled_dfs:
            return {}, "No valid files selected.", {}, "No data available."
        
        combined_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
        
        # Verify features exist
        if x_feature not in combined_df.columns or y_feature not in combined_df.columns:
            return {}, f"Features {x_feature} or {y_feature} not found in data.", {}, "Features not found."
        
        if parameter not in combined_df.columns:
            return {}, f"Parameter {parameter} not found in data.", {}, "Parameter not found."
        
        # Filter by parameter range
        parameter_mask = (combined_df[parameter] >= parameter_range[0]) & (combined_df[parameter] <= parameter_range[1])
        filtered_df = combined_df.loc[parameter_mask].copy()
        
        if len(filtered_df) == 0:
            return {}, "No points remain after filtering.", {}, "Try adjusting the parameter range."
        
        # Create visualization of filtered data - simpler version
        fig = go.Figure()
        
        # Add filtered data traces - group by file label
        color_map = px.colors.qualitative.Plotly  # Use standard Plotly colors
        
        for i, label in enumerate(filtered_df['file_label'].unique()):
            mask = filtered_df['file_label'] == label
            df_subset = filtered_df[mask]
            
            # Use standard color palette
            color = color_map[i % len(color_map)]
            
            fig.add_trace(go.Scatter(
                x=df_subset[x_feature],
                y=df_subset[y_feature],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                name=f"{label} ({len(df_subset)} pts)"
            ))
        
        # Add a separate trace for all original data points (no histogram inset)
        fig.add_trace(go.Scatter(
            x=combined_df[x_feature],
            y=combined_df[y_feature],
            mode='markers',
            marker=dict(
                size=4,
                color='gray',
                opacity=0.1
            ),
            name='All data points',
            showlegend=True
        ))
        
        # Update figure layout - simpler version
        fig.update_layout(
            height=600,
            title=f"Parameter-Filtered Data: {parameter} in range [{parameter_range[0]:.2f}, {parameter_range[1]:.2f}]",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            legend_title="Data Source"
        )
        
        # Store filtered data
        filtered_data_store = {
            'filtered_df': filtered_df.to_json(date_format='iso', orient='split'),
            'x_feature': x_feature,
            'y_feature': y_feature,
            'filtering_method': 'parameter',
            'params': {
                'parameter': parameter,
                'range': parameter_range
            }
        }
        
        # Create info text
        file_counts = filtered_df['file_label'].value_counts().to_dict()
        info_text = [
            html.Div(f"Total points after filtering: {len(filtered_df)} ({len(filtered_df)/len(combined_df):.1%} of original)"),
            html.Div(f"Filter: {parameter} in range [{parameter_range[0]:.2f}, {parameter_range[1]:.2f}]"),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div([html.Div(f"{file}: {count} events") for file, count in file_counts.items()], 
                     style={"marginLeft": "10px"})
        ]
        
        return fig, "Parameter filtering applied successfully!", filtered_data_store, info_text
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Parameter filter error: {str(e)}\n{trace}")
        return {}, f"Error applying parameter filter: {str(e)}", {}, [html.Div(f"Error: {str(e)}")]


# Parameter-filter-controls display callback
@app.callback(
    Output('parameter-filter-controls', 'style', allow_duplicate=True),
    Input('physics-parameter-dropdown', 'value'),
    prevent_initial_call=True
)
def update_parameter_filter_controls_visibility(selected_parameter):
    """Show/hide parameter filter controls based on selection."""
    if not selected_parameter:
        return {'display': 'none'}
    return {'display': 'block'}


@app.callback(
    Output('parameter-range-slider', 'min', allow_duplicate=True),
    Output('parameter-range-slider', 'max', allow_duplicate=True),
    Output('parameter-range-slider', 'marks', allow_duplicate=True),
    Output('parameter-range-slider', 'value', allow_duplicate=True),
    Input('physics-parameter-dropdown', 'value'),
    State('file-selector-graph15', 'value'),
    State('stored-files', 'data'),
    prevent_initial_call=True
)
def update_parameter_filter_range(selected_parameter, selected_ids, stored_files):
    """Update the parameter filter range based on the selected parameter."""
    if not selected_parameter:
        return 0, 100, {0: '0', 100: '100'}, [0, 100]
    
    # Load data to determine parameter range
    try:
        sampled_dfs = []
        for f in stored_files:
            if f['id'] in selected_ids:
                df = pd.read_json(f['data'], orient='split')
                if selected_parameter not in df.columns:
                    df = calculate_physics_features(df)
                if selected_parameter in df.columns:
                    sampled_dfs.append(df)
        
        if not sampled_dfs:
            return 0, 100, {0: '0', 100: '100'}, [0, 100]
        
        combined_df = pd.concat(sampled_dfs, ignore_index=True)
        if selected_parameter not in combined_df.columns:
            return 0, 100, {0: '0', 100: '100'}, [0, 100]
        
        param_min = float(combined_df[selected_parameter].min())
        param_max = float(combined_df[selected_parameter].max())
        
        # Round the values for better display
        param_min = np.floor(param_min)
        param_max = np.ceil(param_max)
        
        # Create marks dictionary with fewer marks
        steps = [0, 0.25, 0.5, 0.75, 1.0]
        marks = {}
        for step in steps:
            value = param_min + step * (param_max - param_min)
            marks[value] = f'{value:.1f}'
        
        return param_min, param_max, marks, [param_min, param_max]
    
    except Exception as e:
        print(f"Error updating parameter range: {e}")
        return 0, 100, {0: '0', 100: '100'}, [0, 100]


@app.callback(
    Output('filtered-data-graph', 'figure', allow_duplicate=True),
    Input('x-axis-feature-graph15', 'value'),
    Input('y-axis-feature-graph15', 'value'),
    prevent_initial_call=True
)
def init_filter_graph(x_feature, y_feature):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    # Check if we have features selected
    if not x_feature or not y_feature:
        # Show placeholder
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[1, 2, 3, 4, 5],
            mode='markers',
            marker=dict(size=10, color='blue')
        ))
        fig.update_layout(
            height=600,
            title="Filter Graph - Select features in Graph 1.5 first",
            xaxis_title="X",
            yaxis_title="Y"
        )
        return fig
    
    # If we have features, show a message to apply a filter
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        text="Apply a filter to see results",
        showarrow=False,
        font=dict(size=20)
    )
    fig.update_layout(
        height=600,
        title=f"Ready to filter: {x_feature} vs {y_feature}",
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    return fig

# Initialize UMAP filtered graph
@app.callback(
    Output('umap-filtered-data-graph', 'figure', allow_duplicate=True),
    Input('umap-graph', 'figure'),
    prevent_initial_call=True
)
def init_umap_filter_graph(umap_figure):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # If UMAP graph exists, show a message to apply a filter
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        text="Apply a filter to see results",
        showarrow=False,
        font=dict(size=20)
    )
    fig.update_layout(
        height=600,
        title=f"Ready to filter UMAP visualization",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    return fig

# Add a proper initialization callback for the filtered-data-graph
@app.callback(
    Output('filtered-data-graph', 'figure'),
    Input('x-axis-feature-graph15', 'value'),
    Input('y-axis-feature-graph15', 'value'),
    prevent_initial_call=True
)
def init_filter_graph(x_feature, y_feature):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
        
    # Check if we have features selected
    if not x_feature or not y_feature:
        # Show placeholder
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Select features in Graph 1.5 first",
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            height=600,
            title="Filter Graph - Select features in Graph 1.5 first",
            xaxis_title="X",
            yaxis_title="Y",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        return fig
    
    # If we have features, show a message to apply a filter
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        text="Apply a filter to see results",
        showarrow=False,
        font=dict(size=20)
    )
    fig.update_layout(
        height=600,
        title=f"Ready to filter: {x_feature} vs {y_feature}",
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    return fig

# Callback to show/hide UMAP parameter filter controls
@app.callback(
    Output('umap-parameter-filter-controls', 'style'),
    Output('umap-parameter-range-slider', 'min'),
    Output('umap-parameter-range-slider', 'max'),
    Output('umap-parameter-range-slider', 'marks'),
    Output('umap-parameter-range-slider', 'value'),
    Input('umap-physics-parameter-dropdown', 'value'),
    State('umap-file-selector', 'value'),
    State('stored-files', 'data'),
    State('combined-data-store', 'data'),
    prevent_initial_call=True
)
def update_umap_parameter_filter_controls(selected_parameter, selected_ids, stored_files, combined_data_json):
    """Update the UMAP parameter filter controls based on the selected parameter."""
    if not selected_parameter:
        return {'display': 'none'}, 0, 100, {0: '0', 100: '100'}, [0, 100]
    
    # Get dataset parameter range
    try:
        # We need to load the combined data from the UMAP visualization
        if not combined_data_json or "combined_df" not in combined_data_json or combined_data_json["combined_df"] == "{}":
            return {'display': 'block'}, 0, 100, {0: '0', 100: '100'}, [0, 100]
        
        combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
        
        if selected_parameter not in combined_df.columns:
            # Some physics parameters might need to be calculated
            try:
                combined_df = calculate_physics_features(combined_df)
            except:
                pass
        
        if selected_parameter not in combined_df.columns:
            return {'display': 'block'}, 0, 100, {0: '0', 100: '100'}, [0, 100]
        
        param_min = float(combined_df[selected_parameter].min())
        param_max = float(combined_df[selected_parameter].max())
        
        # Round the values for better display
        param_min = np.floor(param_min)
        param_max = np.ceil(param_max)
        
        # Create marks dictionary - USING FEWER MARKS
        # Just show min, 25%, 50%, 75%, and max values
        steps = [0, 0.25, 0.5, 0.75, 1.0]
        marks = {}
        for step in steps:
            value = param_min + step * (param_max - param_min)
            marks[value] = f'{value:.1f}'
        
        return {'display': 'block'}, param_min, param_max, marks, [param_min, param_max]
    
    except Exception as e:
        print(f"Error updating UMAP parameter filter controls: {e}")
        return {'display': 'block'}, 0, 100, {0: '0', 100: '100'}, [0, 100]

# Callback to apply UMAP density filter
@app.callback(
    Output('umap-filtered-data-graph', 'figure', allow_duplicate=True),
    Output('umap-density-filter-status', 'children'),
    Output('umap-density-filter-info', 'children'),
    Output('umap-filtered-data-store', 'data', allow_duplicate=True),
    Input('apply-umap-density-filter', 'n_clicks'),
    State('umap-density-bandwidth-slider', 'value'),
    State('umap-density-threshold-slider', 'value'),
    State('combined-data-store', 'data'),
    State('umap-graph', 'figure'),
    prevent_initial_call=True
)
def apply_umap_density_filter(n_clicks, bandwidth, threshold_percentile, combined_data_json, umap_figure):
    ctx = dash.callback_context
    if not ctx.triggered or 'apply-umap-density-filter' not in ctx.triggered[0]['prop_id']:
        raise dash.exceptions.PreventUpdate
    
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    print(f"UMAP density filter called with bandwidth={bandwidth}, threshold={threshold_percentile}")
    
    try:
        # Get the UMAP coordinates
        if not combined_data_json or "umap_coords" not in combined_data_json:
            return {}, "No UMAP data available. Run UMAP first.", "No data available.", {}
        
        # Load UMAP coordinates and combined dataframe
        umap_df = pd.read_json(combined_data_json["umap_coords"], orient='split')
        combined_df = None
        if combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
        
        if umap_df.empty:
            return {}, "No UMAP data available.", "No data available.", {}
        
        # Extract UMAP coordinates for density calculation
        umap_coords = umap_df[['UMAP1', 'UMAP2']].values
        
        # Fast grid-based density estimation
        # Create a 2D histogram (like a heatmap)
        num_bins = 100  # Adjust based on data size and desired granularity
        x_min, x_max = np.min(umap_coords[:,0]), np.max(umap_coords[:,0])
        y_min, y_max = np.min(umap_coords[:,1]), np.max(umap_coords[:,1])
        
        # Add small padding to min/max
        padding = 0.01
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(
            umap_coords[:,0], umap_coords[:,1], 
            bins=num_bins, 
            range=[[x_min, x_max], [y_min, y_max]]
        )
        
        # Apply smoothing - simple box blur
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H, sigma=bandwidth*10)  # Adjust multiplier as needed
        
        # Assign density values to each point
        x_indices = np.clip(
            np.floor((umap_coords[:,0] - x_min) / (x_max - x_min) * num_bins).astype(int),
            0, num_bins-1
        )
        y_indices = np.clip(
            np.floor((umap_coords[:,1] - y_min) / (y_max - y_min) * num_bins).astype(int),
            0, num_bins-1
        )
        
        densities = H_smooth[x_indices, y_indices]
        
        # Calculate density threshold
        threshold = np.percentile(densities, threshold_percentile)
        
        # Filter by density
        high_density_mask = densities >= threshold
        filtered_umap_df = umap_df.iloc[high_density_mask].copy()
        filtered_umap_df['density'] = densities[high_density_mask]  # Store density for reference
        
        # If we have the original data, also filter that
        filtered_data_df = None
        if combined_df is not None:
            if len(combined_df) == len(umap_df):
                filtered_data_df = combined_df.iloc[high_density_mask].copy()
        
        print(f"Kept {len(filtered_umap_df)} high-density UMAP points ({len(filtered_umap_df)/len(umap_df):.1%} of original)")
        
        # Check if we have any points left
        if len(filtered_umap_df) == 0:
            return {}, "No points remain after filtering.", "Try lowering the threshold.", {}
        
        # Create visualization of filtered UMAP data
        fig = go.Figure()
        
        # Extract color information from original UMAP figure
        color_map = {}
        if umap_figure and 'data' in umap_figure:
            for trace in umap_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    # Clean the label if it contains point count
                    clean_name = trace['name']
                    if ' (' in clean_name:
                        clean_name = clean_name.split(' (')[0]
                    color_map[clean_name] = trace['marker']['color']
        
        # Add original data as background with reduced opacity
        fig.add_trace(go.Scatter(
            x=umap_df['UMAP1'],
            y=umap_df['UMAP2'],
            mode='markers',
            marker=dict(
                size=4,
                color='gray',
                opacity=0.1
            ),
            name='Original data (low density)',
            showlegend=True
        ))
        
        # Add traces for each file label
        for label in filtered_umap_df['file_label'].unique():
            mask = filtered_umap_df['file_label'] == label
            df_subset = filtered_umap_df[mask]
            
            # Use color from original figure if available
            color = color_map.get(label, None)
            
            fig.add_trace(go.Scatter(
                x=df_subset["UMAP1"],
                y=df_subset["UMAP2"],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                name=f"{label} ({len(df_subset)} pts)"
            ))
        
        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"Density-Filtered UMAP (Kept {len(filtered_umap_df)} points, {len(filtered_umap_df)/len(umap_df):.1%} of original)",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data Source"
        )
        
        # Store filtered data
        filtered_data_store = {
            'filtered_umap_df': filtered_umap_df.to_json(date_format='iso', orient='split'),
            'filtered_data_df': filtered_data_df.to_json(date_format='iso', orient='split') if filtered_data_df is not None else "{}",
            'filtering_method': 'density',
            'params': {
                'bandwidth': bandwidth,
                'threshold_percentile': threshold_percentile,
                'threshold_value': float(threshold)
            }
        }
        
        # Create info text
        file_counts = filtered_umap_df['file_label'].value_counts().to_dict()
        info_text = [
            html.Div(f"Total points after filtering: {len(filtered_umap_df)} ({len(filtered_umap_df)/len(umap_df):.1%} of original)"),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div([html.Div(f"{file}: {count} events") for file, count in file_counts.items()], 
                     style={"marginLeft": "10px"})
        ]
        
        return fig, "UMAP density filtering applied successfully!", info_text, filtered_data_store
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"UMAP density filter error: {str(e)}\n{trace}")
        return {}, f"Error applying UMAP density filter: {str(e)}", f"Error: {str(e)}", {}

# Callback to apply UMAP parameter filter
@app.callback(
    Output('umap-filtered-data-graph', 'figure', allow_duplicate=True),
    Output('umap-parameter-filter-status', 'children'),
    Output('umap-filtered-data-store', 'data', allow_duplicate=True),
    Output('umap-filtered-data-info', 'children', allow_duplicate=True),
    Input('apply-umap-parameter-filter', 'n_clicks'),
    State('umap-physics-parameter-dropdown', 'value'),
    State('umap-parameter-range-slider', 'value'),
    State('combined-data-store', 'data'),
    State('umap-graph', 'figure'),
    prevent_initial_call=True
)
def apply_umap_parameter_filter(n_clicks, parameter, parameter_range, combined_data_json, umap_figure):
    ctx = dash.callback_context
    if not ctx.triggered or 'apply-umap-parameter-filter' not in ctx.triggered[0]['prop_id']:
        raise dash.exceptions.PreventUpdate
    
    print(f"UMAP parameter filter called with {parameter}=[{parameter_range[0]}, {parameter_range[1]}]")
    
    try:
        # Get the UMAP coordinates and combined dataframe
        if not combined_data_json or "umap_coords" not in combined_data_json or "combined_df" not in combined_data_json:
            return {}, "No UMAP data available. Run UMAP first.", {}, "No data available."
        
        # Load UMAP coordinates and combined dataframe
        umap_df = pd.read_json(combined_data_json["umap_coords"], orient='split')
        combined_df = None
        if combined_data_json["combined_df"] != "{}":
            combined_df = pd.read_json(combined_data_json["combined_df"], orient='split')
        else:
            return {}, "No original data available for filtering.", {}, "No original data available."
        
        if umap_df.empty or combined_df is None or combined_df.empty:
            return {}, "No data available for UMAP parameter filtering.", {}, "No data available."
        
        # Check if parameter exists in the data
        if parameter not in combined_df.columns:
            # Try to calculate physics features if needed
            try:
                combined_df = calculate_physics_features(combined_df)
            except:
                pass
            
            if parameter not in combined_df.columns:
                return {}, f"Parameter {parameter} not found in data.", {}, "Parameter not found."
        
        # Filter by parameter range
        parameter_mask = (combined_df[parameter] >= parameter_range[0]) & (combined_df[parameter] <= parameter_range[1])
        
        # Make sure we can apply the mask
        if len(parameter_mask) != len(umap_df):
            return {}, "UMAP and data dimensions don't match.", {}, "Data mismatch error."
        
        # Apply filter
        filtered_data_df = combined_df.loc[parameter_mask].copy()
        filtered_umap_df = umap_df.loc[parameter_mask].copy()
        
        if len(filtered_umap_df) == 0:
            return {}, "No points remain after filtering.", {}, "Try adjusting the parameter range."
        
        # Create visualization of filtered UMAP data
        fig = go.Figure()
        
        # Extract color information from original UMAP figure
        color_map = {}
        if umap_figure and 'data' in umap_figure:
            for trace in umap_figure['data']:
                if 'name' in trace and 'marker' in trace and 'color' in trace['marker']:
                    # Clean the label if it contains point count
                    clean_name = trace['name']
                    if ' (' in clean_name:
                        clean_name = clean_name.split(' (')[0]
                    color_map[clean_name] = trace['marker']['color']
        
        # Add original data as background with reduced opacity
        fig.add_trace(go.Scatter(
            x=umap_df['UMAP1'],
            y=umap_df['UMAP2'],
            mode='markers',
            marker=dict(
                size=4,
                color='gray',
                opacity=0.1
            ),
            name='All UMAP points',
            showlegend=True
        ))
        
        # Add traces for each file label in filtered data
        for label in filtered_umap_df['file_label'].unique():
            mask = filtered_umap_df['file_label'] == label
            df_subset = filtered_umap_df[mask]
            
            # Use color from original figure if available
            color = color_map.get(label, None)
            
            fig.add_trace(go.Scatter(
                x=df_subset["UMAP1"],
                y=df_subset["UMAP2"],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                name=f"{label} ({len(df_subset)} pts)"
            ))
        
        # Update figure layout
        fig.update_layout(
            height=600,
            title=f"Parameter-Filtered UMAP: {parameter} in range [{parameter_range[0]:.2f}, {parameter_range[1]:.2f}]",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data Source"
        )
        
        # Store filtered data
        filtered_data_store = {
            'filtered_umap_df': filtered_umap_df.to_json(date_format='iso', orient='split'),
            'filtered_data_df': filtered_data_df.to_json(date_format='iso', orient='split') if filtered_data_df is not None else "{}",
            'filtering_method': 'parameter',
            'params': {
                'parameter': parameter,
                'range': parameter_range
            }
        }
        
        # Create info text
        file_counts = filtered_umap_df['file_label'].value_counts().to_dict()
        info_text = [
            html.Div(f"Total points after filtering: {len(filtered_umap_df)} ({len(filtered_umap_df)/len(umap_df):.1%} of original)"),
            html.Div(f"Filter: {parameter} in range [{parameter_range[0]:.2f}, {parameter_range[1]:.2f}]"),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div([html.Div(f"{file}: {count} events") for file, count in file_counts.items()], 
                     style={"marginLeft": "10px"})
        ]
        
        return fig, "UMAP parameter filtering applied successfully!", filtered_data_store, info_text
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"UMAP parameter filter error: {str(e)}\n{trace}")
        return {}, f"Error applying UMAP parameter filter: {str(e)}", {}, [html.Div(f"Error: {str(e)}")]

# Callback to download filtered UMAP data
@app.callback(
    Output('download-umap-filtered-data', 'data'),
    Output('save-umap-filtered-data-status', 'children'),
    Input('save-umap-filtered-data-btn', 'n_clicks'),
    State('umap-filtered-data-filename', 'value'),
    State('umap-filtered-data-store', 'data'),
    prevent_initial_call=True
)
def download_umap_filtered_data(n_clicks, filename, filtered_data_store):
    """Generate CSV file of filtered UMAP data points for download."""
    if not n_clicks or not filename or not filtered_data_store:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Load the filtered data
        if 'filtered_data_df' in filtered_data_store and filtered_data_store['filtered_data_df'] != "{}":
            # Use the original data if available for more complete information
            filtered_df = pd.read_json(filtered_data_store['filtered_data_df'], orient='split')
        elif 'filtered_umap_df' in filtered_data_store:
            # Fallback to UMAP coordinates
            filtered_df = pd.read_json(filtered_data_store['filtered_umap_df'], orient='split')
        else:
            raise dash.exceptions.PreventUpdate
        
        if filtered_df.empty:
            raise dash.exceptions.PreventUpdate
        
        # Remove columns not needed for export
        export_df = filtered_df.copy()
        
        # Remove density column which was added during filtering
        if 'density' in export_df.columns:
            export_df = export_df.drop(columns=['density'])
        
        # If it's UMAP data only, keep the important columns
        if 'filtered_data_df' not in filtered_data_store or filtered_data_store['filtered_data_df'] == "{}":
            important_cols = ['UMAP1', 'UMAP2', 'file_label']
            export_df = export_df[important_cols]
        else:
            # For full data, prepare for proper export format
            # Remove file_label which was added during processing
            if 'file_label' in export_df.columns:
                export_df = export_df.drop(columns=['file_label'])
            
            # Get only the particle momentum columns (original data format)
            momentum_columns = [col for col in export_df.columns if col.startswith('particle_')]
            
            # If we have the momentum columns, convert back to original format
            if momentum_columns and len(momentum_columns) == 15:  # Should be 5 particles x 3 dimensions
                # Create the reverse mapping from standardized to original column names
                reverse_columns = [
                    'Px_ion1', 'Py_ion1', 'Pz_ion1',
                    'Px_ion2', 'Py_ion2', 'Pz_ion2',
                    'Px_neutral', 'Py_neutral', 'Pz_neutral',
                    'Px_electron1', 'Py_electron1', 'Pz_electron1',
                    'Px_electron2', 'Py_electron2', 'Pz_electron2'
                ]
                
                # Extract and rename the momentum columns
                original_format_df = pd.DataFrame()
                for i, col in enumerate(momentum_columns):
                    if i < len(reverse_columns):
                        original_format_df[reverse_columns[i]] = export_df[col]
                        
                # Also include other physics features if available
                for col in export_df.columns:
                    if not col.startswith('particle_') and col not in ['UMAP1', 'UMAP2']:
                        original_format_df[col] = export_df[col]
                
                # Use this for export
                export_df = original_format_df
        
        # Return the dataframe as a CSV for download
        return dcc.send_data_frame(
            export_df.to_csv, 
            f"{filename}.csv",
            index=False
        ), f"Successfully saved {len(export_df)} filtered UMAP points to {filename}.csv"
        
    except Exception as e:
        print(f"Error saving filtered UMAP data: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error saving filtered UMAP data: {str(e)}"
        

# Initialize with example profiles
@app.callback(
    Output('configuration-profiles-store', 'data'),
    Input('url', 'pathname'),  # Triggers on page load
    prevent_initial_call=False
)
def initialize_profiles(pathname):
    """Initialize with example profiles."""
    return {
        'D2O': {
            'name': 'D2O',
            'particle_count': {'ions': 2, 'neutrals': 1, 'electrons': 2},
            'particles': {
                'ion_0': {'name': 'D+', 'mass': 2, 'charge': 1, 'type': 'ion', 'index': 0},
                'ion_1': {'name': 'D+', 'mass': 2, 'charge': 1, 'type': 'ion', 'index': 1},
                'neutral_0': {'name': 'O', 'mass': 16, 'charge': 0, 'type': 'neutral', 'index': 0},
                'electron_0': {'name': 'e-', 'mass': 0.000545, 'charge': -1, 'type': 'electron', 'index': 0},
                'electron_1': {'name': 'e-', 'mass': 0.000545, 'charge': -1, 'type': 'electron', 'index': 1}
            }
        },
        'HDO': {
            'name': 'HDO',
            'particle_count': {'ions': 2, 'neutrals': 1, 'electrons': 2},
            'particles': {
                'ion_0': {'name': 'H+', 'mass': 1, 'charge': 1, 'type': 'ion', 'index': 0},
                'ion_1': {'name': 'D+', 'mass': 2, 'charge': 1, 'type': 'ion', 'index': 1},
                'neutral_0': {'name': 'O', 'mass': 16, 'charge': 0, 'type': 'neutral', 'index': 0},
                'electron_0': {'name': 'e-', 'mass': 0.000545, 'charge': -1, 'type': 'electron', 'index': 0},
                'electron_1': {'name': 'e-', 'mass': 0.000545, 'charge': -1, 'type': 'electron', 'index': 1}
            }
        }
    }

# Update profile dropdown options
@app.callback(
    Output('active-profile-dropdown', 'options'),
    Input('configuration-profiles-store', 'data')
)
def update_profile_dropdown(profiles_store):
    """Update the profile dropdown options."""
    if not profiles_store:
        return []
    return [{'label': name, 'value': name} for name in profiles_store.keys()]

# Generate particle configuration UI
@app.callback(
    Output('particle-config-container', 'children'),
    Input('num-ions', 'value'),
    Input('num-neutrals', 'value'),
    Input('num-electrons', 'value')
)
def update_particle_config_ui(num_ions, num_neutrals, num_electrons):
    """Generate UI for particle configuration."""
    config_elements = []
    
    # Create configuration for ions
    if num_ions and num_ions > 0:
        ion_inputs = []
        for i in range(num_ions):
            ion_inputs.append(
                html.Div([
                    html.Label(f"Ion {i+1}:", style={'width': '80px', 'display': 'inline-block'}),
                    dcc.Input(
                        id={'type': 'ion-name', 'index': i},
                        type='text',
                        placeholder='e.g., H+, D+, O+',
                        value='D+' if i < 2 else 'Ion',  # Default to D+ for first two
                        style={'width': '100px', 'marginRight': '10px'}
                    ),
                    html.Label("Mass (amu):", style={'marginLeft': '10px'}),
                    dcc.Input(
                        id={'type': 'ion-mass', 'index': i},
                        type='number',
                        value=2 if i < 2 else 1,  # Default to 2 for deuterium
                        min=1,
                        style={'width': '80px', 'marginLeft': '5px'}
                    ),
                    html.Label("Charge:", style={'marginLeft': '10px'}),
                    dcc.Input(
                        id={'type': 'ion-charge', 'index': i},
                        type='number',
                        value=1,
                        style={'width': '60px', 'marginLeft': '5px'}
                    ),
                ], style={'marginBottom': '8px'})
            )
        
        config_elements.append(html.Div([
            html.H5("Ion Configuration", style={'color': '#1976d2', 'marginBottom': '10px'}),
            html.Div(ion_inputs, style={'padding': '10px', 'backgroundColor': '#e3f2fd', 'borderRadius': '5px'})
        ]))
    
    # Create configuration for neutrals
    if num_neutrals and num_neutrals > 0:
        neutral_inputs = []
        for i in range(num_neutrals):
            neutral_inputs.append(
                html.Div([
                    html.Label(f"Neutral {i+1}:", style={'width': '80px', 'display': 'inline-block'}),
                    dcc.Input(
                        id={'type': 'neutral-name', 'index': i},
                        type='text',
                        placeholder='e.g., O, N, C',
                        value='O' if i == 0 else 'Neutral',  # Default to O for first
                        style={'width': '100px', 'marginRight': '10px'}
                    ),
                    html.Label("Mass (amu):", style={'marginLeft': '10px'}),
                    dcc.Input(
                        id={'type': 'neutral-mass', 'index': i},
                        type='number',
                        value=16 if i == 0 else 1,  # Default to 16 for oxygen
                        min=1,
                        style={'width': '80px', 'marginLeft': '5px'}
                    ),
                ], style={'marginBottom': '8px'})
            )
        
        config_elements.append(html.Div([
            html.H5("Neutral Configuration", style={'color': '#388e3c', 'marginBottom': '10px', 'marginTop': '15px'}),
            html.Div(neutral_inputs, style={'padding': '10px', 'backgroundColor': '#e8f5e9', 'borderRadius': '5px'})
        ]))
    
    # Note about electrons
    if num_electrons and num_electrons > 0:
        config_elements.append(html.Div([
            html.H5("Electron Configuration", style={'color': '#f57c00', 'marginBottom': '10px', 'marginTop': '15px'}),
            html.Div([
                html.I(f"System includes {num_electrons} electron(s). "),
                html.I("Electron mass is fixed at 1/1836 amu (0.000545 amu)"),
            ], style={'fontSize': '12px', 'color': 'gray', 'padding': '10px', 
                     'backgroundColor': '#fff3e0', 'borderRadius': '5px'})
        ]))
    
    return config_elements

# Create new profile
@app.callback(
    Output('configuration-profiles-store', 'data', allow_duplicate=True),
    Output('config-profile-name', 'value'),
    Output('profile-edit-status', 'children'),
    Input('create-profile-btn', 'n_clicks'),
    State('config-profile-name', 'value'),
    State('configuration-profiles-store', 'data'),
    State('num-ions', 'value'),
    State('num-neutrals', 'value'),
    State('num-electrons', 'value'),
    State({'type': 'ion-name', 'index': ALL}, 'value'),
    State({'type': 'ion-mass', 'index': ALL}, 'value'),
    State({'type': 'ion-charge', 'index': ALL}, 'value'),
    State({'type': 'neutral-name', 'index': ALL}, 'value'),
    State({'type': 'neutral-mass', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def create_profile(n_clicks, profile_name, profiles_store, 
                  num_ions, num_neutrals, num_electrons,
                  ion_names, ion_masses, ion_charges,
                  neutral_names, neutral_masses):
    """Create a new configuration profile with actual particle data."""
    if not n_clicks or not profile_name:
        raise dash.exceptions.PreventUpdate
    
    # Initialize profiles store if needed
    if profiles_store is None:
        profiles_store = {}
    
    # Check if profile already exists
    if profile_name in profiles_store:
        return profiles_store, profile_name, "Profile with this name already exists!"
    
    # Build particles dictionary with actual input values
    particles = {}
    
    # Process ions with actual input values
    for i in range(num_ions or 0):
        if i < len(ion_names):  # Make sure we have input values
            particles[f'ion_{i}'] = {
                'name': ion_names[i] if ion_names[i] else f'Ion{i+1}',
                'mass': ion_masses[i] if i < len(ion_masses) and ion_masses[i] else 1,
                'charge': ion_charges[i] if i < len(ion_charges) and ion_charges[i] else 1,
                'type': 'ion',
                'index': i
            }
    
    # Process neutrals with actual input values
    for i in range(num_neutrals or 0):
        if i < len(neutral_names):  # Make sure we have input values
            particles[f'neutral_{i}'] = {
                'name': neutral_names[i] if neutral_names[i] else f'Neutral{i+1}',
                'mass': neutral_masses[i] if i < len(neutral_masses) and neutral_masses[i] else 16,
                'charge': 0,
                'type': 'neutral',
                'index': i
            }
    
    # Add electrons (always with fixed properties)
    for i in range(num_electrons or 0):
        particles[f'electron_{i}'] = {
            'name': 'e-',
            'mass': 0.000545,  # Electron mass in amu
            'charge': -1,
            'type': 'electron',
            'index': i
        }
    
    # Create new profile with actual configuration
    new_profile = {
        'name': profile_name,
        'particle_count': {
            'ions': num_ions or 0,
            'neutrals': num_neutrals or 0,
            'electrons': num_electrons or 0
        },
        'particles': particles
    }
    
    # Add to profiles store
    profiles_store[profile_name] = new_profile
    
    # Success message
    success_msg = f"Successfully created profile '{profile_name}' with {num_ions} ion(s), {num_neutrals} neutral(s), and {num_electrons} electron(s)"
    
    return profiles_store, "", success_msg  # Clear the profile name input


@app.callback(
    Output('num-ions', 'value'),
    Output('num-neutrals', 'value'),
    Output('num-electrons', 'value'),
    Input('active-profile-dropdown', 'value'),
    State('configuration-profiles-store', 'data'),
    prevent_initial_call=True
)
def load_profile_for_editing(selected_profile, profiles_store):
    """Load a profile's particle counts when selected for editing."""
    if not selected_profile or not profiles_store or selected_profile not in profiles_store:
        raise dash.exceptions.PreventUpdate
    
    profile = profiles_store[selected_profile]
    particle_count = profile.get('particle_count', {})
    
    return (
        particle_count.get('ions', 2),
        particle_count.get('neutrals', 1),
        particle_count.get('electrons', 2)
    )


# Update file assignment UI
@app.callback(
    Output('file-configuration-assignment', 'children'),
    Input('stored-files', 'data'),
    Input('configuration-profiles-store', 'data'),
    State('file-config-assignments-store', 'data')
)
def update_file_assignment_ui(stored_files, profiles_store, assignments_store):
    """Create dropdowns for file-to-profile assignment."""
    if not stored_files:
        return html.Div("Upload files first to assign configurations", 
                       style={"color": "gray", "fontStyle": "italic"})
    
    if not profiles_store:
        return html.Div("Create configuration profiles first", 
                       style={"color": "orange", "fontStyle": "italic"})
    
    # Create profile options
    profile_options = [{'label': name, 'value': name} for name in profiles_store.keys()]
    profile_options.insert(0, {'label': 'None (Skip file)', 'value': 'none'})
    
    assignment_elements = []
    
    for f in stored_files:
        if not f.get('is_selection', False):  # Only for data files
            current_value = assignments_store.get(f['filename'], 'none') if assignments_store else 'none'
            
            assignment_elements.append(
                html.Div([
                    html.Span(f"{f['filename']}: ", 
                             style={'fontWeight': 'bold', 'width': '300px', 
                                   'display': 'inline-block'}),
                    dcc.Dropdown(
                        id={'type': 'file-profile-assignment', 'filename': f['filename']},
                        options=profile_options,
                        value=current_value,
                        style={'width': '250px', 'display': 'inline-block'}
                    )
                ], style={'marginBottom': '10px'})
            )
    
    return assignment_elements

# Save file assignments
@app.callback(
    Output('file-config-assignments-store', 'data'),
    Output('file-assignment-status', 'children'),  # Changed from 'file-config-status'
    Input('apply-file-config-btn', 'n_clicks'),
    State({'type': 'file-profile-assignment', 'filename': ALL}, 'value'),
    State({'type': 'file-profile-assignment', 'filename': ALL}, 'id'),
    prevent_initial_call=True
)
def save_file_assignments(n_clicks, assignments, assignment_ids):
    """Save file-to-profile assignments."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    assignments_dict = {}
    assigned_count = 0
    
    for i, (assignment, id_info) in enumerate(zip(assignments, assignment_ids)):
        if assignment and assignment != 'none':
            filename = id_info['filename']
            assignments_dict[filename] = assignment
            assigned_count += 1
    
    return assignments_dict, f"Assigned {assigned_count} files to configuration profiles"

# Add this callback after the initialize_profiles callback
@app.callback(
    Output('active-profiles-display', 'children'),
    Input('configuration-profiles-store', 'data')
)
def update_active_profiles_display(profiles_store):
    """Update the display of active profiles."""
    if not profiles_store:
        return html.Div("No profiles created yet", style={"color": "gray", "fontStyle": "italic"})
    
    profile_cards = []
    
    for profile_name, profile_data in profiles_store.items():
        particle_count = profile_data.get('particle_count', {})
        
        # Create a summary of the profile
        particle_summary = []
        if particle_count.get('ions', 0) > 0:
            particle_summary.append(f"{particle_count['ions']} ion(s)")
        if particle_count.get('neutrals', 0) > 0:
            particle_summary.append(f"{particle_count['neutrals']} neutral(s)")
        if particle_count.get('electrons', 0) > 0:
            particle_summary.append(f"{particle_count['electrons']} electron(s)")
        
        # Get particle details
        particles = profile_data.get('particles', {})
        particle_details = []
        
        # Group particles by type
        for p_type in ['ion', 'neutral', 'electron']:
            type_particles = [p for p in particles.values() if p.get('type') == p_type]
            if type_particles:
                if p_type == 'electron':
                    particle_details.append(f"Electrons: {len(type_particles)}")
                else:
                    names = [f"{p.get('name', 'Unknown')} (m={p.get('mass', '?')})" for p in type_particles]
                    particle_details.append(f"{p_type.capitalize()}s: {', '.join(names)}")
        
        # Create profile card
        profile_card = html.Div([
            html.Div([
                html.H6(profile_name, style={'margin': '0', 'color': '#2e7d32', 'fontWeight': 'bold'}),
                html.Div(', '.join(particle_summary), style={'fontSize': '14px', 'color': '#666'}),
                html.Div(particle_details, style={'fontSize': '12px', 'color': '#888', 'marginTop': '5px'})
            ], style={'padding': '10px'}),
            html.Button(
                "Delete", 
                id={'type': 'delete-profile-btn', 'profile': profile_name},
                n_clicks=0,
                style={'position': 'absolute', 'top': '10px', 'right': '10px', 
                       'padding': '5px 10px', 'fontSize': '12px', 'backgroundColor': '#f44336',
                       'width': 'auto'}
            )
        ], style={
            'position': 'relative',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'padding': '10px',
            'marginBottom': '10px',
            'backgroundColor': 'white',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
        
        profile_cards.append(profile_card)
    
    return profile_cards

# Add callback to handle profile deletion
@app.callback(
    Output('configuration-profiles-store', 'data', allow_duplicate=True),
    Input({'type': 'delete-profile-btn', 'profile': ALL}, 'n_clicks'),
    State({'type': 'delete-profile-btn', 'profile': ALL}, 'id'),
    State('configuration-profiles-store', 'data'),
    prevent_initial_call=True
)
def delete_profile(n_clicks_list, button_ids, profiles_store):
    """Delete a profile when delete button is clicked."""
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    
    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Extract the profile name from the triggered button
    triggered_id = ctx.triggered[0]['prop_id']
    if '"profile"' in triggered_id:
        # Parse the profile name from the ID
        import json
        button_id = json.loads(triggered_id.split('.')[0])
        profile_to_delete = button_id['profile']
        
        # Delete the profile
        if profile_to_delete in profiles_store:
            del profiles_store[profile_to_delete]
    
    return profiles_store

@app.callback(
    [Output({'type': 'ion-name', 'index': ALL}, 'value'),
     Output({'type': 'ion-mass', 'index': ALL}, 'value'),
     Output({'type': 'ion-charge', 'index': ALL}, 'value'),
     Output({'type': 'neutral-name', 'index': ALL}, 'value'),
     Output({'type': 'neutral-mass', 'index': ALL}, 'value')],
    Input('active-profile-dropdown', 'value'),
    State('configuration-profiles-store', 'data'),
    State('num-ions', 'value'),
    State('num-neutrals', 'value'),
    State('num-electrons', 'value'),
    prevent_initial_call=True
)
def populate_particle_fields(selected_profile, profiles_store, num_ions, num_neutrals, num_electrons):
    """Populate particle configuration fields when a profile is selected."""
    if not selected_profile or not profiles_store or selected_profile not in profiles_store:
        raise dash.exceptions.PreventUpdate
    
    profile = profiles_store[selected_profile]
    particles = profile.get('particles', {})
    
    # Initialize empty lists for all outputs
    ion_names = []
    ion_masses = []
    ion_charges = []
    neutral_names = []
    neutral_masses = []
    
    # Populate ion data
    for i in range(num_ions or 0):
        particle_key = f'ion_{i}'
        if particle_key in particles:
            particle = particles[particle_key]
            ion_names.append(particle.get('name', f'Ion{i+1}'))
            ion_masses.append(particle.get('mass', 1))
            ion_charges.append(particle.get('charge', 1))
        else:
            ion_names.append(f'Ion{i+1}')
            ion_masses.append(1)
            ion_charges.append(1)
    
    # Populate neutral data
    for i in range(num_neutrals or 0):
        particle_key = f'neutral_{i}'
        if particle_key in particles:
            particle = particles[particle_key]
            neutral_names.append(particle.get('name', f'Neutral{i+1}'))
            neutral_masses.append(particle.get('mass', 16))
        else:
            neutral_names.append(f'Neutral{i+1}')
            neutral_masses.append(16)
    
    return ion_names, ion_masses, ion_charges, neutral_names, neutral_masses


@app.callback(
    Output('configuration-profiles-store', 'data', allow_duplicate=True),
    Output('profile-edit-status', 'children', allow_duplicate=True),
    Input('update-profile-btn', 'n_clicks'),
    State('active-profile-dropdown', 'value'),
    State('configuration-profiles-store', 'data'),
    State('num-ions', 'value'),
    State('num-neutrals', 'value'),
    State('num-electrons', 'value'),
    State({'type': 'ion-name', 'index': ALL}, 'value'),
    State({'type': 'ion-mass', 'index': ALL}, 'value'),
    State({'type': 'ion-charge', 'index': ALL}, 'value'),
    State({'type': 'neutral-name', 'index': ALL}, 'value'),
    State({'type': 'neutral-mass', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def update_profile(n_clicks, profile_name, profiles_store, 
                  num_ions, num_neutrals, num_electrons,
                  ion_names, ion_masses, ion_charges,
                  neutral_names, neutral_masses):
    """Update an existing configuration profile."""
    if not n_clicks or not profile_name or not profiles_store or profile_name not in profiles_store:
        raise dash.exceptions.PreventUpdate
    
    # Build updated particles dictionary
    particles = {}
    
    # Process ions
    for i in range(num_ions or 0):
        if i < len(ion_names):
            particles[f'ion_{i}'] = {
                'name': ion_names[i] if ion_names[i] else f'Ion{i+1}',
                'mass': ion_masses[i] if i < len(ion_masses) and ion_masses[i] else 1,
                'charge': ion_charges[i] if i < len(ion_charges) and ion_charges[i] else 1,
                'type': 'ion',
                'index': i
            }
    
    # Process neutrals
    for i in range(num_neutrals or 0):
        if i < len(neutral_names):
            particles[f'neutral_{i}'] = {
                'name': neutral_names[i] if neutral_names[i] else f'Neutral{i+1}',
                'mass': neutral_masses[i] if i < len(neutral_masses) and neutral_masses[i] else 16,
                'charge': 0,
                'type': 'neutral',
                'index': i
            }
    
    # Add electrons
    for i in range(num_electrons or 0):
        particles[f'electron_{i}'] = {
            'name': 'e-',
            'mass': 0.000545,
            'charge': -1,
            'type': 'electron',
            'index': i
        }
    
    # Update the profile
    profiles_store[profile_name] = {
        'name': profile_name,
        'particle_count': {
            'ions': num_ions or 0,
            'neutrals': num_neutrals or 0,
            'electrons': num_electrons or 0
        },
        'particles': particles
    }
    
    return profiles_store, f"Successfully updated profile '{profile_name}'"

if __name__ == '__main__':
    app.run_server(debug=True, port=9000)


# In[ ]:




