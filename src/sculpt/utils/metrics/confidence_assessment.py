import numpy as np


def calculate_adaptive_confidence_score(
    metrics, data_characteristics=None, clustering_method="dbscan"
):
    """Calculate an adaptive confidence score with improved weighting and logic."""

    # Enhanced tier definitions with better balance
    tier1_metrics = {
        "silhouette": {
            "weight": 0.35,
            "reliability": 0.9,
            "description": "Most reliable cluster quality measure",
            "min_threshold": 0.2,
            "good_threshold": 0.4,
        },
        "hopkins": {
            "weight": 0.25,
            "reliability": 0.85,
            "description": "Fundamental clusterability assessment",
            "min_threshold": 0.5,
            "good_threshold": 0.7,
        },
    }

    # Tier 2: Useful but context-dependent metrics
    tier2_metrics = {
        "stability": {
            "weight": 0.15,
            "reliability": 0.7,
            "description": "Reproducibility under perturbation",
            "min_threshold": 0.3,
            "good_threshold": 0.6,
            "conditions": ["sufficient_data"],
        },
        "physics_consistency": {
            "weight": 0.2,
            "reliability": 0.8,
            "description": "Domain-specific validation",
            "min_threshold": 0.2,
            "good_threshold": 0.5,
            "conditions": ["physics_relevant"],
        },
        "calinski_harabasz": {
            "weight": 0.1,  # Include CH with low weight
            "reliability": 0.6,
            "description": "Between-cluster separation",
            "min_threshold": 30,
            "good_threshold": 100,
        },
    }

    # Tier 3: Often misleading metrics (very low weight)
    tier3_metrics = {
        "davies_bouldin": {
            "weight": 0.05,
            "reliability": 0.4,
            "description": "Can be misleading with noise",
            "issues": ["sensitive_to_noise", "poor_with_varying_densities"],
        }
    }

    # Analyze data characteristics if provided
    if data_characteristics is None:
        data_characteristics = analyze_data_characteristics(metrics)

    # Adaptive weighting based on context
    adaptive_weights = calculate_adaptive_weights(
        tier1_metrics,
        tier2_metrics,
        tier3_metrics,
        data_characteristics,
        clustering_method,
    )

    # Calculate confidence with reliability-adjusted weights
    confidence_result = calculate_weighted_confidence(metrics, adaptive_weights)

    # Apply bonus for exceptional clustering
    confidence_result = apply_clustering_bonus(confidence_result, metrics)

    # Add uncertainty estimation
    confidence_result["uncertainty"] = calculate_confidence_uncertainty(
        confidence_result, metrics
    )

    # Add validation warnings
    confidence_result["warnings"] = validate_confidence_score(confidence_result)

    # Ensure final confidence respects theoretical maximum
    confidence_result["overall_confidence"] = min(
        0.95, confidence_result["overall_confidence"]
    )

    # RE-CATEGORIZE confidence level after all adjustments (THIS IS THE KEY FIX)
    final_score = confidence_result["overall_confidence"]

    confidence_result["confidence_level"] = categorize_confidence(final_score)

    # Add context-aware analysis
    confidence_result["analysis"] = analyze_confidence_context(
        confidence_result, metrics, data_characteristics
    )

    return confidence_result


def analyze_data_characteristics(metrics):
    """Infer data characteristics from available metrics."""
    characteristics = {
        "has_noise": False,
        "sufficient_data": True,
        "physics_relevant": False,
        "cluster_quality": "unknown",
    }

    # Detect noise presence
    if "noise_ratio" in metrics and metrics["noise_ratio"] > 0.1:
        characteristics["has_noise"] = True

    # Detect if physics features are relevant
    if "physics_consistency" in metrics:
        characteristics["physics_relevant"] = True

    # Assess cluster quality
    if "silhouette" in metrics:
        if metrics["silhouette"] > 0.5:
            characteristics["cluster_quality"] = "good"
        elif metrics["silhouette"] > 0.2:
            characteristics["cluster_quality"] = "moderate"
        else:
            characteristics["cluster_quality"] = "poor"

    return characteristics


def calculate_adaptive_weights(tier1, tier2, tier3, characteristics, clustering_method):
    """Calculate adaptive weights based on context."""
    weights = {}

    # Always include Tier 1 metrics (most reliable)
    for metric, info in tier1.items():
        weights[metric] = {
            "weight": info["weight"],
            "reliability": info["reliability"],
            "tier": 1,
            "reason": info["description"],
        }

    # Conditionally include Tier 2 metrics
    for metric, info in tier2.items():
        include = True
        exclusion_reason = None

        if "conditions" in info:
            for condition in info["conditions"]:
                if (
                    condition == "sufficient_data"
                    and not characteristics["sufficient_data"]
                ):
                    include = False
                    exclusion_reason = (
                        "Insufficient data for reliable stability assessment"
                    )
                elif (
                    condition == "physics_relevant"
                    and not characteristics["physics_relevant"]
                ):
                    include = False
                    exclusion_reason = "Physics consistency not applicable"

        if include:
            weights[metric] = {
                "weight": info["weight"],
                "reliability": info["reliability"],
                "tier": 2,
                "reason": info["description"],
            }
        else:
            weights[metric] = {
                "weight": 0,
                "reliability": 0,
                "tier": 2,
                "excluded": True,
                "exclusion_reason": exclusion_reason,
            }

    # Generally exclude Tier 3 metrics, but include with warnings
    for metric, info in tier3.items():
        weights[metric] = {
            "weight": info["weight"] * 0.1,  # Heavily downweight
            "reliability": info["reliability"],
            "tier": 3,
            "reason": f"Low reliability: {info['description']}",
            "issues": info["issues"],
        }

    # Normalize weights
    total_weight = sum(w["weight"] for w in weights.values())
    if total_weight > 0:
        for metric in weights:
            weights[metric]["normalized_weight"] = (
                weights[metric]["weight"] / total_weight
            )

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
        if metric in metrics and weight_info["weight"] > 0:
            available_metrics += 1

            # Normalize metric to 0-1 scale
            normalized_value = normalize_metric(metric, metrics[metric])
            normalized_scores[metric] = normalized_value

            # Weight by both importance and reliability
            effective_weight = weight_info["weight"] * weight_info["reliability"]
            contribution = normalized_value * effective_weight

            confidence_components[metric] = {
                "raw_value": metrics[metric],
                "normalized_value": normalized_value,
                "weight": weight_info["weight"],
                "reliability": weight_info["reliability"],
                "effective_weight": effective_weight,
                "contribution": contribution,
                "tier": weight_info["tier"],
                "reason": weight_info["reason"],
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
    if "silhouette" in normalized_scores and normalized_scores["silhouette"] > 0.6:
        overall_confidence = max(overall_confidence, 0.5)  # At least moderate

    # Ensure confidence is in valid range [0, 1]
    overall_confidence = np.clip(overall_confidence, 0, 1)
    overall_confidence = min(0.95, overall_confidence)

    # Adjust confidence based on critical thresholds
    adjusted_confidence = apply_critical_thresholds(
        overall_confidence, confidence_components
    )

    # Calculate reliability score
    num_metrics_with_weight = len(
        [w for w in adaptive_weights.values() if w["weight"] > 0]
    )
    if num_metrics_with_weight > 0:
        reliability_score = total_reliability_weight / num_metrics_with_weight
    else:
        reliability_score = 0

    return {
        "overall_confidence": adjusted_confidence,
        "raw_confidence": overall_confidence,
        "components": confidence_components,
        "adaptive_weights": adaptive_weights,
        "confidence_level": categorize_confidence(adjusted_confidence),
        "reliability_score": reliability_score,
        "available_metrics": available_metrics,
        "debug_info": {
            "weighted_sum": weighted_sum,
            "total_reliability_weight": total_reliability_weight,
            "num_components": len(confidence_components),
            "num_metrics_available": len(metrics),
        },
    }


def normalize_metric(metric_name, value):
    """Normalize different metrics to 0-1 scale (higher = better) with improved scaling."""

    # Handle edge cases
    if value is None or np.isnan(value) or np.isinf(value):
        print(f"DEBUG normalize_metric: Invalid value for {metric_name}: {value}")
        return 0

    if metric_name == "silhouette":
        # Silhouette score range: [-1, 1]
        # More nuanced scaling that doesn't penalize moderate clustering too harshly
        if value < -0.25:
            return 0  # Very poor clustering
        elif value < 0:
            # Scale -0.25 to 0 → 0 to 0.2
            return 0.2 * (1 + value / 0.25)
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
            return min(0.98, 0.85 + ((value - 0.7) / 0.3) * 0.13)

    elif metric_name == "hopkins":
        # Hopkins statistic range: [0, 1], higher is better
        # More generous scaling for hopkins
        if value < 0.5:
            return value * 0.4  # Poor clustering tendency
        elif value < 0.7:
            return 0.2 + ((value - 0.5) / 0.2) * 0.3  # Moderate
        else:
            return min(0.98, 0.5 + ((value - 0.7) / 0.3) * 0.48)  # Cap at 0.98

    elif metric_name == "stability":
        # Stability range: [0, 1], higher is better
        # Stability is often lower, so be more generous
        if value < 0.3:
            return value * 0.5
        elif value < 0.6:
            return 0.15 + ((value - 0.3) / 0.3) * 0.35
        else:
            return min(0.98, 0.5 + ((value - 0.6) / 0.4) * 0.48)  # Cap at 0.98

    elif metric_name == "physics_consistency":
        # Physics consistency range: [0, 1], higher is better
        return min(0.98, np.clip(value**0.7, 0, 1))  # Cap at 0.98

    elif metric_name == "davies_bouldin":
        # Davies-Bouldin range: [0, ∞], lower is better
        # More reasonable transformation
        if value <= 0.3:
            return 0.98  # Excellent - cap at 0.98
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

    elif metric_name == "calinski_harabasz":
        # Calinski-Harabasz range: [0, ∞], higher is better
        # Log transform for better scaling
        if value <= 0:
            return 0
        else:
            # Log scale that's more generous
            log_val = np.log1p(value / 10)  # log(1 + value/10)
            return min(0.98, np.tanh(log_val / 3))  # Cap at 0.98

    elif metric_name == "noise_ratio":
        # Noise ratio range: [0, 1], lower is better
        # More tolerance for noise
        if value < 0.1:
            return 0.98  # Excellent - cap at 0.98
        elif value < 0.2:
            return 0.9 - (value - 0.1) * 2  # Good
        elif value < 0.4:
            return 0.7 - (value - 0.2) * 2  # Moderate
        else:
            return max(0.1, 0.3 - (value - 0.4) * 0.5)  # Poor

    else:
        # Default: assume [0, 1] range, higher is better
        print(
            f"DEBUG normalize_metric: Unknown metric {metric_name}, using default normalization"
        )
        return min(0.98, np.clip(value, 0, 1))  # Cap at 0.98


def apply_critical_thresholds(confidence, components):
    """Apply critical thresholds with more reasonable penalties."""

    critical_failures = []
    moderate_issues = []

    # Check for critical failures (but be more lenient)
    if "silhouette" in components:
        sil_raw = components["silhouette"]["raw_value"]
        if sil_raw < -0.1:  # Changed from 0.1
            critical_failures.append(f"Silhouette score very low ({sil_raw:.3f})")
        elif sil_raw < 0.2:
            moderate_issues.append(f"Silhouette score moderate ({sil_raw:.3f})")

    if "hopkins" in components:
        hop_raw = components["hopkins"]["raw_value"]
        if hop_raw < 0.3:
            critical_failures.append(
                f"Hopkins statistic too low ({hop_raw:.3f}) - data appears random"
            )
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
    if "silhouette" in components and components["silhouette"]["raw_value"] > 0.6:
        exceptional_count += 1
    if "hopkins" in components and components["hopkins"]["raw_value"] > 0.75:
        exceptional_count += 1
    if "stability" in components and components["stability"]["raw_value"] > 0.8:
        exceptional_count += 1

    if exceptional_count >= 2:
        # Use asymptotic scaling instead of multiplicative boost
        boost_factor = 0.1 * (exceptional_count / 3)  # Max 0.1 boost for 3 metrics
        confidence = confidence + (1 - confidence) * boost_factor
        confidence = min(0.95, confidence)  # Cap at 0.95, reserve 1.0 for perfection

    return confidence


def apply_clustering_bonus(confidence_result, metrics):
    """Apply bonus points for exceptional clustering characteristics."""

    bonus = 0
    bonus_reasons = []

    # Check for exceptional silhouette score
    if "silhouette" in metrics and metrics["silhouette"] > 0.6:
        bonus += 0.1
        bonus_reasons.append("Excellent cluster separation")

    # Check for very low noise ratio
    if "noise_ratio" in metrics and metrics["noise_ratio"] < 0.05:
        bonus += 0.05
        bonus_reasons.append("Very low noise")

    # Check for high stability
    if "stability" in metrics and metrics["stability"] > 0.8:
        bonus += 0.05
        bonus_reasons.append("High cluster stability")

    # Check for good hopkins statistic
    if "hopkins" in metrics and metrics["hopkins"] > 0.8:
        bonus += 0.05
        bonus_reasons.append("Strong clustering tendency")

    # Apply bonus with cap
    if bonus > 0:
        original_confidence = confidence_result["overall_confidence"]
        asymptotic_bonus = bonus * (0.95 - original_confidence) / 0.95
        new_confidence = min(0.95, original_confidence + asymptotic_bonus)
        confidence_result["overall_confidence"] = new_confidence
        confidence_result["bonus_applied"] = bonus
        confidence_result["bonus_reasons"] = bonus_reasons

    return confidence_result


def categorize_confidence(score):
    """Categorize confidence score with realistic thresholds."""
    if score >= 0.8:  # Very rare, reserved for exceptional cases
        return {
            "level": "Excellent",
            "color": "darkgreen",
            "description": "Exceptionally reliable results",
        }
    elif score >= 0.65:
        return {"level": "High", "color": "green", "description": "Reliable results"}
    elif score >= 0.5:
        return {
            "level": "Moderate",
            "color": "orange",
            "description": "Reasonably reliable",
        }
    elif score >= 0.35:
        return {"level": "Low", "color": "red", "description": "Use with caution"}
    else:
        return {
            "level": "Very Low",
            "color": "darkred",
            "description": "Results may be unreliable",
        }


def analyze_confidence_context(confidence_result, metrics, characteristics):
    """Provide context-aware analysis and recommendations."""

    analysis = {
        "primary_factors": [],
        "concerns": [],
        "recommendations": [],
        "reliability_notes": [],
    }

    # Identify primary confidence drivers
    sorted_components = sorted(
        confidence_result["components"].items(),
        key=lambda x: x[1]["contribution"],
        reverse=True,
    )

    for metric, data in sorted_components[:2]:
        analysis["primary_factors"].append(
            f"{metric.replace('_', ' ').title()}: {data['normalized_value']:.2f} "
            f"(contributes {data['contribution']/confidence_result['raw_confidence']:.1%})"
        )

    # Identify concerns
    for metric, data in confidence_result["components"].items():
        if data["normalized_value"] < 0.4:
            analysis["concerns"].append(
                f"Low {metric.replace('_', ' ')}: {data['raw_value']:.3f}"
            )

    # Generate recommendations
    if confidence_result["overall_confidence"] < 0.5:
        analysis["recommendations"].extend(
            [
                "Consider different UMAP parameters (n_neighbors, min_dist)",
                "Try alternative feature selection or engineering",
                "Verify data quality and preprocessing",
            ]
        )

    if "silhouette" in confidence_result["components"]:
        sil_val = confidence_result["components"]["silhouette"]["raw_value"]
        if sil_val < 0.3:
            analysis["recommendations"].append(
                "Poor cluster separation - try increasing n_neighbors or different clustering algorithm"
            )

    # Add reliability notes
    for metric, weight_info in confidence_result["adaptive_weights"].items():
        if weight_info.get("excluded"):
            analysis["reliability_notes"].append(
                f"{metric}: {weight_info['exclusion_reason']}"
            )
        elif weight_info["tier"] == 3:
            analysis["reliability_notes"].append(f"{metric}: {weight_info['reason']}")

    return analysis


def get_metric_color(metric_name, value):
    """Get color for metric value based on thresholds."""
    if metric_name == "silhouette":
        if value > 0.5:
            return "green"
        elif value > 0.25:
            return "orange"
        else:
            return "red"
    elif metric_name == "davies_bouldin":
        if value < 0.8:
            return "green"
        elif value < 1.5:
            return "orange"
        else:
            return "red"
    elif metric_name == "calinski_harabasz":
        if value > 100:
            return "green"
        elif value > 50:
            return "orange"
        else:
            return "red"
    elif metric_name == "hopkins":
        if value > 0.75:
            return "green"
        elif value > 0.6:
            return "orange"
        else:
            return "red"
    elif metric_name == "stability":
        if value > 0.8:
            return "green"
        elif value > 0.6:
            return "orange"
        else:
            return "red"
    elif metric_name == "physics_consistency":
        if value > 0.6:
            return "green"
        elif value > 0.3:
            return "orange"
        else:
            return "red"
    elif metric_name == "noise_ratio":
        # For noise ratio, lower is better
        if value < 0.1:
            return "green"
        elif value < 0.3:
            return "orange"
        else:
            return "red"
    else:
        return "black"


def calculate_confidence_uncertainty(confidence_result, metrics):
    """Calculate uncertainty bounds for confidence score."""
    base_confidence = confidence_result["overall_confidence"]

    # Uncertainty increases with fewer metrics and lower reliability
    num_metrics = confidence_result["available_metrics"]
    reliability_score = confidence_result["reliability_score"]

    # Base uncertainty (higher with fewer metrics)
    base_uncertainty = 0.1 + max(0, (3 - num_metrics)) * 0.05

    # Adjust for reliability
    reliability_adjustment = max(0, (0.8 - reliability_score)) * 0.1

    total_uncertainty = base_uncertainty + reliability_adjustment

    return {
        "point_estimate": base_confidence,
        "lower_bound": max(0, base_confidence - total_uncertainty),
        "upper_bound": min(0.95, base_confidence + total_uncertainty),
        "uncertainty": total_uncertainty,
    }


def validate_confidence_score(confidence_result):
    """Validate and warn about potential issues with confidence score."""
    warnings = []

    if confidence_result["overall_confidence"] > 0.9:
        warnings.append("Very high confidence - verify this is justified")

    if confidence_result["available_metrics"] < 3:
        warnings.append("Limited metrics available - confidence may be less reliable")

    # Check for bonus stacking
    if (
        "bonus_applied" in confidence_result
        and confidence_result["bonus_applied"] > 0.15
    ):
        warnings.append("High bonus applied - may indicate over-optimistic scoring")

    return warnings
