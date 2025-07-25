from dash import dcc, html

from sculpt.utils.metrics.confidence_assessment import get_metric_color


def create_smart_confidence_ui(confidence_data):
    """Create an intelligent confidence UI."""

    if not confidence_data:
        return html.Div("No confidence data available")

    conf_score = confidence_data["overall_confidence"]
    conf_info = confidence_data["confidence_level"]
    analysis = confidence_data.get("analysis", {})
    components = confidence_data.get("components", {})  # Added this line - was missing

    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H4(
                        "UMAP Reliability Assessment",
                        style={
                            "fontSize": "16px",
                            "marginBottom": "5px",
                            "color": "#2e7d32",
                        },
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{conf_score:.2f}",
                                style={
                                    "fontSize": "28px",
                                    "fontWeight": "bold",
                                    "color": conf_info["color"],
                                    "marginRight": "15px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        conf_info["level"],
                                        style={
                                            "fontSize": "16px",
                                            "fontWeight": "bold",
                                            "color": conf_info["color"],
                                        },
                                    ),
                                    html.Div(
                                        conf_info["description"],
                                        style={
                                            "fontSize": "12px",
                                            "color": "gray",
                                            "fontStyle": "italic",
                                        },
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "15px",
                        },
                    ),
                ]
            ),
            # Confidence bar
            create_confidence_bar(conf_score, conf_info["color"]),
            # Raw Metrics Display
            html.Div(
                [
                    html.H5(
                        "Individual Metrics",
                        style={
                            "fontSize": "14px",
                            "marginTop": "15px",
                            "marginBottom": "8px",
                            "color": "#1976d2",
                            "borderBottom": "1px solid #e0e0e0",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        f"{metric.replace('_', ' ').title()}: ",
                                        style={"fontWeight": "bold"},
                                    ),
                                    html.Span(
                                        f"{data['raw_value']:.4f}",
                                        style={
                                            "color": get_metric_color(
                                                metric, data["raw_value"]
                                            )
                                        },
                                    ),
                                ],
                                style={"fontSize": "12px", "marginBottom": "3px"},
                            )
                            for metric, data in components.items()
                            if "raw_value" in data
                        ]
                    ),
                ]
            ),
            # Key indicators
            html.Div(
                [
                    html.H5(
                        "Key Quality Indicators",
                        style={
                            "fontSize": "14px",
                            "marginTop": "15px",
                            "marginBottom": "8px",
                            "color": "#1976d2",
                            "borderBottom": "1px solid #e0e0e0",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                factor,
                                style={"fontSize": "12px", "marginBottom": "3px"},
                            )
                            for factor in analysis.get("primary_factors", [])
                        ]
                        if analysis.get("primary_factors")
                        else [
                            html.Div(
                                "No primary factors identified",
                                style={"fontSize": "12px", "color": "gray"},
                            )
                        ]
                    ),
                ]
            ),
            # Recommendations
            (
                html.Div(
                    [
                        html.H6(
                            "💡 Recommendations",
                            style={
                                "fontSize": "12px",
                                "color": "#1976d2",
                                "marginTop": "12px",
                                "marginBottom": "5px",
                                "fontWeight": "bold",
                            },
                        ),
                        html.Ul(
                            [
                                html.Li(
                                    rec,
                                    style={"fontSize": "11px", "marginBottom": "2px"},
                                )
                                for rec in analysis.get(
                                    "recommendations", ["Results appear reliable"]
                                )
                            ],
                            style={"paddingLeft": "15px", "margin": "0"},
                        ),
                    ]
                )
                if analysis.get("recommendations")
                else html.Div(
                    [
                        html.H6(
                            "✓ Status",
                            style={
                                "fontSize": "12px",
                                "color": "#2e7d32",
                                "marginTop": "12px",
                                "marginBottom": "5px",
                                "fontWeight": "bold",
                            },
                        ),
                        html.Div(
                            "Results appear reliable based on current metrics",
                            style={"fontSize": "11px", "color": "#2e7d32"},
                        ),
                    ]
                )
            ),  # Added missing comma here
            # Metric explanations tooltip
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
                                        "• Davies-Bouldin Index: Measures average similarity between clusters (lower is better"
                                        ", range: 0 to ∞)"
                                    ),
                                    html.P(
                                        "• Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion "
                                        "(higher is better)"
                                    ),
                                    html.P(
                                        "• Hopkins Statistic: Measures clusterability of the data (>0.75 indicates good "
                                        "clustering)"
                                    ),
                                    html.P(
                                        "• Cluster Stability: How stable clusters are with small perturbations (higher is "
                                        "better)"
                                    ),
                                    html.P(
                                        "• Physics Consistency: How well clusters align with physical parameters (higher is "
                                        "better)"
                                    ),
                                ],
                                style={"fontSize": "11px", "paddingLeft": "10px"},
                            ),
                        ]
                    ),
                ],
                style={"marginTop": "10px"},
            ),
        ],
        style={
            "padding": "15px",
            "border": f"2px solid {conf_info['color']}",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
            "marginTop": "15px",
        },
    )


def create_confidence_bar(score, color):
    """Create a segmented confidence bar."""

    segments = [
        {"threshold": 0.0, "color": "#d32f2f", "label": "Very Low"},
        {"threshold": 0.4, "color": "#f57c00", "label": "Low"},
        {"threshold": 0.55, "color": "#fbc02d", "label": "Moderate"},
        {"threshold": 0.7, "color": "#689f38", "label": "High"},
        {"threshold": 0.85, "color": "#388e3c", "label": "Excellent"},
    ]

    bar_segments = []
    for i, segment in enumerate(segments):
        if i < len(segments) - 1:
            width = segments[i + 1]["threshold"] - segment["threshold"]
        else:
            width = 1.0 - segment["threshold"]

        is_active = score >= segment["threshold"] and (
            i == len(segments) - 1 or score < segments[i + 1]["threshold"]
        )

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
                    "boxSizing": "border-box",  # Add this
                },
            )
        )

    return html.Div(
        bar_segments,
        style={
            "position": "relative",
            "width": "100%",
            "marginBottom": "15px",
            "display": "flex",  # Change from default to flex
            "flexDirection": "row",  # Ensure horizontal layout
            "flexWrap": "nowrap",  # Prevent wrapping
        },
    )


def create_feature_categories_ui(feature_columns, id_prefix):
    """Create the feature selection UI organized by categories."""
    # Group features into categories
    feature_categories = {
        "Original Momentum": [
            col for col in feature_columns if col.startswith("particle_")
        ],
        "Momentum Magnitudes": [col for col in feature_columns if "mom_mag" in col],
        "Energies": [
            col
            for col in feature_columns
            if any(
                x in col
                for x in ["energy_", "KER", "EESum", "TotalEnergy", "EESharing"]
            )
        ],
        "Angles": [
            col
            for col in feature_columns
            if any(x in col for x in ["theta_", "phi_", "angle_"])
        ],
        "Dot Products": [col for col in feature_columns if "dot_product" in col],
        "Differences": [
            col
            for col in feature_columns
            if any(x in col for x in ["diff_", "mom_diff"])
        ],
    }

    # Create the selection UI with feature categories
    feature_selection_ui = []
    for category, cols in feature_categories.items():
        if cols:  # Only add categories that have features
            category_ui = html.Div(
                [
                    html.Div(category, className="feature-category-title"),
                    dcc.Checklist(
                        id={
                            "type": f"feature-selector-{id_prefix}",
                            "category": category,
                        },
                        options=[{"label": col, "value": col} for col in cols],
                        value=[],  # No default selection
                        labelStyle={"display": "block"},
                    ),
                ],
                className="feature-category",
            )
            feature_selection_ui.append(category_ui)

    if not feature_selection_ui:
        feature_selection_ui = [
            html.Div(
                "No features available. Please upload files.", style={"color": "gray"}
            )
        ]

    return feature_selection_ui
