from datetime import datetime

from dash import dash_table, dcc, html


# TODO: Split this into multiple files
def create_autoencoder_tab():
    return [
        html.H3(
            "Deep Autoencoder with UMAP on Latent Space", style={"textAlign": "center"}
        ),
        # Main container with proper flex layout
        html.Div(
            [
                # Left panel - controls (fixed width)
                html.Div(
                    [
                        # Feature Selection Section
                        html.Div(
                            [
                                html.H4(
                                    "Select Features for Autoencoder:",
                                    style={"color": "#1976d2", "marginBottom": "10px"},
                                ),
                                # Original Features
                                html.Div(
                                    [
                                        html.H5(
                                            "Original Features:",
                                            style={
                                                "fontSize": "14px",
                                                "color": "#388e3c",
                                            },
                                        ),
                                        html.Div(
                                            id="feature-selection-ui-autoencoder",
                                            children=[
                                                html.Div(
                                                    "Upload files to see available features",
                                                    style={"color": "gray"},
                                                )
                                            ],
                                            className="feature-checklist",
                                            style={
                                                "maxHeight": "200px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "marginBottom": "15px",
                                        "padding": "10px",
                                        "backgroundColor": "#e8f5e9",
                                        "borderRadius": "5px",
                                    },
                                ),
                                # Genetic Features (if available)
                                html.Div(
                                    [
                                        html.H5(
                                            "Genetic Features:",
                                            style={
                                                "fontSize": "14px",
                                                "color": "#d32f2f",
                                            },
                                        ),
                                        html.Div(
                                            id="genetic-feature-selection-autoencoder",
                                            children=[
                                                html.Div(
                                                    "Run Genetic Feature Discovery first to see"
                                                    " genetic features",
                                                    style={
                                                        "color": "gray",
                                                        "fontStyle": "italic",
                                                    },
                                                )
                                            ],
                                            style={
                                                "maxHeight": "150px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "marginBottom": "15px",
                                        "padding": "10px",
                                        "backgroundColor": "#ffebee",
                                        "borderRadius": "5px",
                                    },
                                ),
                                # Feature combination summary
                                html.Div(
                                    id="feature-combination-summary",
                                    children=[
                                        html.Div(
                                            "Select features above to see combination summary",
                                            style={"color": "gray", "fontSize": "12px"},
                                        )
                                    ],
                                    style={
                                        "marginBottom": "15px",
                                        "padding": "8px",
                                        "backgroundColor": "#f5f5f5",
                                        "borderRadius": "5px",
                                        "fontSize": "12px",
                                    },
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        # Autoencoder Parameters
                        html.Div(
                            [
                                html.H4(
                                    "Autoencoder Parameters:",
                                    style={"color": "#1976d2", "marginBottom": "10px"},
                                ),
                                html.Label("Latent Dimension Size:"),
                                dcc.Input(
                                    id="autoencoder-latent-dim",
                                    type="number",
                                    value=7,
                                    min=2,
                                    max=20,
                                ),
                                html.Br(),
                                html.Label("Number of Epochs:"),
                                dcc.Input(
                                    id="autoencoder-epochs",
                                    type="number",
                                    value=50,
                                    min=10,
                                    max=500,
                                ),
                                html.Br(),
                                html.Label("Batch Size:"),
                                dcc.Input(
                                    id="autoencoder-batch-size",
                                    type="number",
                                    value=64,
                                    min=8,
                                    max=512,
                                ),
                                html.Br(),
                                html.Label("Learning Rate:"),
                                dcc.Input(
                                    id="autoencoder-learning-rate",
                                    type="number",
                                    value=0.001,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                ),
                            ],
                            style={
                                "marginBottom": "20px",
                                "padding": "10px",
                                "backgroundColor": "#f0f4ff",
                                "borderRadius": "5px",
                            },
                        ),
                        # Data Source Selection with Better Explanations
                        html.Div(
                            [
                                html.H4(
                                    "Data Source:",
                                    style={"color": "#1976d2", "marginBottom": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Select which data to use for training:",
                                            style={
                                                "fontWeight": "bold",
                                                "marginBottom": "10px",
                                            },
                                        ),
                                        dcc.RadioItems(
                                            id="autoencoder-data-source",
                                            options=[
                                                {
                                                    "label": html.Div(
                                                        [
                                                            html.Span(
                                                                "All Data",
                                                                style={
                                                                    "fontWeight": "bold"
                                                                },
                                                            ),
                                                            html.Br(),
                                                            html.Span(
                                                                "Use complete dataset from all files",
                                                                style={
                                                                    "fontSize": "11px",
                                                                    "color": "gray",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    "value": "all",
                                                },
                                                {
                                                    "label": html.Div(
                                                        [
                                                            html.Span(
                                                                "Graph 1 Selection",
                                                                style={
                                                                    "fontWeight": "bold"
                                                                },
                                                            ),
                                                            html.Br(),
                                                            html.Span(
                                                                "Use points selected from main UMAP plot",
                                                                style={
                                                                    "fontSize": "11px",
                                                                    "color": "gray",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    "value": "graph1-selection",
                                                },
                                                {
                                                    "label": html.Div(
                                                        [
                                                            html.Span(
                                                                "Graph 3 Selection",
                                                                style={
                                                                    "fontWeight": "bold"
                                                                },
                                                            ),
                                                            html.Br(),
                                                            html.Span(
                                                                "Use points selected from re-run UMAP "
                                                                "plot",
                                                                style={
                                                                    "fontSize": "11px",
                                                                    "color": "gray",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    "value": "graph3-selection",
                                                },
                                            ],
                                            value="all",
                                            labelStyle={
                                                "display": "block",
                                                "marginBottom": "10px",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={
                                "marginBottom": "20px",
                                "padding": "10px",
                                "backgroundColor": "#fff3e0",
                                "borderRadius": "5px",
                            },
                        ),
                        html.Button(
                            "Train Autoencoder",
                            id="train-autoencoder",
                            n_clicks=0,
                            className="btn-secondary",
                        ),
                        html.Div(
                            id="train-autoencoder-status",
                            style={
                                "marginTop": "5px",
                                "fontSize": "12px",
                                "color": "blue",
                            },
                        ),
                        html.Div(
                            id="training-progress",
                            style={
                                "marginTop": "5px",
                                "fontSize": "12px",
                                "color": "green",
                            },
                        ),
                        html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                        # UMAP Parameters
                        html.Label("UMAP on Latent Space:"),
                        html.Div(
                            [
                                html.Label("UMAP Neighbors:"),
                                dcc.Input(
                                    id="autoencoder-umap-neighbors",
                                    type="number",
                                    value=15,
                                    min=1,
                                ),
                                html.Br(),
                                html.Label("Min Dist:"),
                                dcc.Input(
                                    id="autoencoder-umap-min-dist",
                                    type="number",
                                    value=0.1,
                                    step=0.01,
                                    min=0,
                                ),
                                html.Br(),
                                html.Button(
                                    "Run UMAP on Latent Space",
                                    id="run-umap-latent",
                                    n_clicks=0,
                                    className="btn-secondary",
                                ),
                                html.Div(
                                    id="run-umap-latent-status",
                                    style={
                                        "marginTop": "5px",
                                        "fontSize": "12px",
                                        "color": "blue",
                                    },
                                ),
                                html.Br(),
                                html.Label("Select Metrics to Calculate:"),
                                dcc.Checklist(
                                    id="metric-selector-autoencoder",
                                    options=[
                                        {
                                            "label": "Silhouette Score",
                                            "value": "silhouette",
                                        },
                                        {
                                            "label": "Davies-Bouldin Index",
                                            "value": "davies_bouldin",
                                        },
                                        {
                                            "label": "Calinski-Harabasz Index",
                                            "value": "calinski_harabasz",
                                        },
                                        {
                                            "label": "Hopkins Statistic",
                                            "value": "hopkins",
                                        },
                                        {
                                            "label": "Cluster Stability",
                                            "value": "stability",
                                        },
                                    ],
                                    value=[
                                        "silhouette",
                                        "davies_bouldin",
                                        "calinski_harabasz",
                                    ],
                                    labelStyle={"display": "block"},
                                ),
                            ]
                        ),
                        html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),
                        # Save Section
                        html.Label("Save Latent Features:"),
                        html.Div(
                            [
                                dcc.Input(
                                    id="latent-features-filename",
                                    type="text",
                                    placeholder="Enter filename (without extension)",
                                    style={"width": "100%", "marginBottom": "10px"},
                                ),
                                html.Button(
                                    "Save Latent Features",
                                    id="save-latent-features-btn",
                                    n_clicks=0,
                                    className="btn-secondary",
                                ),
                                html.Div(
                                    id="save-latent-features-status",
                                    style={
                                        "marginTop": "5px",
                                        "fontSize": "12px",
                                        "color": "blue",
                                    },
                                ),
                                dcc.Download(id="download-latent-features"),
                            ]
                        ),
                    ],
                    style={
                        "width": "30%",
                        "paddingRight": "20px",
                        "verticalAlign": "top",
                    },
                ),
                # Right panel - graph and metrics (flexible width)
                html.Div(
                    [
                        # Graph container with fixed height
                        html.Div(
                            [
                                dcc.Graph(
                                    id="autoencoder-umap-graph",
                                    config={"displayModeBar": True},
                                    style={"height": "550px", "width": "100%"},
                                )
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        # Debug output
                        html.Div(
                            id="autoencoder-debug-output",
                            style={
                                "marginTop": "10px",
                                "fontSize": "12px",
                                "color": "gray",
                                "marginBottom": "10px",
                            },
                        ),
                        # Metrics container
                        html.Div(
                            id="umap-quality-metrics-autoencoder",
                            children=[],
                            style={
                                "marginTop": "10px",
                                "padding": "10px",
                                "border": "1px solid #ddd",
                                "borderRadius": "5px",
                                "backgroundColor": "#f9f9f9",
                                "marginBottom": "20px",
                            },
                        ),
                        # Feature importance container with controlled height
                        html.Div(
                            id="feature-importance-container",
                            children=[],
                            style={
                                "marginTop": "15px",
                                "maxHeight": "400px",
                                "overflowY": "auto",
                            },
                        ),
                    ],
                    style={"width": "70%", "verticalAlign": "top"},
                ),
            ],
            style={"display": "flex", "alignItems": "flex-start"},
        ),
    ]


def create_genetic_tab():
    return [
        html.H3("Genetic Feature Engineering", style={"textAlign": "center"}),
        # Add explanation section at the top
        html.Div(
            [
                html.H4(
                    "How Genetic Feature Engineering Works:", style={"color": "#1976d2"}
                ),
                html.Div(
                    [
                        html.P(
                            "1. **Select Original Features**: Choose which momentum/energy features to use as building"
                            " blocks"
                        ),
                        html.P(
                            "2. **Choose Clustering Method**: This creates a target for the genetic algorithm to optimize"
                            "against"
                        ),
                        html.P(
                            "3. **Select Data Source**: Choose which subset of your data to analyze"
                        ),
                        html.P(
                            "4. **Run Discovery**: The algorithm creates new features by combining original ones"
                            "(e.g., X1*X2, X1+X3, etc.)"
                        ),
                        html.P(
                            "5. **Use in Autoencoder**: Combine discovered features with original features for enhanced"
                            "analysis"
                        ),
                    ],
                    style={
                        "fontSize": "12px",
                        "padding": "10px",
                        "backgroundColor": "#e3f2fd",
                        "borderRadius": "5px",
                    },
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        # Feature Selection
                        html.Div(
                            [
                                html.H4(
                                    "1. Select Original Features:",
                                    style={"color": "#2e7d32"},
                                ),
                                html.Div(
                                    "These will be used as building blocks for creating new features",
                                    style={
                                        "fontSize": "12px",
                                        "color": "gray",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Div(
                                    id="feature-selection-ui-genetic",
                                    children=[
                                        html.Div(
                                            "Upload files to see available features",
                                            style={"color": "gray"},
                                        )
                                    ],
                                    className="feature-checklist",
                                ),
                            ],
                            style={
                                "marginBottom": "20px",
                                "padding": "10px",
                                "backgroundColor": "#e8f5e9",
                                "borderRadius": "5px",
                            },
                        ),
                        # Clustering Section with Better Explanation
                        html.Div(
                            [
                                html.H4(
                                    "2. Clustering for Target Creation:",
                                    style={"color": "#d32f2f"},
                                ),
                                html.Div(
                                    [
                                        html.I(
                                            "The genetic algorithm needs a target to optimize against. "
                                        ),
                                        html.I(
                                            "We cluster your data and use the cluster structure as the target. "
                                        ),
                                        html.I(
                                            "This helps discover features that capture natural groupings in your data."
                                        ),
                                    ],
                                    style={
                                        "fontSize": "12px",
                                        "color": "gray",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Label("Clustering Method:"),
                                dcc.Dropdown(
                                    id="clustering-method",
                                    options=[
                                        {
                                            "label": "DBSCAN (finds arbitrary shaped clusters)",
                                            "value": "dbscan",
                                        },
                                        {
                                            "label": "KMeans (finds spherical clusters)",
                                            "value": "kmeans",
                                        },
                                        {
                                            "label": "Agglomerative (hierarchical clustering)",
                                            "value": "agglomerative",
                                        },
                                    ],
                                    value="dbscan",
                                ),
                                html.Div(
                                    id="dbscan-params",
                                    children=[
                                        html.Label("DBSCAN Epsilon (cluster size):"),
                                        dcc.Input(
                                            id="dbscan-eps",
                                            type="number",
                                            value=0.3,
                                            min=0.01,
                                            max=2.0,
                                            step=0.05,
                                        ),
                                        html.Br(),
                                        html.Label(
                                            "DBSCAN Min Samples (density threshold):"
                                        ),
                                        dcc.Input(
                                            id="dbscan-min-samples",
                                            type="number",
                                            value=5,
                                            min=2,
                                            max=50,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="kmeans-params",
                                    children=[
                                        html.Label("Number of Clusters:"),
                                        dcc.Input(
                                            id="kmeans-n-clusters",
                                            type="number",
                                            value=5,
                                            min=2,
                                            max=20,
                                        ),
                                    ],
                                    style={"display": "none"},
                                ),
                                html.Div(
                                    id="agglomerative-params",
                                    children=[
                                        html.Label("Number of Clusters:"),
                                        dcc.Input(
                                            id="agglomerative-n-clusters",
                                            type="number",
                                            value=5,
                                            min=2,
                                            max=20,
                                        ),
                                        html.Br(),
                                        html.Label("Linkage Method:"),
                                        dcc.Dropdown(
                                            id="agglomerative-linkage",
                                            options=[
                                                {
                                                    "label": "Ward (minimizes variance)",
                                                    "value": "ward",
                                                },
                                                {
                                                    "label": "Complete (maximum linkage)",
                                                    "value": "complete",
                                                },
                                                {
                                                    "label": "Average (average linkage)",
                                                    "value": "average",
                                                },
                                                {
                                                    "label": "Single (minimum linkage)",
                                                    "value": "single",
                                                },
                                            ],
                                            value="ward",
                                        ),
                                    ],
                                    style={"display": "none"},
                                ),
                            ],
                            style={
                                "marginBottom": "20px",
                                "padding": "10px",
                                "backgroundColor": "#ffebee",
                                "borderRadius": "5px",
                            },
                        ),
                        # Data Source with Better Explanations
                        html.Div(
                            [
                                html.H4(
                                    "3. Select Data Source:", style={"color": "#f57c00"}
                                ),
                                dcc.RadioItems(
                                    id="genetic-data-source",
                                    options=[
                                        {
                                            "label": html.Div(
                                                [
                                                    html.Span(
                                                        "All Data",
                                                        style={"fontWeight": "bold"},
                                                    ),
                                                    html.Br(),
                                                    html.Span(
                                                        "Use complete dataset (recommended for most cases)",
                                                        style={
                                                            "fontSize": "11px",
                                                            "color": "gray",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            "value": "all",
                                        },
                                        {
                                            "label": html.Div(
                                                [
                                                    html.Span(
                                                        "Graph 1 Selection",
                                                        style={"fontWeight": "bold"},
                                                    ),
                                                    html.Br(),
                                                    html.Span(
                                                        "Use only points selected from main UMAP plot",
                                                        style={
                                                            "fontSize": "11px",
                                                            "color": "gray",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            "value": "graph1-selection",
                                        },
                                        {
                                            "label": html.Div(
                                                [
                                                    html.Span(
                                                        "Graph 3 Selection",
                                                        style={"fontWeight": "bold"},
                                                    ),
                                                    html.Br(),
                                                    html.Span(
                                                        "Use points from re-run UMAP analysis",
                                                        style={
                                                            "fontSize": "11px",
                                                            "color": "gray",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            "value": "graph3-selection",
                                        },
                                        {
                                            "label": html.Div(
                                                [
                                                    html.Span(
                                                        "Autoencoder Latent Space",
                                                        style={"fontWeight": "bold"},
                                                    ),
                                                    html.Br(),
                                                    html.Span(
                                                        "Use compressed features from autoencoder",
                                                        style={
                                                            "fontSize": "11px",
                                                            "color": "gray",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            "value": "autoencoder-latent",
                                        },
                                    ],
                                    value="all",
                                    labelStyle={
                                        "display": "block",
                                        "marginBottom": "10px",
                                    },
                                ),
                            ],
                            style={
                                "marginBottom": "20px",
                                "padding": "10px",
                                "backgroundColor": "#fff3e0",
                                "borderRadius": "5px",
                            },
                        ),
                        # Genetic Programming Parameters
                        html.H4("4. Feature Engineering Parameters:"),
                        html.Label("Number of Generations:"),
                        dcc.Input(
                            id="gp-generations", type="number", value=20, min=5, max=100
                        ),
                        html.Br(),
                        html.Label("Population Size:"),
                        dcc.Input(
                            id="gp-population-size",
                            type="number",
                            value=1000,
                            min=500,
                            max=5000,
                            step=100,
                        ),
                        html.Br(),
                        html.Label("Number of Features to Generate:"),
                        dcc.Input(
                            id="gp-n-components", type="number", value=10, min=2, max=20
                        ),
                        html.Br(),
                        html.Label("Mathematical Functions:"),
                        dcc.Checklist(
                            id="gp-functions",
                            options=[
                                {
                                    "label": "Basic (add, sub, mul, div)",
                                    "value": "basic",
                                },
                                {
                                    "label": "Trigonometric (sin, cos, tan)",
                                    "value": "trig",
                                },
                                {
                                    "label": "Exponential & Logarithmic",
                                    "value": "exp_log",
                                },
                                {"label": "Square Root & Power", "value": "sqrt_pow"},
                                {"label": "Special (abs, inv)", "value": "special"},
                            ],
                            value=["basic", "trig", "sqrt_pow", "special"],
                        ),
                        html.Br(),
                        html.Div(
                            [
                                html.Label("Select Data Source:"),
                                dcc.RadioItems(
                                    id="genetic-data-source",
                                    options=[
                                        {
                                            "label": "All Data",
                                            "value": "all",
                                        },
                                        {
                                            "label": "Graph 1 Selection",
                                            "value": "graph1-selection",
                                        },
                                        {
                                            "label": "Graph 3 Selection",
                                            "value": "graph3-selection",
                                        },
                                        {
                                            "label": "Autoencoder Latent Space",
                                            "value": "autoencoder-latent",
                                        },
                                    ],
                                    value="all",
                                    labelStyle={
                                        "display": "block",
                                        "marginBottom": "5px",
                                    },
                                ),
                            ]
                        ),
                        html.Br(),
                        html.Button(
                            "Run Genetic Feature Discovery",
                            id="run-genetic-features",
                            n_clicks=0,
                            className="btn-secondary",
                        ),
                        html.Div(
                            id="genetic-features-status",
                            style={
                                "marginTop": "5px",
                                "fontSize": "12px",
                                "color": "blue",
                            },
                        ),
                        html.Br(),
                        # Visualization Section
                        html.Div(
                            [
                                html.H4(
                                    "5. Visualize Discovered Features:",
                                    style={"marginTop": "15px", "color": "#2e7d32"},
                                ),
                                html.Div(
                                    "After feature discovery, select which features to visualize with UMAP:",
                                    style={
                                        "color": "#555",
                                        "fontSize": "12px",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Button(
                                    "Run UMAP on Genetic Features",
                                    id="run-umap-genetic",
                                    n_clicks=0,
                                    className="btn-secondary",
                                    style={
                                        "backgroundColor": "#2e7d32",
                                        "marginTop": "5px",
                                    },
                                ),
                                html.Div(
                                    id="run-umap-genetic-status",
                                    style={
                                        "marginTop": "5px",
                                        "fontSize": "12px",
                                        "color": "blue",
                                    },
                                ),
                                html.Br(),
                                html.Label("Select Metrics to Calculate:"),
                                dcc.Checklist(
                                    id="metric-selector-genetic",
                                    options=[
                                        {
                                            "label": "Silhouette Score",
                                            "value": "silhouette",
                                        },
                                        {
                                            "label": "Davies-Bouldin Index",
                                            "value": "davies_bouldin",
                                        },
                                        {
                                            "label": "Calinski-Harabasz Index",
                                            "value": "calinski_harabasz",
                                        },
                                        {
                                            "label": "Hopkins Statistic",
                                            "value": "hopkins",
                                        },
                                        {
                                            "label": "Cluster Stability",
                                            "value": "stability",
                                        },
                                    ],
                                    value=[
                                        "silhouette",
                                        "davies_bouldin",
                                        "calinski_harabasz",
                                    ],
                                    labelStyle={"display": "block"},
                                ),
                            ],
                            style={
                                "border": "1px solid #c8e6c9",
                                "padding": "10px",
                                "borderRadius": "5px",
                                "backgroundColor": "#f1f8e9",
                            },
                        ),
                        html.Hr(
                            style={
                                "marginTop": "20px",
                                "marginBottom": "20px",
                            }
                        ),
                        html.Label("Save Discovered Features:"),
                        html.Div(
                            [
                                dcc.Input(
                                    id="genetic-features-filename",
                                    type="text",
                                    placeholder="Enter filename (without extension)",
                                    style={
                                        "width": "100%",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Button(
                                    "Save Genetic Features",
                                    id="save-genetic-features-btn",
                                    n_clicks=0,
                                    className="btn-secondary",
                                ),
                                html.Div(
                                    id="save-genetic-features-status",
                                    style={
                                        "marginTop": "5px",
                                        "fontSize": "12px",
                                        "color": "blue",
                                    },
                                ),
                                dcc.Download(id="download-genetic-features"),
                            ]
                        ),
                    ],
                    style={
                        "width": "25%",
                        "paddingRight": "20px",
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="genetic-features-graph",
                            config={"displayModeBar": True},
                            style={"height": "600px"},
                        ),
                        html.Div(
                            id="genetic-features-debug-output",
                            style={
                                "marginTop": "10px",
                                "fontSize": "12px",
                                "color": "gray",
                            },
                        ),
                        html.Div(
                            id="umap-quality-metrics-genetic",
                            children=[],
                            style={
                                "marginTop": "10px",
                                "padding": "10px",
                                "border": "1px solid #ddd",
                                "borderRadius": "5px",
                                "backgroundColor": "#f9f9f9",
                            },
                        ),
                    ],
                    style={"width": "75%"},
                ),
            ],
            style={"display": "flex"},
        ),
    ]


def create_machine_learning_tab():
    return [
        dcc.Tabs(
            id="ml-sub-tabs",
            value="autoencoder-tab",
            children=[
                # Autoencoder Tab
                dcc.Tab(
                    label="Deep Autoencoder",
                    value="autoencoder-tab",
                    children=[
                        html.Div(create_autoencoder_tab(), className="container")
                    ],
                ),
                # Genetic Features Tab
                dcc.Tab(
                    label="Genetic Feature Engineering",
                    value="genetic-tab",
                    children=[
                        html.Div(
                            create_genetic_tab(),
                            className="container",
                        )
                    ],
                ),
                # Mutual Information Tab
                dcc.Tab(
                    label="Mutual Information Feature Selection",
                    value="mi-tab",
                    children=create_mutual_information_tab(),
                ),
            ],
        )
    ]


def create_mutual_information_tab():
    return html.Div(
        [
            html.H3(
                "Mutual Information Feature Selection",
                style={"textAlign": "center"},
            ),
            # Explanation section
            html.Div(
                [
                    html.H4(
                        "How Mutual Information Feature Selection Works:",
                        style={"color": "#d32f2f"},
                    ),
                    html.Div(
                        [
                            html.P(
                                "Mutual Information (MI) measures how much information one variable "
                                "provides about another. This method selects features that are most "
                                "informative about your target variables while avoiding redundancy."
                            ),
                            html.P(
                                "Benefits: • Captures non-linear relationships • Removes redundant "
                                "features • Optimizes for prediction accuracy"
                            ),
                        ],
                        style={
                            "backgroundColor": "#ffebee",
                            "padding": "15px",
                            "borderRadius": "5px",
                            "marginBottom": "20px",
                        },
                    ),
                ]
            ),
            # Two-column layout
            html.Div(
                [
                    # Left column - Controls
                    html.Div(
                        [
                            # Step 1: Select Features
                            html.Div(
                                [
                                    html.H4(
                                        "Step 1: Select Features",
                                        style={
                                            "color": "#1976d2",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        "Choose which feature categories to include:",
                                        style={
                                            "fontSize": "12px",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(id="mi-feature-selection-ui", children=[]),
                                    html.Div(
                                        id="mi-feature-summary",
                                        style={
                                            "marginTop": "10px",
                                            "fontSize": "12px",
                                            "padding": "10px",
                                            "backgroundColor": "#f5f5f5",
                                            "borderRadius": "5px",
                                            "maxHeight": "150px",
                                            "overflowY": "auto",
                                        },
                                    ),
                                ],
                                style={
                                    "marginBottom": "20px",
                                    "padding": "10px",
                                    "backgroundColor": "#e3f2fd",
                                    "borderRadius": "5px",
                                },
                            ),
                            # Step 2: Configure MI Parameters
                            html.Div(
                                [
                                    html.H4(
                                        "Step 2: Configure Parameters",
                                        style={
                                            "color": "#388e3c",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Target Variables:",
                                                style={"fontWeight": "bold"},
                                            ),
                                            dcc.Dropdown(
                                                id="mi-target-variables",
                                                options=[
                                                    {"label": "KER", "value": "KER"},
                                                    {
                                                        "label": "EESum",
                                                        "value": "EESum",
                                                    },
                                                    {
                                                        "label": "EESharing",
                                                        "value": "EESharing",
                                                    },
                                                    {
                                                        "label": "TotalEnergy",
                                                        "value": "TotalEnergy",
                                                    },
                                                ],
                                                value=["KER", "EESum", "TotalEnergy"],
                                                multi=True,
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Label(
                                                "Redundancy Threshold:",
                                                style={"fontWeight": "bold"},
                                            ),
                                            dcc.Input(
                                                id="mi-redundancy-threshold",
                                                type="number",
                                                value=0.5,
                                                min=0.1,
                                                max=0.9,
                                                step=0.1,
                                                style={
                                                    "width": "100px",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            html.Div(
                                                "Features with correlation above this threshold will be "
                                                "considered redundant",
                                                style={
                                                    "fontSize": "11px",
                                                    "color": "gray",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            html.Label(
                                                "Maximum Features:",
                                                style={"fontWeight": "bold"},
                                            ),
                                            dcc.Input(
                                                id="mi-max-features",
                                                type="number",
                                                value=20,
                                                min=5,
                                                max=50,
                                                style={"width": "100px"},
                                            ),
                                        ]
                                    ),
                                ],
                                style={
                                    "marginBottom": "20px",
                                    "padding": "10px",
                                    "backgroundColor": "#e8f5e9",
                                    "borderRadius": "5px",
                                },
                            ),
                            # Step 3: Data Source Selection
                            html.Div(
                                [
                                    html.H4(
                                        "Step 3: Select Data Source",
                                        style={
                                            "color": "#7b1fa2",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    dcc.RadioItems(
                                        id="mi-data-source",
                                        options=[
                                            {"label": "All Data", "value": "all"},
                                            {
                                                "label": "Graph 1 Selection",
                                                "value": "graph1",
                                            },
                                            {
                                                "label": "Graph 3 Selection",
                                                "value": "graph3",
                                            },
                                        ],
                                        value="all",
                                        labelStyle={
                                            "display": "block",
                                            "marginBottom": "5px",
                                        },
                                    ),
                                ],
                                style={
                                    "marginBottom": "20px",
                                    "padding": "10px",
                                    "backgroundColor": "#f3e5f5",
                                    "borderRadius": "5px",
                                },
                            ),
                            # Step 4: Run MI Feature Selection
                            html.Div(
                                [
                                    html.H4(
                                        "Step 4: Run MI Feature Selection",
                                        style={
                                            "color": "#d32f2f",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        "This will analyze your features and rank them by importance:",
                                        style={
                                            "fontSize": "12px",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Button(
                                        "Run MI Feature Selection",
                                        id="run-mi-features",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "#d32f2f",
                                            "color": "white",
                                            "padding": "10px 20px",
                                            "fontSize": "14px",
                                        },
                                    ),
                                    html.Div(
                                        id="mi-features-status",
                                        style={
                                            "marginTop": "10px",
                                            "fontSize": "12px",
                                            "color": "blue",
                                        },
                                    ),
                                ],
                                style={
                                    "marginBottom": "20px",
                                    "padding": "10px",
                                    "backgroundColor": "#ffebee",
                                    "borderRadius": "5px",
                                },
                            ),
                            # Step 5: Autoencoder Training
                            html.Div(
                                [
                                    html.H4(
                                        "Step 5: Train Autoencoder",
                                        style={
                                            "color": "#1976d2",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        "Configure and train the autoencoder on selected features:",
                                        style={
                                            "fontSize": "12px",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Latent Dimension Size:"),
                                            dcc.Input(
                                                id="mi-latent-dim",
                                                type="number",
                                                value=7,
                                                min=2,
                                                max=20,
                                            ),
                                            html.Br(),
                                            html.Label("Number of Epochs:"),
                                            dcc.Input(
                                                id="mi-epochs",
                                                type="number",
                                                value=100,
                                                min=10,
                                                max=500,
                                            ),
                                            html.Br(),
                                            html.Label("Batch Size:"),
                                            dcc.Input(
                                                id="mi-batch-size",
                                                type="number",
                                                value=32,
                                                min=8,
                                                max=128,
                                            ),
                                            html.Br(),
                                            html.Label("Learning Rate:"),
                                            dcc.Input(
                                                id="mi-learning-rate",
                                                type="number",
                                                value=0.001,
                                                min=0.0001,
                                                max=0.1,
                                                step=0.0001,
                                            ),
                                        ],
                                        style={"marginBottom": "10px"},
                                    ),
                                    html.Button(
                                        "Train Autoencoder",
                                        id="train-mi-autoencoder",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "#1976d2",
                                            "color": "white",
                                            "padding": "10px 20px",
                                            "fontSize": "14px",
                                        },
                                    ),
                                    html.Div(
                                        id="train-mi-autoencoder-status",
                                        style={
                                            "marginTop": "10px",
                                            "fontSize": "12px",
                                            "color": "blue",
                                        },
                                    ),
                                ],
                                style={
                                    "marginBottom": "20px",
                                    "padding": "10px",
                                    "backgroundColor": "#e3f2fd",
                                    "borderRadius": "5px",
                                },
                            ),
                            # Step 6: Run UMAP
                            html.Div(
                                [
                                    html.H4(
                                        "Step 6: Run UMAP",
                                        style={
                                            "color": "#388e3c",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        "Visualize the latent space:",
                                        style={
                                            "fontSize": "12px",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label("n_neighbors:"),
                                            dcc.Input(
                                                id="mi-umap-neighbors",
                                                type="number",
                                                value=15,
                                                min=5,
                                                max=50,
                                            ),
                                            html.Br(),
                                            html.Label("min_dist:"),
                                            dcc.Input(
                                                id="mi-umap-min-dist",
                                                type="number",
                                                value=0.1,
                                                min=0.01,
                                                max=1.0,
                                                step=0.01,
                                            ),
                                            html.Br(),
                                            html.Label("Select Metrics to Calculate:"),
                                            dcc.Checklist(
                                                id="metric-selector-mi",
                                                options=[
                                                    {
                                                        "label": "Silhouette Score",
                                                        "value": "silhouette",
                                                    },
                                                    {
                                                        "label": "Davies-Bouldin Index",
                                                        "value": "davies_bouldin",
                                                    },
                                                    {
                                                        "label": "Calinski-Harabasz Index",
                                                        "value": "calinski_harabasz",
                                                    },
                                                    {
                                                        "label": "Hopkins Statistic",
                                                        "value": "hopkins",
                                                    },
                                                    {
                                                        "label": "Cluster Stability",
                                                        "value": "cluster_stability",
                                                    },
                                                    {
                                                        "label": "Physics Consistency",
                                                        "value": "physics_consistency",
                                                    },
                                                ],
                                                value=["silhouette", "hopkins"],
                                                labelStyle={
                                                    "display": "block",
                                                    "marginBottom": "5px",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "10px"},
                                    ),
                                    html.Button(
                                        "Run UMAP",
                                        id="run-umap-mi",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": "#388e3c",
                                            "color": "white",
                                            "padding": "10px 20px",
                                            "fontSize": "14px",
                                        },
                                    ),
                                    html.Div(
                                        id="run-umap-mi-status",
                                        style={
                                            "marginTop": "10px",
                                            "fontSize": "12px",
                                            "color": "blue",
                                        },
                                    ),
                                ],
                                style={
                                    "marginBottom": "20px",
                                    "padding": "10px",
                                    "backgroundColor": "#e8f5e9",
                                    "borderRadius": "5px",
                                },
                            ),
                            # Step 7: Save Results
                            html.Div(
                                [
                                    html.H4(
                                        "Step 7: Save Results",
                                        style={
                                            "color": "#f57c00",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Filename:"),
                                            dcc.Input(
                                                id="mi-features-filename",
                                                type="text",
                                                value=f'mi_features_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                                                style={
                                                    "width": "100%",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            html.Button(
                                                "Download MI Features",
                                                id="save-mi-features-btn",
                                                n_clicks=0,
                                                style={
                                                    "backgroundColor": "#f57c00",
                                                    "color": "white",
                                                    "padding": "10px 20px",
                                                    "fontSize": "14px",
                                                },
                                            ),
                                            html.Div(
                                                id="save-mi-features-status",
                                                style={
                                                    "marginTop": "10px",
                                                    "fontSize": "12px",
                                                    "color": "blue",
                                                },
                                            ),
                                            dcc.Download(id="download-mi-features"),
                                        ]
                                    ),
                                ],
                                style={
                                    "padding": "10px",
                                    "backgroundColor": "#fff3e0",
                                    "borderRadius": "5px",
                                },
                            ),
                        ],
                        style={
                            "width": "30%",
                            "paddingRight": "20px",
                            "verticalAlign": "top",
                        },
                    ),
                    # Right column - Visualizations and Results
                    html.Div(
                        [
                            # UMAP visualization
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="mi-umap-graph",
                                        config={"displayModeBar": True},
                                        style={"height": "550px", "width": "100%"},
                                    )
                                ],
                                style={"marginBottom": "20px"},
                            ),
                            # Quality metrics container (this was missing!)
                            html.Div(
                                id="umap-quality-metrics-mi",
                                children=[],
                                style={
                                    "marginTop": "10px",
                                    "padding": "10px",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "5px",
                                    "backgroundColor": "#f9f9f9",
                                    "marginBottom": "20px",
                                },
                            ),
                            # Feature Analysis Section
                            html.Div(
                                [
                                    # Feature scatter plot section
                                    html.Div(
                                        [
                                            html.H4(
                                                "Feature Scatter Plot",
                                                style={"color": "#388e3c"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "X-axis feature:",
                                                                style={
                                                                    "marginRight": "10px"
                                                                },
                                                            ),
                                                            dcc.Dropdown(
                                                                id="mi-scatter-x-feature",
                                                                options=[],
                                                                value=None,
                                                                style={
                                                                    "width": "300px"
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "inline-block",
                                                            "marginRight": "20px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Y-axis feature:",
                                                                style={
                                                                    "marginRight": "10px"
                                                                },
                                                            ),
                                                            dcc.Dropdown(
                                                                id="mi-scatter-y-feature",
                                                                options=[],
                                                                value=None,
                                                                style={
                                                                    "width": "300px"
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "inline-block"
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "10px"},
                                            ),
                                            html.Button(
                                                "Update Scatter Plot",
                                                id="update-mi-scatter-btn",
                                                n_clicks=0,
                                                style={
                                                    "backgroundColor": "#388e3c",
                                                    "color": "white",
                                                    "padding": "8px 16px",
                                                    "fontSize": "14px",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            dcc.Graph(
                                                id="mi-feature-scatter",
                                                figure={},
                                                style={"height": "400px"},
                                            ),
                                        ],
                                        style={"marginBottom": "30px"},
                                    ),
                                    # Feature Importance Table Section
                                    html.Div(
                                        [
                                            html.H4(
                                                "Feature Importance Table",
                                                style={
                                                    "color": "#388e3c",
                                                    "marginBottom": "15px",
                                                },
                                            ),
                                            # Search and sort controls
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            dcc.Input(
                                                                id="mi-feature-search-input",
                                                                type="text",
                                                                placeholder="Search features...",
                                                                style={
                                                                    "width": "300px",
                                                                    "marginRight": "10px",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "Search",
                                                                id="mi-feature-search-button",
                                                                n_clicks=0,
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "inline-block",
                                                            "marginRight": "30px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Sort by: ",
                                                                style={
                                                                    "marginRight": "10px"
                                                                },
                                                            ),
                                                            dcc.RadioItems(
                                                                id="mi-feature-sort-option",
                                                                options=[
                                                                    {
                                                                        "label": "MI Score",
                                                                        "value": "mi",
                                                                    },
                                                                    {
                                                                        "label": "Selection Order",
                                                                        "value": "order",
                                                                    },
                                                                    {
                                                                        "label": "UMAP MI",
                                                                        "value": "umap_mi",
                                                                    },
                                                                    {
                                                                        "label": "UMAP Corr",
                                                                        "value": "umap_corr",
                                                                    },
                                                                ],
                                                                value="mi",
                                                                inline=True,
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "inline-block"
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "15px"},
                                            ),
                                            # Feature importance table
                                            dash_table.DataTable(
                                                id="mi-feature-importance-table",
                                                columns=[
                                                    {
                                                        "name": "Rank",
                                                        "id": "Rank",
                                                        "type": "numeric",
                                                    },
                                                    {
                                                        "name": "Feature",
                                                        "id": "Feature",
                                                        "type": "text",
                                                    },
                                                    {
                                                        "name": "MI Score",
                                                        "id": "MI Score",
                                                        "type": "text",
                                                    },
                                                    {
                                                        "name": "Selected",
                                                        "id": "Selected",
                                                        "type": "text",
                                                        "presentation": "markdown",
                                                    },  # For checkmark
                                                    {
                                                        "name": "UMAP MI",
                                                        "id": "UMAP MI",
                                                        "type": "text",
                                                    },  # Will appear when UMAP is run
                                                    {
                                                        "name": "UMAP Corr",
                                                        "id": "UMAP Corr",
                                                        "type": "text",
                                                    },  # Will appear when UMAP is run
                                                ],
                                                data=[],
                                                style_cell={
                                                    "textAlign": "left",
                                                    "fontSize": "12px",
                                                },
                                                style_data_conditional=[
                                                    {
                                                        "if": {"column_id": "Selected"},
                                                        "textAlign": "center",
                                                        "fontWeight": "bold",
                                                        "color": "#2e7d32",
                                                    },
                                                    {
                                                        "if": {
                                                            "filter_query": '{Selected} = "✓"'
                                                        },
                                                        "backgroundColor": "#e8f5e9",
                                                    },
                                                ],
                                                style_header={
                                                    "backgroundColor": "#f5f5f5",
                                                    "fontWeight": "bold",
                                                    "fontSize": "13px",
                                                },
                                                page_size=15,
                                                style_table={
                                                    "maxHeight": "400px",
                                                    "overflowY": "auto",
                                                },
                                                sort_action="native",
                                                filter_action="native",
                                            ),
                                        ],
                                        style={
                                            "marginTop": "20px",
                                            "padding": "15px",
                                            "backgroundColor": "#f5f5f5",
                                            "borderRadius": "5px",
                                        },
                                    ),
                                    # Summary info
                                    html.Div(
                                        id="mi-sorted-features-info",
                                        children=[],
                                        style={
                                            "marginTop": "20px",
                                            "padding": "10px",
                                            "backgroundColor": "#e8f5e9",
                                            "borderRadius": "5px",
                                        },
                                    ),
                                ],
                                style={"marginTop": "20px"},
                            ),
                        ],
                        style={"width": "70%", "verticalAlign": "top"},
                    ),
                ],
                style={"display": "flex", "alignItems": "flex-start"},
            ),
        ],
        className="container",
    )
