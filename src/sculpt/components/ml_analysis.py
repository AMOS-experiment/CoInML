from dash import dcc, html


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
                        html.Div(
                            [
                                html.H3(
                                    "Deep Autoencoder with UMAP on Latent Space",
                                    style={"textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H4(
                                                    "Select Features for Autoencoder:"
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
                                                ),
                                                html.Br(),
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
                                                html.Br(),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Select Data Source:"
                                                        ),
                                                        dcc.RadioItems(
                                                            id="autoencoder-data-source",
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
                                                html.Hr(
                                                    style={
                                                        "marginTop": "20px",
                                                        "marginBottom": "20px",
                                                    }
                                                ),
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
                                                        html.Label(
                                                            "Select Metrics to Calculate:"
                                                        ),
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
                                                            labelStyle={
                                                                "display": "block"
                                                            },
                                                        ),
                                                    ]
                                                ),
                                                html.Hr(
                                                    style={
                                                        "marginTop": "20px",
                                                        "marginBottom": "20px",
                                                    }
                                                ),
                                                html.Label("Save Latent Features:"),
                                                html.Div(
                                                    [
                                                        dcc.Input(
                                                            id="latent-features-filename",
                                                            type="text",
                                                            placeholder="Enter filename (without extension)",
                                                            style={
                                                                "width": "100%",
                                                                "marginBottom": "10px",
                                                            },
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
                                                        dcc.Download(
                                                            id="download-latent-features"
                                                        ),
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
                                                    id="autoencoder-umap-graph",
                                                    config={"displayModeBar": True},
                                                    style={"height": "600px"},
                                                ),
                                                html.Div(
                                                    id="autoencoder-debug-output",
                                                    style={
                                                        "marginTop": "10px",
                                                        "fontSize": "12px",
                                                        "color": "gray",
                                                    },
                                                ),
                                                html.Div(
                                                    id="umap-quality-metrics-autoencoder",
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
                                        html.Div(
                                            id="feature-importance-container",
                                            children=[],
                                            style={"marginTop": "15px"},
                                        ),
                                    ],
                                    style={"display": "flex"},
                                ),
                            ],
                            className="container",
                        )
                    ],
                ),
                # Genetic Features Tab
                dcc.Tab(
                    label="Genetic Feature Engineering",
                    value="genetic-tab",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Genetic Feature Engineering",
                                    style={"textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H4(
                                                    "Select Features for Genetic Programming:"
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
                                                html.Br(),
                                                html.H4("Clustering Parameters:"),
                                                html.Label("Clustering Method:"),
                                                dcc.Dropdown(
                                                    id="clustering-method",
                                                    options=[
                                                        {
                                                            "label": "DBSCAN",
                                                            "value": "dbscan",
                                                        },
                                                        {
                                                            "label": "KMeans",
                                                            "value": "kmeans",
                                                        },
                                                        {
                                                            "label": "Agglomerative",
                                                            "value": "agglomerative",
                                                        },
                                                    ],
                                                    value="dbscan",
                                                ),
                                                html.Div(
                                                    id="dbscan-params",
                                                    children=[
                                                        html.Label("DBSCAN Epsilon:"),
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
                                                            "DBSCAN Min Samples:"
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
                                                        html.Label("K-Means Clusters:"),
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
                                                        html.Label(
                                                            "Agglomerative Clusters:"
                                                        ),
                                                        dcc.Input(
                                                            id="agglomerative-n-clusters",
                                                            type="number",
                                                            value=5,
                                                            min=2,
                                                            max=20,
                                                        ),
                                                        html.Br(),
                                                        html.Label("Linkage:"),
                                                        dcc.Dropdown(
                                                            id="agglomerative-linkage",
                                                            options=[
                                                                {
                                                                    "label": "Ward",
                                                                    "value": "ward",
                                                                },
                                                                {
                                                                    "label": "Complete",
                                                                    "value": "complete",
                                                                },
                                                                {
                                                                    "label": "Average",
                                                                    "value": "average",
                                                                },
                                                                {
                                                                    "label": "Single",
                                                                    "value": "single",
                                                                },
                                                            ],
                                                            value="ward",
                                                        ),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                                html.Br(),
                                                html.H4(
                                                    "Genetic Programming Parameters:"
                                                ),
                                                html.Label("Number of Generations:"),
                                                dcc.Input(
                                                    id="gp-generations",
                                                    type="number",
                                                    value=20,
                                                    min=5,
                                                    max=100,
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
                                                html.Label(
                                                    "Number of Features to Generate:"
                                                ),
                                                dcc.Input(
                                                    id="gp-n-components",
                                                    type="number",
                                                    value=10,
                                                    min=2,
                                                    max=20,
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
                                                        {
                                                            "label": "Square Root & Power",
                                                            "value": "sqrt_pow",
                                                        },
                                                        {
                                                            "label": "Special (abs, inv)",
                                                            "value": "special",
                                                        },
                                                    ],
                                                    value=[
                                                        "basic",
                                                        "trig",
                                                        "sqrt_pow",
                                                        "special",
                                                    ],
                                                ),
                                                html.Br(),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Select Data Source:"
                                                        ),
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
                                                html.Div(
                                                    [
                                                        html.H4(
                                                            "Visualize Genetic Features:",
                                                            style={
                                                                "marginTop": "15px",
                                                                "color": "#2e7d32",
                                                            },
                                                        ),
                                                        html.Div(
                                                            "First run genetic discovery above, then select specific features"
                                                            " to visualize with UMAP:",
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
                                                        html.Label(
                                                            "Select Metrics to Calculate:"
                                                        ),
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
                                                            labelStyle={
                                                                "display": "block"
                                                            },
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
                                                        dcc.Download(
                                                            id="download-genetic-features"
                                                        ),
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
                            ],
                            className="container",
                        )
                    ],
                ),
                # Mutual Information Tab
                dcc.Tab(
                    label="Mutual Information Feature Selection",
                    value="mi-tab",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Mutual Information Feature Selection",
                                    style={"textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H4(
                                                    "Select Features for MI Analysis:"
                                                ),
                                                html.Div(
                                                    id="feature-selection-ui-mi",
                                                    children=[
                                                        html.Div(
                                                            "Upload files to see available features",
                                                            style={"color": "gray"},
                                                        )
                                                    ],
                                                    className="feature-checklist",
                                                ),
                                                html.Br(),
                                                html.H4("MI Analysis Parameters:"),
                                                html.Label("Target Variables:"),
                                                dcc.Dropdown(
                                                    id="mi-target-variables",
                                                    options=[
                                                        {
                                                            "label": "KER (Kinetic Energy Release)",
                                                            "value": "KER",
                                                        },
                                                        {
                                                            "label": "EESum (Sum of Electron Energies)",
                                                            "value": "EESum",
                                                        },
                                                        {
                                                            "label": "Total Energy",
                                                            "value": "TotalEnergy",
                                                        },
                                                        {
                                                            "label": "Energy Ion 1",
                                                            "value": "energy_ion1",
                                                        },
                                                        {
                                                            "label": "Energy Ion 2",
                                                            "value": "energy_ion2",
                                                        },
                                                        {
                                                            "label": "Energy Electron 1",
                                                            "value": "energy_electron1",
                                                        },
                                                        {
                                                            "label": "Energy Electron 2",
                                                            "value": "energy_electron2",
                                                        },
                                                    ],
                                                    value=[
                                                        "KER",
                                                        "EESum",
                                                        "TotalEnergy",
                                                    ],
                                                    multi=True,
                                                ),
                                                html.Br(),
                                                html.Label(
                                                    "Redundancy Threshold (0-1):"
                                                ),
                                                html.Div(
                                                    [
                                                        "Low values reduce redundancy, high values allow more similar"
                                                        " features",
                                                        dcc.Slider(
                                                            id="mi-redundancy-threshold",
                                                            min=0.1,
                                                            max=0.9,
                                                            step=0.1,
                                                            value=0.5,
                                                            marks={
                                                                i / 10: f"{i/10}"
                                                                for i in range(1, 10)
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ]
                                                ),
                                                html.Br(),
                                                html.Label(
                                                    "Maximum Number of Features:"
                                                ),
                                                dcc.Input(
                                                    id="mi-max-features",
                                                    type="number",
                                                    value=20,
                                                    min=5,
                                                    max=100,
                                                ),
                                                html.Br(),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Select Data Source:"
                                                        ),
                                                        dcc.RadioItems(
                                                            id="mi-data-source",
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
                                                html.H4("Autoencoder Parameters:"),
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
                                                    value=64,
                                                    min=8,
                                                    max=512,
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
                                                html.Br(),
                                                html.Button(
                                                    "Run MI Feature Selection",
                                                    id="run-mi-features",
                                                    n_clicks=0,
                                                    className="btn-secondary",
                                                ),
                                                html.Div(
                                                    id="mi-features-status",
                                                    style={
                                                        "marginTop": "5px",
                                                        "fontSize": "12px",
                                                        "color": "blue",
                                                    },
                                                ),
                                                html.Br(),
                                                html.Div(
                                                    [
                                                        html.H4(
                                                            "Visualize MI Features:",
                                                            style={
                                                                "marginTop": "15px",
                                                                "color": "#2e7d32",
                                                            },
                                                        ),
                                                        html.Div(
                                                            "First run MI feature selection above, then visualize with UMAP:",
                                                            style={
                                                                "color": "#555",
                                                                "fontSize": "12px",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        html.Button(
                                                            "Run UMAP on MI Features",
                                                            id="run-umap-mi",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                            style={
                                                                "backgroundColor": "#2e7d32",
                                                                "marginTop": "5px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="run-umap-mi-status",
                                                            style={
                                                                "marginTop": "5px",
                                                                "fontSize": "12px",
                                                                "color": "blue",
                                                            },
                                                        ),
                                                        html.Br(),
                                                        html.Label(
                                                            "Select Metrics to Calculate:"
                                                        ),
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
                                                                    "value": "stability",
                                                                },
                                                            ],
                                                            value=[
                                                                "silhouette",
                                                                "davies_bouldin",
                                                                "calinski_harabasz",
                                                            ],
                                                            labelStyle={
                                                                "display": "block"
                                                            },
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
                                                html.Label("Save MI Features:"),
                                                html.Div(
                                                    [
                                                        dcc.Input(
                                                            id="mi-features-filename",
                                                            type="text",
                                                            placeholder="Enter filename (without extension)",
                                                            style={
                                                                "width": "100%",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        html.Button(
                                                            "Save MI Features",
                                                            id="save-mi-features-btn",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                        ),
                                                        html.Div(
                                                            id="save-mi-features-status",
                                                            style={
                                                                "marginTop": "5px",
                                                                "fontSize": "12px",
                                                                "color": "blue",
                                                            },
                                                        ),
                                                        dcc.Download(
                                                            id="download-mi-features"
                                                        ),
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
                                                    id="mi-features-graph",
                                                    config={"displayModeBar": True},
                                                    style={"height": "600px"},
                                                ),
                                                html.Div(
                                                    id="mi-features-debug-output",
                                                    style={
                                                        "marginTop": "10px",
                                                        "fontSize": "12px",
                                                        "color": "gray",
                                                    },
                                                ),
                                                html.Div(
                                                    id="umap-quality-metrics-mi",
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
                            ],
                            className="container",
                        )
                    ],
                ),
            ],
        )
    ]
