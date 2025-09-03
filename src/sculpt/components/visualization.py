from dash import dcc, html


def create_basic_viz_tab():
    return [
        dcc.Tabs(
            id="basic-sub-tabs",
            value="umap-tab",
            children=[
                # UMAP Tab
                dcc.Tab(
                    label="UMAP Embedding",
                    value="umap-tab",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "UMAP Embedding Analysis",
                                    style={"textAlign": "center"},
                                ),
                                # Main container with flex layout
                                html.Div(
                                    [
                                        # Left panel with controls
                                        html.Div(
                                            [
                                                html.H4("Select Files for UMAP:"),
                                                dcc.Checklist(
                                                    id="umap-file-selector",
                                                    options=[],
                                                    value=[],
                                                    labelStyle={"display": "block"},
                                                ),
                                                html.Br(),
                                                html.H4("Select Features for UMAP:"),
                                                html.Div(
                                                    id="feature-selection-ui-graph1",
                                                    children=[
                                                        html.Div(
                                                            "Upload files to see available features",
                                                            style={"color": "gray"},
                                                        )
                                                    ],
                                                    className="feature-checklist",
                                                ),
                                                html.Br(),
                                                # UMAP Parameters
                                                html.Div(
                                                    [
                                                        html.H5("UMAP Parameters"),
                                                        html.Label(
                                                            "Number of Neighbors:"
                                                        ),
                                                        dcc.Input(
                                                            id="num-neighbors",
                                                            type="number",
                                                            value=15,
                                                            min=1,
                                                        ),
                                                        html.Br(),
                                                        html.Label("Min Dist:"),
                                                        dcc.Input(
                                                            id="min-dist",
                                                            type="number",
                                                            value=0.1,
                                                            step=0.01,
                                                            min=0,
                                                        ),
                                                        html.Br(),
                                                        html.Label(
                                                            "Sample Fraction (0-1):"
                                                        ),
                                                        dcc.Input(
                                                            id="sample-frac",
                                                            type="number",
                                                            value=0.01,
                                                            min=0.001,
                                                            max=1,
                                                            step=0.001,
                                                        ),
                                                    ],
                                                    style={
                                                        "backgroundColor": "#f5f5f5",
                                                        "padding": "10px",
                                                        "borderRadius": "5px",
                                                        "marginBottom": "15px",
                                                    },
                                                ),
                                                # Visualization Options
                                                html.Div(
                                                    [
                                                        html.H5(
                                                            "Visualization Options"
                                                        ),
                                                        html.Label(
                                                            "Visualization Type:"
                                                        ),
                                                        dcc.RadioItems(
                                                            id="visualization-type",
                                                            options=[
                                                                {
                                                                    "label": "Scatter Plot",
                                                                    "value": "scatter",
                                                                },
                                                                {
                                                                    "label": "Heatmap",
                                                                    "value": "heatmap",
                                                                },
                                                            ],
                                                            value="scatter",
                                                            labelStyle={
                                                                "display": "inline-block",
                                                                "marginRight": "10px",
                                                            },
                                                        ),
                                                        html.Br(),
                                                        html.Div(
                                                            id="scatter-settings",
                                                            children=[
                                                                html.Label(
                                                                    "Point Opacity:"
                                                                ),
                                                                dcc.Slider(
                                                                    id="point-opacity",
                                                                    min=0.05,
                                                                    max=1.0,
                                                                    step=0.05,
                                                                    value=0.3,
                                                                    marks={
                                                                        i
                                                                        / 10: str(
                                                                            i / 10
                                                                        )
                                                                        for i in range(
                                                                            1, 11, 2
                                                                        )
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            id="heatmap-settings",
                                                            children=[
                                                                html.Label(
                                                                    "Heatmap Bandwidth:"
                                                                ),
                                                                dcc.Slider(
                                                                    id="heatmap-bandwidth",
                                                                    min=0.05,
                                                                    max=1.0,
                                                                    step=0.05,
                                                                    value=0.2,
                                                                    marks={
                                                                        i
                                                                        / 10: str(
                                                                            i / 10
                                                                        )
                                                                        for i in range(
                                                                            1, 11, 2
                                                                        )
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Label(
                                                                    "Heatmap Color Scale:"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="heatmap-colorscale",
                                                                    options=[
                                                                        {
                                                                            "label": "Viridis",
                                                                            "value": "Viridis",
                                                                        },
                                                                        {
                                                                            "label": "Plasma",
                                                                            "value": "Plasma",
                                                                        },
                                                                        {
                                                                            "label": "Inferno",
                                                                            "value": "Inferno",
                                                                        },
                                                                        {
                                                                            "label": "Magma",
                                                                            "value": "Magma",
                                                                        },
                                                                        {
                                                                            "label": "Cividis",
                                                                            "value": "Cividis",
                                                                        },
                                                                        {
                                                                            "label": "Turbo",
                                                                            "value": "Turbo",
                                                                        },
                                                                        {
                                                                            "label": "Hot",
                                                                            "value": "Hot",
                                                                        },
                                                                        {
                                                                            "label": "Jet",
                                                                            "value": "Jet",
                                                                        },
                                                                    ],
                                                                    value="Viridis",
                                                                    clearable=False,
                                                                ),
                                                                html.Br(),
                                                                html.Label(
                                                                    "Show Points Overlay:"
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="show-points-overlay",
                                                                    options=[
                                                                        {
                                                                            "label": "Yes",
                                                                            "value": "yes",
                                                                        },
                                                                        {
                                                                            "label": "No",
                                                                            "value": "no",
                                                                        },
                                                                    ],
                                                                    value="yes",
                                                                    labelStyle={
                                                                        "display": "inline-block",
                                                                        "marginRight": "10px",
                                                                    },
                                                                ),
                                                            ],
                                                            style={"display": "none"},
                                                        ),
                                                    ],
                                                    style={
                                                        "backgroundColor": "#f5f5f5",
                                                        "padding": "10px",
                                                        "borderRadius": "5px",
                                                        "marginBottom": "15px",
                                                    },
                                                ),
                                                html.Label("Color by:"),
                                                dcc.RadioItems(
                                                    id="color-mode",
                                                    options=[
                                                        {
                                                            "label": "File Source",
                                                            "value": "file",
                                                        },
                                                        {
                                                            "label": "DBSCAN Clusters",
                                                            "value": "cluster",
                                                        },
                                                    ],
                                                    value="file",
                                                    labelStyle={
                                                        "display": "inline-block",
                                                        "marginRight": "10px",
                                                    },
                                                ),
                                                html.Br(),
                                                html.Button(
                                                    "Run UMAP",
                                                    id="run-umap",
                                                    n_clicks=0,
                                                ),
                                                html.Div(
                                                    id="run-umap-status",
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
                                                    id="metric-selector",
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
                                                        {
                                                            "label": "Physics Consistency",
                                                            "value": "physics_consistency",
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
                                                "width": "25%",
                                                "paddingRight": "20px",
                                            },
                                        ),
                                        # Right panel with graph
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="umap-graph",
                                                    config={
                                                        "displayModeBar": True,
                                                        "displaylogo": False,
                                                        "modeBarButtonsToRemove": ["pan2d", "autoScale2d"],
                                                        "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                                                        "toImageButtonOptions": {
                                                            "format": "png",
                                                            "filename": "plot_export",
                                                            "height": 600,
                                                            "width": 800,
                                                            "scale": 2
                                                        },
                                                        "doubleClick": "reset+autosize"
                                                    },
                                                    style={"height": "600px"},
                                                ),
                                                html.Div(
                                                    id="debug-output",
                                                    style={
                                                        "marginTop": "10px",
                                                        "fontSize": "12px",
                                                        "color": "gray",
                                                    },
                                                ),
                                                html.Div(
                                                    id="umap-quality-metrics",
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
                # Custom Scatter Tab
                dcc.Tab(
                    label="Custom Feature Plot",
                    value="custom-scatter-tab",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Enhanced Custom Scatter Plot",
                                    style={"textAlign": "center"},
                                ),
                                # Main container with flex layout
                                html.Div(
                                    [
                                        # Left panel
                                        html.Div(
                                            [
                                                html.H4("Select Files:"),
                                                dcc.Checklist(
                                                    id="file-selector-graph15",
                                                    options=[],
                                                    value=[],
                                                    labelStyle={"display": "block"},
                                                ),
                                                html.Br(),
                                                html.H4("Select Features to Plot:"),
                                                html.Div(
                                                    [
                                                        html.Label("X-Axis Feature:"),
                                                        dcc.Dropdown(
                                                            id="x-axis-feature-graph15",
                                                            options=[],
                                                            value=None,
                                                            placeholder="Select X-Axis Feature",
                                                        ),
                                                        html.Br(),
                                                        html.Label("Y-Axis Feature:"),
                                                        dcc.Dropdown(
                                                            id="y-axis-feature-graph15",
                                                            options=[],
                                                            value=None,
                                                            placeholder="Select Y-Axis Feature",
                                                        ),
                                                    ]
                                                ),
                                                html.Br(),
                                                html.Label("Sample Fraction (0-1):"),
                                                dcc.Input(
                                                    id="sample-frac-graph15",
                                                    type="number",
                                                    value=0.1,
                                                    min=0.001,
                                                    max=1,
                                                    step=0.001,
                                                ),
                                                html.Br(),
                                                html.Br(),
                                                html.Label("Visualization Type:"),
                                                dcc.RadioItems(
                                                    id="visualization-type-graph15",
                                                    options=[
                                                        {
                                                            "label": "Scatter Plot",
                                                            "value": "scatter",
                                                        },
                                                        {
                                                            "label": "Heatmap",
                                                            "value": "heatmap",
                                                        },
                                                    ],
                                                    value="scatter",
                                                    labelStyle={
                                                        "display": "inline-block",
                                                        "marginRight": "10px",
                                                    },
                                                ),
                                                html.Br(),
                                                html.Div(
                                                    id="scatter-settings-graph15",
                                                    children=[
                                                        html.Label("Point Opacity:"),
                                                        dcc.Slider(
                                                            id="point-opacity-graph15",
                                                            min=0.05,
                                                            max=1.0,
                                                            step=0.05,
                                                            value=0.3,
                                                            marks={
                                                                i / 10: str(i / 10)
                                                                for i in range(1, 11, 2)
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="heatmap-settings-graph15",
                                                    children=[
                                                        html.Label(
                                                            "Heatmap Bandwidth:"
                                                        ),
                                                        dcc.Slider(
                                                            id="heatmap-bandwidth-graph15",
                                                            min=0.05,
                                                            max=1.0,
                                                            step=0.05,
                                                            value=0.2,
                                                            marks={
                                                                i / 10: str(i / 10)
                                                                for i in range(1, 11, 2)
                                                            },
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                        html.Br(),
                                                        html.Label(
                                                            "Heatmap Color Scale:"
                                                        ),
                                                        dcc.Dropdown(
                                                            id="heatmap-colorscale-graph15",
                                                            options=[
                                                                {
                                                                    "label": "Viridis",
                                                                    "value": "Viridis",
                                                                },
                                                                {
                                                                    "label": "Plasma",
                                                                    "value": "Plasma",
                                                                },
                                                                {
                                                                    "label": "Inferno",
                                                                    "value": "Inferno",
                                                                },
                                                                {
                                                                    "label": "Magma",
                                                                    "value": "Magma",
                                                                },
                                                                {
                                                                    "label": "Cividis",
                                                                    "value": "Cividis",
                                                                },
                                                                {
                                                                    "label": "Turbo",
                                                                    "value": "Turbo",
                                                                },
                                                                {
                                                                    "label": "Hot",
                                                                    "value": "Hot",
                                                                },
                                                                {
                                                                    "label": "Jet",
                                                                    "value": "Jet",
                                                                },
                                                            ],
                                                            value="Viridis",
                                                            clearable=False,
                                                        ),
                                                        html.Br(),
                                                        html.Label(
                                                            "Show Points Overlay:"
                                                        ),
                                                        dcc.RadioItems(
                                                            id="show-points-overlay-graph15",
                                                            options=[
                                                                {
                                                                    "label": "Yes",
                                                                    "value": "yes",
                                                                },
                                                                {
                                                                    "label": "No",
                                                                    "value": "no",
                                                                },
                                                            ],
                                                            value="yes",
                                                            labelStyle={
                                                                "display": "inline-block",
                                                                "marginRight": "10px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                                html.Br(),
                                                html.Label("Color by:"),
                                                dcc.RadioItems(
                                                    id="color-mode-graph15",
                                                    options=[
                                                        {
                                                            "label": "File Source",
                                                            "value": "file",
                                                        },
                                                        {
                                                            "label": "DBSCAN Clusters",
                                                            "value": "cluster",
                                                        },
                                                    ],
                                                    value="file",
                                                    labelStyle={
                                                        "display": "inline-block",
                                                        "marginRight": "10px",
                                                    },
                                                ),
                                                html.Br(),
                                                html.Button(
                                                    "Generate Plot",
                                                    id="generate-plot-graph15",
                                                    n_clicks=0,
                                                ),
                                                html.Div(
                                                    id="generate-plot-graph15-status",
                                                    style={
                                                        "marginTop": "5px",
                                                        "fontSize": "12px",
                                                        "color": "blue",
                                                    },
                                                ),
                                                html.Br(),
                                                # Save selection components
                                                html.Hr(
                                                    style={
                                                        "marginTop": "20px",
                                                        "marginBottom": "20px",
                                                    }
                                                ),
                                                html.Label(
                                                    "Save selected points to file:"
                                                ),
                                                html.Div(
                                                    [
                                                        dcc.Input(
                                                            id="selection-filename-graph15",
                                                            type="text",
                                                            placeholder="Enter filename (without extension)",
                                                            style={
                                                                "width": "100%",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        html.Button(
                                                            "Save Selection",
                                                            id="save-selection-graph15-btn",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                        ),
                                                        html.Div(
                                                            id="save-selection-graph15-status",
                                                            style={
                                                                "marginTop": "5px",
                                                                "fontSize": "12px",
                                                                "color": "blue",
                                                            },
                                                        ),
                                                        dcc.Download(
                                                            id="download-selection-graph15"
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            style={
                                                "width": "25%",
                                                "paddingRight": "20px",
                                            },
                                        ),
                                        # Right panel
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="scatter-graph15",
                                                    config={
                                                        "displayModeBar": True,
                                                        "displaylogo": False,
                                                        "modeBarButtonsToRemove": ["pan2d", "autoScale2d"],
                                                        "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                                                        "toImageButtonOptions": {
                                                            "format": "png",
                                                            "filename": "plot_export",
                                                            "height": 600,
                                                            "width": 800,
                                                            "scale": 2
                                                        },
                                                        "doubleClick": "reset+autosize"
                                                    },
                                                    style={"height": "600px"},
                                                ),
                                                html.Div(
                                                    id="debug-output-graph15",
                                                    style={
                                                        "marginTop": "10px",
                                                        "fontSize": "12px",
                                                        "color": "gray",
                                                    },
                                                ),
                                                html.Div(
                                                    id="quality-metrics-graph15",
                                                    children=[],
                                                    style={
                                                        "marginTop": "10px",
                                                        "padding": "10px",
                                                        "border": "1px solid #ddd",
                                                        "borderRadius": "5px",
                                                        "backgroundColor": "#f9f9f9",
                                                    },
                                                ),
                                                html.Div(
                                                    id="selected-points-info-graph15",
                                                    style={
                                                        "marginTop": "15px",
                                                        "fontSize": "12px",
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
