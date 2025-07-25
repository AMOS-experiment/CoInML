from dash import dcc, html


def create_advanced_analysis_tab():
    return [
        dcc.Tabs(
            id="advanced-sub-tabs",
            value="rerun-umap-tab",
            children=[
                # Re-run UMAP Tab
                dcc.Tab(
                    label="Re-run UMAP",
                    value="rerun-umap-tab",
                    children=[
                        html.Div(
                            [
                                # Graph 3: Re-run UMAP on Selected Points
                                html.Div(
                                    [
                                        html.H3(
                                            "Graph 3: Re-run UMAP on Selected Points from Graph 1",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.H4(
                                                            "Select Features for Re-run UMAP:"
                                                        ),
                                                        html.Div(
                                                            id="feature-selection-ui-graph3",
                                                            children=[
                                                                html.Div(
                                                                    "Upload files to see available features",
                                                                    style={
                                                                        "color": "gray"
                                                                    },
                                                                )
                                                            ],
                                                            className="feature-checklist",
                                                        ),
                                                        html.Br(),
                                                        html.Label(
                                                            "UMAP Neighbors (Selected Re-run):"
                                                        ),
                                                        dcc.Input(
                                                            id="num-neighbors-selected-run",
                                                            type="number",
                                                            value=15,
                                                            min=1,
                                                        ),
                                                        html.Br(),
                                                        html.Label(
                                                            "Min Dist (Selected Re-run):"
                                                        ),
                                                        dcc.Input(
                                                            id="min-dist-selected-run",
                                                            type="number",
                                                            value=0.1,
                                                            step=0.01,
                                                            min=0,
                                                        ),
                                                        html.Br(),
                                                        html.Button(
                                                            "Re-run UMAP on Selected Points",
                                                            id="run-umap-selected-run",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                        ),
                                                        html.Div(
                                                            id="run-umap-selected-run-status",
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
                                                            id="metric-selector-graph3",
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
                                                            labelStyle={
                                                                "display": "block"
                                                            },
                                                        ),
                                                        html.Hr(
                                                            style={
                                                                "marginTop": "20px",
                                                                "marginBottom": "20px",
                                                            }
                                                        ),
                                                        html.Label(
                                                            "Select points in Graph 3 and save:"
                                                        ),
                                                        html.Button(
                                                            "Show Graph 3 Selection",
                                                            id="show-selected-run",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                        ),
                                                        html.Div(
                                                            id="show-selected-run-status",
                                                            style={
                                                                "marginTop": "5px",
                                                                "fontSize": "12px",
                                                                "color": "blue",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="selected-points-run-info",
                                                            style={
                                                                "marginTop": "10px",
                                                                "fontSize": "12px",
                                                            },
                                                        ),
                                                        html.Hr(
                                                            style={
                                                                "marginTop": "15px",
                                                                "marginBottom": "15px",
                                                            }
                                                        ),
                                                        html.Label(
                                                            "Save Graph 3 selected points:"
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.Input(
                                                                    id="selection-run-filename",
                                                                    type="text",
                                                                    placeholder="Enter filename (without extension)",
                                                                    style={
                                                                        "width": "100%",
                                                                        "marginBottom": "10px",
                                                                    },
                                                                ),
                                                                html.Button(
                                                                    "Save Graph 3 Selection",
                                                                    id="save-selection-run-btn",
                                                                    n_clicks=0,
                                                                    className="btn-secondary",
                                                                ),
                                                                html.Div(
                                                                    id="save-selection-run-status",
                                                                    style={
                                                                        "marginTop": "5px",
                                                                        "fontSize": "12px",
                                                                        "color": "blue",
                                                                    },
                                                                ),
                                                                dcc.Download(
                                                                    id="download-selection-run"
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
                                                            id="umap-graph-selected-run",
                                                            config={
                                                                "displayModeBar": True
                                                            },
                                                            style={"height": "600px"},
                                                        ),
                                                        html.Div(
                                                            id="debug-output-selected-run",
                                                            style={
                                                                "marginTop": "10px",
                                                                "fontSize": "12px",
                                                                "color": "gray",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="umap-quality-metrics-graph3",
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
                                ),
                                # Graph 3 Selected Points Visualization
                                html.Div(
                                    [
                                        html.H3(
                                            "Graph 3 Selected Points Visualization",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "This shows points selected from Graph 3."
                                                        ),
                                                        html.Div(
                                                            id="graph3-selection-info-viz",
                                                            style={
                                                                "marginTop": "15px",
                                                                "fontSize": "12px",
                                                            },
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
                                                            id="umap-graph-selected-run-only",
                                                            config={
                                                                "displayModeBar": True
                                                            },
                                                            style={"height": "600px"},
                                                        ),
                                                        html.Div(
                                                            id="debug-output-selected-run-only",
                                                            style={
                                                                "marginTop": "10px",
                                                                "fontSize": "12px",
                                                                "color": "gray",
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
                                ),
                                # UMAP on Graph 3 Selected Points
                                html.Div(
                                    [
                                        html.H3(
                                            "UMAP Re-run on Graph 3 Selected Points",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.H4(
                                                            "Select Features for Re-run UMAP on Graph 3 Selection:"
                                                        ),
                                                        html.Div(
                                                            id="feature-selection-ui-graph3-selection",
                                                            children=[
                                                                html.Div(
                                                                    "Upload files to see available features",
                                                                    style={
                                                                        "color": "gray"
                                                                    },
                                                                )
                                                            ],
                                                            className="feature-checklist",
                                                        ),
                                                        html.Br(),
                                                        html.Label("UMAP Neighbors:"),
                                                        dcc.Input(
                                                            id="num-neighbors-graph3-selection",
                                                            type="number",
                                                            value=15,
                                                            min=1,
                                                        ),
                                                        html.Br(),
                                                        html.Label("Min Dist:"),
                                                        dcc.Input(
                                                            id="min-dist-graph3-selection",
                                                            type="number",
                                                            value=0.1,
                                                            step=0.01,
                                                            min=0,
                                                        ),
                                                        html.Br(),
                                                        html.Button(
                                                            "Run UMAP on Graph 3 Selection",
                                                            id="run-umap-graph3-selection",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                        ),
                                                        html.Div(
                                                            id="run-umap-graph3-selection-status",
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
                                                            id="metric-selector-graph3-selection",
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
                                                            labelStyle={
                                                                "display": "block"
                                                            },
                                                        ),
                                                        html.Hr(
                                                            style={
                                                                "marginTop": "20px",
                                                                "marginBottom": "20px",
                                                            }
                                                        ),
                                                        html.Label("Save selection:"),
                                                        html.Div(
                                                            [
                                                                dcc.Input(
                                                                    id="selection-graph3-selection-filename",
                                                                    type="text",
                                                                    placeholder="Enter filename (without extension)",
                                                                    style={
                                                                        "width": "100%",
                                                                        "marginBottom": "10px",
                                                                    },
                                                                ),
                                                                html.Button(
                                                                    "Save Selection",
                                                                    id="save-selection-graph3-selection-btn",
                                                                    n_clicks=0,
                                                                    className="btn-secondary",
                                                                ),
                                                                html.Div(
                                                                    id="save-selection-graph3-selection-status",
                                                                    style={
                                                                        "marginTop": "5px",
                                                                        "fontSize": "12px",
                                                                        "color": "blue",
                                                                    },
                                                                ),
                                                                dcc.Download(
                                                                    id="download-selection-graph3-selection"
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
                                                            id="umap-graph-graph3-selection",
                                                            config={
                                                                "displayModeBar": True
                                                            },
                                                            style={"height": "600px"},
                                                        ),
                                                        html.Div(
                                                            id="debug-output-graph3-selection",
                                                            style={
                                                                "marginTop": "10px",
                                                                "fontSize": "12px",
                                                                "color": "gray",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="umap-quality-metrics-graph3-selection",
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
                                ),
                            ]
                        )
                    ],
                ),
                # Custom Feature Plot Tab
                dcc.Tab(
                    label="Custom Feature Analysis",
                    value="custom-feature-tab",
                    children=[
                        html.Div(
                            [
                                html.H3(
                                    "Custom Feature Scatter Plot",
                                    style={"textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H4("Select Features to Plot:"),
                                                html.Div(
                                                    [
                                                        html.Label("X-Axis Feature:"),
                                                        dcc.Dropdown(
                                                            id="x-axis-feature",
                                                            options=[],
                                                            value=None,
                                                            placeholder="Select X-Axis Feature",
                                                        ),
                                                        html.Br(),
                                                        html.Label("Y-Axis Feature:"),
                                                        dcc.Dropdown(
                                                            id="y-axis-feature",
                                                            options=[],
                                                            value=None,
                                                            placeholder="Select Y-Axis Feature",
                                                        ),
                                                        html.Br(),
                                                        html.Label(
                                                            "Display Selection From:"
                                                        ),
                                                        dcc.RadioItems(
                                                            id="selection-source",
                                                            options=[
                                                                {
                                                                    "label": "Graph 2 (Selection from Graph 1)",
                                                                    "value": "graph2",
                                                                },
                                                                {
                                                                    "label": "Graph 3 Selection",
                                                                    "value": "graph3",
                                                                },
                                                                {
                                                                    "label": "Both Selections",
                                                                    "value": "both",
                                                                },
                                                            ],
                                                            value="graph2",
                                                            labelStyle={
                                                                "display": "block",
                                                                "marginBottom": "5px",
                                                            },
                                                        ),
                                                        html.Br(),
                                                        html.Button(
                                                            "Plot Selected Features",
                                                            id="plot-custom-features",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                        ),
                                                        html.Div(
                                                            id="plot-custom-features-status",
                                                            style={
                                                                "marginTop": "5px",
                                                                "fontSize": "12px",
                                                                "color": "blue",
                                                            },
                                                        ),
                                                    ],
                                                    className="feature-checklist",
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
                                                    id="custom-feature-plot",
                                                    config={"displayModeBar": True},
                                                    style={"height": "600px"},
                                                ),
                                                html.Div(
                                                    id="debug-output-custom-plot",
                                                    style={
                                                        "marginTop": "10px",
                                                        "fontSize": "12px",
                                                        "color": "gray",
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
