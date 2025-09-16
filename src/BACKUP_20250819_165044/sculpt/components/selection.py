from dash import dcc, html


def create_selection_tab():
    return [
        dcc.Tabs(
            id="selection-sub-tabs",
            value="selection-view-tab",
            children=[
                # Selection Viewing Tab
                dcc.Tab(
                    label="View Selections",
                    value="selection-view-tab",
                    children=[
                        html.Div(
                            [
                                # Graph 2: Selected Points from UMAP
                                html.Div(
                                    [
                                        html.H3(
                                            "Selected Points from UMAP",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Select points on UMAP visualization using the lasso"
                                                            " or box tool, then click below:"
                                                        ),
                                                        html.Button(
                                                            "Show Selected Points",
                                                            id="show-selected",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                        ),
                                                        html.Div(
                                                            id="show-selected-status",
                                                            style={
                                                                "marginTop": "5px",
                                                                "fontSize": "12px",
                                                                "color": "blue",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="selected-points-info",
                                                            style={
                                                                "marginTop": "15px",
                                                                "fontSize": "12px",
                                                            },
                                                        ),
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
                                                                    id="selection-filename",
                                                                    type="text",
                                                                    placeholder="Enter filename (without extension)",
                                                                    style={
                                                                        "width": "100%",
                                                                        "marginBottom": "10px",
                                                                    },
                                                                ),
                                                                html.Button(
                                                                    "Save Selection",
                                                                    id="save-selection-btn",
                                                                    n_clicks=0,
                                                                    className="btn-secondary",
                                                                ),
                                                                html.Div(
                                                                    id="save-selection-status",
                                                                    style={
                                                                        "marginTop": "5px",
                                                                        "fontSize": "12px",
                                                                        "color": "blue",
                                                                    },
                                                                ),
                                                                dcc.Download(
                                                                    id="download-selection"
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
                                                            id="umap-graph-selected-only",
                                                            config={
                                                                "displayModeBar": True
                                                            },
                                                            style={"height": "600px"},
                                                        ),
                                                        html.Div(
                                                            id="debug-output-selected-only",
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
                                # Graph 2.5: Selected Points from Custom Scatter
                                html.Div(
                                    [
                                        html.H3(
                                            "Selected Points from Custom Feature Plot",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Select points on Custom Feature Plot using the lasso"
                                                            " or box tool, then click below:"
                                                        ),
                                                        html.Button(
                                                            "Show Selected Points",
                                                            id="show-selected-graph15",
                                                            n_clicks=0,
                                                            className="btn-secondary",
                                                        ),
                                                        html.Div(
                                                            id="show-selected-graph15-status",
                                                            style={
                                                                "marginTop": "5px",
                                                                "fontSize": "12px",
                                                                "color": "blue",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="selected-points-info-graph25",
                                                            style={
                                                                "marginTop": "15px",
                                                                "fontSize": "12px",
                                                            },
                                                        ),
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
                                                                    id="selection-filename-graph25",
                                                                    type="text",
                                                                    placeholder="Enter filename (without extension)",
                                                                    style={
                                                                        "width": "100%",
                                                                        "marginBottom": "10px",
                                                                    },
                                                                ),
                                                                html.Button(
                                                                    "Save Selection",
                                                                    id="save-selection-graph25-btn",
                                                                    n_clicks=0,
                                                                    className="btn-secondary",
                                                                ),
                                                                html.Div(
                                                                    id="save-selection-graph25-status",
                                                                    style={
                                                                        "marginTop": "5px",
                                                                        "fontSize": "12px",
                                                                        "color": "blue",
                                                                    },
                                                                ),
                                                                dcc.Download(
                                                                    id="download-selection-graph25"
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
                                                            id="graph25",
                                                            config={
                                                                "displayModeBar": True
                                                            },
                                                            style={"height": "600px"},
                                                        ),
                                                        html.Div(
                                                            id="debug-output-graph25",
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
                            ]
                        )
                    ],
                ),
                # Filtering Tab
                dcc.Tab(
                    label="Data Filtering",
                    value="filtering-tab",
                    children=[
                        html.Div(
                            [
                                # UMAP Filtering Section
                                html.Div(
                                    [
                                        html.H3(
                                            "UMAP Filtering Options",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            [
                                                # Left panel with controls
                                                html.Div(
                                                    [
                                                        # Density-based filtering section
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Density-Based UMAP Filtering",
                                                                    style={
                                                                        "marginBottom": "10px"
                                                                    },
                                                                ),
                                                                html.Label(
                                                                    "Density Calculation Bandwidth:"
                                                                ),
                                                                dcc.Slider(
                                                                    id="umap-density-bandwidth-slider",
                                                                    min=0.01,
                                                                    max=1.0,
                                                                    step=0.01,
                                                                    value=0.1,
                                                                    marks={
                                                                        0.1: "0.1",
                                                                        0.3: "0.3",
                                                                        0.5: "0.5",
                                                                        0.7: "0.7",
                                                                        0.9: "0.9",
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Label(
                                                                    "Density Threshold Percentile:"
                                                                ),
                                                                dcc.Slider(
                                                                    id="umap-density-threshold-slider",
                                                                    min=0,
                                                                    max=100,
                                                                    step=1,
                                                                    value=50,
                                                                    marks={
                                                                        0: "0%",
                                                                        25: "25%",
                                                                        50: "50%",
                                                                        75: "75%",
                                                                        100: "100%",
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Button(
                                                                    "Apply UMAP Density Filter",
                                                                    id="apply-umap-density-filter",
                                                                    className="btn-secondary",
                                                                ),
                                                                html.Div(
                                                                    id="umap-density-filter-status",
                                                                    style={
                                                                        "marginTop": "5px",
                                                                        "fontSize": "12px",
                                                                        "color": "blue",
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Div(
                                                                    id="umap-density-filter-info",
                                                                    style={
                                                                        "fontSize": "12px"
                                                                    },
                                                                ),
                                                            ],
                                                            style={
                                                                "padding": "10px",
                                                                "border": "1px solid #ddd",
                                                                "borderRadius": "5px",
                                                                "marginBottom": "15px",
                                                            },
                                                        ),
                                                        # Physics-based filtering section
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Physics Parameter UMAP Filtering",
                                                                    style={
                                                                        "marginBottom": "10px"
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="umap-physics-filter-container",
                                                                    children=[
                                                                        html.Label(
                                                                            "Select Physics Parameter:"
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="umap-physics-parameter-dropdown",
                                                                            options=[
                                                                                {
                                                                                    "label": "KER (Kinetic Energy Release)",
                                                                                    "value": "KER",
                                                                                },
                                                                                {
                                                                                    "label": "EESum (Sum of Electron"
                                                                                    " Energies)",
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
                                                                                {
                                                                                    "label": "Ion-Ion Angle",
                                                                                    "value": "angle_ion1_ion2",
                                                                                },
                                                                            ],
                                                                            value=None,
                                                                            placeholder="Select parameter to filter",
                                                                        ),
                                                                        html.Br(),
                                                                        html.Div(
                                                                            id="umap-parameter-filter-controls",
                                                                            style={
                                                                                "display": "none"
                                                                            },
                                                                            children=[
                                                                                html.Label(
                                                                                    "Parameter Range:"
                                                                                ),
                                                                                dcc.RangeSlider(
                                                                                    id="umap-parameter-range-slider",
                                                                                    min=0,
                                                                                    max=100,
                                                                                    step=0.1,
                                                                                    value=[
                                                                                        0,
                                                                                        100,
                                                                                    ],
                                                                                    marks={
                                                                                        0: "0",
                                                                                        100: "100",
                                                                                    },
                                                                                    tooltip={
                                                                                        "placement": "bottom",
                                                                                        "always_visible": True,
                                                                                    },
                                                                                ),
                                                                                html.Br(),
                                                                                html.Button(
                                                                                    "Apply UMAP Parameter Filter",
                                                                                    id="apply-umap-parameter-filter",
                                                                                    className="btn-secondary",
                                                                                ),
                                                                                html.Div(
                                                                                    id="umap-parameter-filter-status",
                                                                                    style={
                                                                                        "marginTop": "5px",
                                                                                        "fontSize": "12px",
                                                                                        "color": "blue",
                                                                                    },
                                                                                ),
                                                                            ],
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                            style={
                                                                "padding": "10px",
                                                                "border": "1px solid #ddd",
                                                                "borderRadius": "5px",
                                                                "marginBottom": "15px",
                                                            },
                                                        ),
                                                        # Combined filtering results section
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Filtered UMAP Results",
                                                                    style={
                                                                        "marginBottom": "10px"
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="umap-filtered-data-info",
                                                                    style={
                                                                        "fontSize": "12px"
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Label(
                                                                    "Save filtered UMAP points:"
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dcc.Input(
                                                                            id="umap-filtered-data-filename",
                                                                            type="text",
                                                                            placeholder="Enter filename (without extension)",
                                                                            style={
                                                                                "width": "100%",
                                                                                "marginBottom": "10px",
                                                                            },
                                                                        ),
                                                                        html.Button(
                                                                            "Save Filtered UMAP Data",
                                                                            id="save-umap-filtered-data-btn",
                                                                            className="btn-secondary",
                                                                        ),
                                                                        html.Div(
                                                                            id="save-umap-filtered-data-status",
                                                                            style={
                                                                                "marginTop": "5px",
                                                                                "fontSize": "12px",
                                                                                "color": "blue",
                                                                            },
                                                                        ),
                                                                        dcc.Download(
                                                                            id="download-umap-filtered-data"
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                            style={
                                                                "padding": "10px",
                                                                "border": "1px solid #ddd",
                                                                "borderRadius": "5px",
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "25%",
                                                        "paddingRight": "20px",
                                                    },
                                                ),
                                                # Right panel with visualization
                                                html.Div(
                                                    [
                                                        dcc.Graph(
                                                            id="umap-filtered-data-graph",
                                                            config={
                                                                "displayModeBar": True
                                                            },
                                                            style={"height": "600px"},
                                                        ),
                                                        html.Div(
                                                            id="umap-filtered-data-debug",
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
                                # Custom Feature Filtering Section
                                html.Div(
                                    [
                                        html.H3(
                                            "Custom Feature Filtering Options",
                                            style={"textAlign": "center"},
                                        ),
                                        html.Div(
                                            [
                                                # Left panel with controls
                                                html.Div(
                                                    [
                                                        # Density-based filtering section
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Density-Based Filtering",
                                                                    style={
                                                                        "marginBottom": "10px"
                                                                    },
                                                                ),
                                                                html.Label(
                                                                    "Density Calculation Bandwidth:"
                                                                ),
                                                                dcc.Slider(
                                                                    id="density-bandwidth-slider",
                                                                    min=0.01,
                                                                    max=1.0,
                                                                    step=0.01,
                                                                    value=0.1,
                                                                    marks={
                                                                        0.1: "0.1",
                                                                        0.3: "0.3",
                                                                        0.5: "0.5",
                                                                        0.7: "0.7",
                                                                        0.9: "0.9",
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Label(
                                                                    "Density Threshold Percentile:"
                                                                ),
                                                                dcc.Slider(
                                                                    id="density-threshold-slider",
                                                                    min=0,
                                                                    max=100,
                                                                    step=1,
                                                                    value=50,
                                                                    marks={
                                                                        0: "0%",
                                                                        25: "25%",
                                                                        50: "50%",
                                                                        75: "75%",
                                                                        100: "100%",
                                                                    },
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Button(
                                                                    "Apply Density Filter",
                                                                    id="apply-density-filter",
                                                                    className="btn-secondary",
                                                                ),
                                                                html.Div(
                                                                    id="density-filter-status",
                                                                    style={
                                                                        "marginTop": "5px",
                                                                        "fontSize": "12px",
                                                                        "color": "blue",
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Div(
                                                                    id="density-filter-info",
                                                                    style={
                                                                        "fontSize": "12px"
                                                                    },
                                                                ),
                                                            ],
                                                            style={
                                                                "padding": "10px",
                                                                "border": "1px solid #ddd",
                                                                "borderRadius": "5px",
                                                                "marginBottom": "15px",
                                                            },
                                                        ),
                                                        # Physics-based filtering section
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Physics Parameter Filtering",
                                                                    style={
                                                                        "marginBottom": "10px"
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="physics-filter-container",
                                                                    children=[
                                                                        html.Label(
                                                                            "Select Physics Parameter:"
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="physics-parameter-dropdown",
                                                                            options=[
                                                                                {
                                                                                    "label": "KER (Kinetic Energy Release)",
                                                                                    "value": "KER",
                                                                                },
                                                                                {
                                                                                    "label": "EESum (Sum of Electron"
                                                                                    " Energies)",
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
                                                                                {
                                                                                    "label": "Ion-Ion Angle",
                                                                                    "value": "angle_ion1_ion2",
                                                                                },
                                                                            ],
                                                                            value=None,
                                                                            placeholder="Select parameter to filter",
                                                                        ),
                                                                        html.Br(),
                                                                        html.Div(
                                                                            id="parameter-filter-controls",
                                                                            style={
                                                                                "display": "none"
                                                                            },
                                                                            children=[
                                                                                html.Label(
                                                                                    "Parameter Range:"
                                                                                ),
                                                                                dcc.RangeSlider(
                                                                                    id="parameter-range-slider",
                                                                                    min=0,
                                                                                    max=100,
                                                                                    step=0.1,
                                                                                    value=[
                                                                                        0,
                                                                                        100,
                                                                                    ],
                                                                                    marks={
                                                                                        0: "0",
                                                                                        100: "100",
                                                                                    },
                                                                                    tooltip={
                                                                                        "placement": "bottom",
                                                                                        "always_visible": True,
                                                                                    },
                                                                                ),
                                                                                html.Br(),
                                                                                html.Button(
                                                                                    "Apply Parameter Filter",
                                                                                    id="apply-parameter-filter",
                                                                                    className="btn-secondary",
                                                                                ),
                                                                                html.Div(
                                                                                    id="parameter-filter-status",
                                                                                    style={
                                                                                        "marginTop": "5px",
                                                                                        "fontSize": "12px",
                                                                                        "color": "blue",
                                                                                    },
                                                                                ),
                                                                            ],
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                            style={
                                                                "padding": "10px",
                                                                "border": "1px solid #ddd",
                                                                "borderRadius": "5px",
                                                                "marginBottom": "15px",
                                                            },
                                                        ),
                                                        # Combined filtering results section
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Filtered Data Results",
                                                                    style={
                                                                        "marginBottom": "10px"
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="filtered-data-info",
                                                                    style={
                                                                        "fontSize": "12px"
                                                                    },
                                                                ),
                                                                html.Br(),
                                                                html.Label(
                                                                    "Save filtered points:"
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dcc.Input(
                                                                            id="filtered-data-filename",
                                                                            type="text",
                                                                            placeholder="Enter filename (without extension)",
                                                                            style={
                                                                                "width": "100%",
                                                                                "marginBottom": "10px",
                                                                            },
                                                                        ),
                                                                        html.Button(
                                                                            "Save Filtered Data",
                                                                            id="save-filtered-data-btn",
                                                                            className="btn-secondary",
                                                                        ),
                                                                        html.Div(
                                                                            id="save-filtered-data-status",
                                                                            style={
                                                                                "marginTop": "5px",
                                                                                "fontSize": "12px",
                                                                                "color": "blue",
                                                                            },
                                                                        ),
                                                                        dcc.Download(
                                                                            id="download-filtered-data"
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                            style={
                                                                "padding": "10px",
                                                                "border": "1px solid #ddd",
                                                                "borderRadius": "5px",
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "25%",
                                                        "paddingRight": "20px",
                                                    },
                                                ),
                                                # Right panel with visualization
                                                html.Div(
                                                    [
                                                        dcc.Graph(
                                                            id="filtered-data-graph",
                                                            config={
                                                                "displayModeBar": True
                                                            },
                                                            style={"height": "600px"},
                                                        ),
                                                        html.Div(
                                                            id="filtered-data-debug",
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
                            ]
                        )
                    ],
                ),
            ],
        )
    ]
