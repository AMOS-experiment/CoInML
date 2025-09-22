from dash import dcc, html

from sculpt.components.advanced_analysis import create_advanced_analysis_tab
from sculpt.components.data_management import create_data_management_tab
from sculpt.components.ml_analysis import create_machine_learning_tab
from sculpt.components.selection import create_selection_tab
from sculpt.components.visualization import create_basic_viz_tab


def create_header():
    return html.Div(
        [
            html.H1("SCULPT", style={"textAlign": "center", "marginBottom": "5px"}),
            html.P(
                "Supervised Clustering and Uncovering Latent Patterns with Training",
                style={"textAlign": "center", "color": "gray", "marginTop": "0px"},
            ),
        ]
    )


def create_layout():
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            create_header(),
            # Main Tabs
            dcc.Tabs(
                id="main-tabs",
                value="tab-data",
                children=[
                    # Tab 1: Data Management
                    dcc.Tab(
                        label="Data & Configuration",
                        value="tab-data",
                        children=create_data_management_tab(),
                    ),
                    # Tab 2: Basic Visualizations
                    dcc.Tab(
                        label="Basic Analysis",
                        value="tab-basic",
                        children=create_basic_viz_tab(),
                    ),
                    # Tab 3: Selection and Filtering
                    dcc.Tab(
                        label="Selection & Filtering",
                        value="tab-selection",
                        children=create_selection_tab(),
                    ),
                    # Tab 4: Advanced Analysis
                    dcc.Tab(
                        label="Advanced Analysis",
                        value="tab-advanced",
                        children=create_advanced_analysis_tab(),
                    ),
                    # Tab 5: Machine Learning
                    dcc.Tab(
                        label="Machine Learning",
                        value="tab-ml",
                        children=create_machine_learning_tab(),
                    ),
                ],
            ),
            # Hidden stores - keep all your existing stores
            dcc.Store(id="stored-files", data=[]),
            dcc.Store(id="combined-data-store", data=""),
            dcc.Store(id="features-data-store", data={}),
            dcc.Store(id="selected-points-store", data=[]),
            dcc.Store(id="selected-points-run-store", data=[]),
            dcc.Store(id="graph3-selection-umap-store", data=[]),
            dcc.Store(id="autoencoder-latent-store", data=[]),
            dcc.Interval(
                id="training-interval", interval=1000, n_intervals=0, disabled=True
            ),
            dcc.Store(id="genetic-features-store", data=[]),
            dcc.Store(id="mi-features-store", data=[]),
            dcc.Store(id="selected-points-store-graph15", data=[]),
            dcc.Store(id="filtered-data-store", data={}),
            dcc.Store(id="umap-filtered-data-store", data={}),
            dcc.Store(id="configuration-profiles-store", data={}),
            dcc.Store(id="file-config-assignments-store", data={}),
            dcc.Store(id="current-profile-store", data=None),
            dcc.Store(
                id="particle-count-store",
                data={"ions": 2, "neutrals": 1, "electrons": 2},
            ),
        ]
    )
