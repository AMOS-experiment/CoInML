from dash import dcc, html


def create_data_management_tab():
    return html.Div(
        [
            # File Upload Section
            html.Div(
                [
                    html.H3("File Upload"),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Drag and drop or ", html.A("select data files")]
                        ),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "margin": "10px 0",
                        },
                        multiple=True,
                    ),
                    html.Div(id="file-list", children=[]),
                ],
                className="container",
            ),
            # Configuration Section
            html.Div(
                [
                    html.H3("Molecular Configuration Management"),
                    # Sub-tabs for configuration
                    dcc.Tabs(
                        id="config-tabs",
                        value="config-profiles",
                        children=[
                            # Configuration Profiles tab
                            dcc.Tab(
                                label="Configuration Profiles",
                                value="config-profiles",
                                children=[
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H4("Configuration Profiles"),
                                                    # Active Profiles Display
                                                    html.Div(
                                                        [
                                                            html.H5(
                                                                "Active Profiles",
                                                                style={
                                                                    "color": "#1976d2",
                                                                    "marginBottom": "15px",
                                                                },
                                                            ),
                                                            html.Div(
                                                                id="active-profiles-display",
                                                                children=[
                                                                    html.Div(
                                                                        "No profiles created yet",
                                                                        style={
                                                                            "color": "gray",
                                                                            "fontStyle": "italic",
                                                                        },
                                                                    )
                                                                ],
                                                                style={
                                                                    "minHeight": "100px",
                                                                    "padding": "15px",
                                                                    "backgroundColor": "#f5f5f5",
                                                                    "borderRadius": "5px",
                                                                    "marginBottom": "20px",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    # Particle Configuration Section
                                                    html.Div(
                                                        [
                                                            html.H5(
                                                                "Particle Configuration",
                                                                style={
                                                                    "color": "#6a1b9a",
                                                                    "marginBottom": "15px",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Div(
                                                                        [
                                                                            html.Label(
                                                                                "Number of Ions:",
                                                                                style={
                                                                                    "marginRight": "10px"
                                                                                },
                                                                            ),
                                                                            dcc.Input(
                                                                                id="num-ions",
                                                                                type="number",
                                                                                value=2,
                                                                                min=0,
                                                                                max=10,
                                                                                style={
                                                                                    "width": "80px",
                                                                                    "marginRight": "20px",
                                                                                },
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
                                                                                "Number of Neutrals:",
                                                                                style={
                                                                                    "marginRight": "10px"
                                                                                },
                                                                            ),
                                                                            dcc.Input(
                                                                                id="num-neutrals",
                                                                                type="number",
                                                                                value=1,
                                                                                min=0,
                                                                                max=10,
                                                                                style={
                                                                                    "width": "80px",
                                                                                    "marginRight": "20px",
                                                                                },
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
                                                                                "Number of Electrons:",
                                                                                style={
                                                                                    "marginRight": "10px"
                                                                                },
                                                                            ),
                                                                            dcc.Input(
                                                                                id="num-electrons",
                                                                                type="number",
                                                                                value=2,
                                                                                min=0,
                                                                                max=10,
                                                                                style={
                                                                                    "width": "80px"
                                                                                },
                                                                            ),
                                                                        ],
                                                                        style={
                                                                            "display": "inline-block"
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "marginBottom": "15px",
                                                                    "padding": "10px",
                                                                    "backgroundColor": "#f5f5f5",
                                                                    "borderRadius": "5px",
                                                                },
                                                            ),
                                                            # Dynamic particle configuration container
                                                            html.Div(
                                                                id="particle-config-container",
                                                                children=[],
                                                            ),
                                                        ],
                                                        style={
                                                            "padding": "15px",
                                                            "backgroundColor": "#f3e5f5",
                                                            "borderRadius": "5px",
                                                            "marginBottom": "20px",
                                                        },
                                                    ),
                                                    # Create New Profile Section
                                                    html.Div(
                                                        [
                                                            html.H5(
                                                                "Create New Profile",
                                                                style={
                                                                    "color": "#388e3c",
                                                                    "marginBottom": "15px",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Div(
                                                                        [
                                                                            html.Label(
                                                                                "Profile Name:",
                                                                                style={
                                                                                    "marginRight": "10px",
                                                                                    "fontWeight": "bold",
                                                                                },
                                                                            ),
                                                                            dcc.Input(
                                                                                id="config-profile-name",
                                                                                type="text",
                                                                                placeholder="e.g., D2O, HDO, H2O",
                                                                                style={
                                                                                    "width": "250px",
                                                                                    "marginRight": "20px",
                                                                                },
                                                                            ),
                                                                            html.Button(
                                                                                "Create New Profile",
                                                                                id="create-profile-btn",
                                                                                n_clicks=0,
                                                                                className="btn-secondary",
                                                                                style={
                                                                                    "width": "auto",
                                                                                    "padding": "10px 20px",
                                                                                },
                                                                            ),
                                                                        ],
                                                                        style={
                                                                            "display": "flex",
                                                                            "alignItems": "center",
                                                                            "marginBottom": "20px",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "15px",
                                                                    "backgroundColor": "#e8f5e9",
                                                                    "borderRadius": "5px",
                                                                    "marginBottom": "20px",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    # Selected Profile Editor
                                                    html.Div(
                                                        [
                                                            html.H5(
                                                                "Edit Selected Profile",
                                                                style={
                                                                    "color": "#f57c00",
                                                                    "marginBottom": "15px",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Select Profile to Edit:",
                                                                        style={
                                                                            "marginRight": "10px",
                                                                            "fontWeight": "bold",
                                                                        },
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="active-profile-dropdown",
                                                                        options=[],
                                                                        value=None,
                                                                        style={
                                                                            "width": "300px",
                                                                            "marginBottom": "15px",
                                                                        },
                                                                    ),
                                                                    html.Button(
                                                                        "Update Profile",
                                                                        id="update-profile-btn",
                                                                        n_clicks=0,
                                                                        className="btn-secondary",
                                                                        style={
                                                                            "width": "auto",
                                                                            "padding": "10px 20px",
                                                                            "marginTop": "10px",
                                                                        },
                                                                    ),
                                                                    html.Div(
                                                                        id="profile-edit-status",
                                                                        style={
                                                                            "marginTop": "10px",
                                                                            "color": "green",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "15px",
                                                                    "backgroundColor": "#fff3e0",
                                                                    "borderRadius": "5px",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                style={"padding": "20px"},
                                            )
                                        ],
                                        style={"padding": "20px"},
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="File Assignment",
                                value="file-assignment",
                                children=[
                                    html.Div(
                                        [
                                            html.H4(
                                                "Assign Configuration Profiles to Files"
                                            ),
                                            html.Div(
                                                id="file-configuration-assignment",
                                                children=[
                                                    html.Div(
                                                        "Upload files first to assign configurations",
                                                        style={
                                                            "color": "gray",
                                                            "fontStyle": "italic",
                                                        },
                                                    )
                                                ],
                                            ),
                                            html.Br(),
                                            html.Button(
                                                "Apply File Assignments",
                                                id="apply-file-config-btn",
                                                n_clicks=0,
                                                className="btn-secondary",
                                            ),
                                            html.Div(
                                                id="file-assignment-status",
                                                style={
                                                    "marginTop": "10px",
                                                    "color": "green",
                                                },
                                            ),
                                        ],
                                        style={"padding": "20px"},
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
                className="container",
            ),
        ]
    )
