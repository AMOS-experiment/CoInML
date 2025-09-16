import json

import dash
from dash import ALL, Input, Output, State, callback, html

from sculpt.utils.file_handlers import parse_contents, reorganize_df
from sculpt.utils.ui import create_feature_categories_ui


# Callback for file upload and removal
@callback(
    Output("stored-files", "data"),
    Output("file-list", "children"),
    Output("features-data-store", "data"),
    Output("feature-selection-ui-graph1", "children"),
    Output("feature-selection-ui-graph3", "children"),
    Output("feature-selection-ui-graph3-selection", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    Input({"type": "remove-button", "index": ALL}, "n_clicks"),
    State("stored-files", "data"),
    State("features-data-store", "data"),
    State("configuration-profiles-store", "data"),
    State("file-config-assignments-store", "data"),
    prevent_initial_call=True,
)
def update_files(
    new_contents,
    new_filenames,
    remove_n_clicks,
    current_store,
    features_store,
    profiles_store,
    assignments_store,
):
    """Update the stored files and calculate physics features."""
    try:
        if current_store is None:
            current_store = []

        if features_store is None:
            features_store = {}

        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        triggered_prop = ctx.triggered[0]["prop_id"]
        print(f"Triggered by: {triggered_prop}")

        # Handle file uploads
        if "upload-data.contents" in triggered_prop:
            print("Handling file upload...")
            if new_contents is not None and new_filenames is not None:
                next_id = max([f["id"] for f in current_store], default=-1) + 1
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
                                    "id": next_id,
                                    "filename": fname,
                                    "data": df.to_json(
                                        date_format="iso", orient="split"
                                    ),
                                    "is_selection": True,  # Flag to indicate this is a selection file
                                }
                                current_store.append(file_dict)
                                next_id += 1
                            else:
                                # Handle regular COLTRIMS file
                                print(f"{fname} is a regular COLTRIMS file")
                                try:
                                    df_reorg = reorganize_df(df)
                                    print(
                                        f"Successfully reorganized dataframe for {fname}"
                                    )

                                    # Calculate physics features
                                    print(
                                        f"Deferring feature calculation for {fname} until profile assignment"
                                    )

                                    # TODO: Check if this was indeed deleted
                                    # try:
                                    #     print(f"Calculating features for {fname}...")
                                    #     sample_size = min(1000, len(df_reorg))
                                    #     df_sample = (
                                    #         df_reorg.sample(
                                    #             n=sample_size, random_state=42
                                    #         )
                                    #         if len(df_reorg) > sample_size
                                    #         else df_reorg
                                    #     )

                                    #     # Get molecular configuration from the stores
                                    #     profiles_store = ctx.states.get(
                                    #         "configuration-profiles-store.data", {}
                                    #     )
                                    #     assignments_store = ctx.states.get(
                                    #         "file-config-assignments-store.data", {}
                                    #     )

                                    #     # Process file with its assigned configuration
                                    #     profile_name = (
                                    #         assignments_store.get(fname)
                                    #         if assignments_store
                                    #         else None
                                    #     )

                                    #     if (
                                    #         profile_name
                                    #         and profiles_store
                                    #         and profile_name in profiles_store
                                    #     ):
                                    #         # Use the assigned profile
                                    #         profile_config = profiles_store[
                                    #             profile_name
                                    #         ]
                                    #         print(
                                    #             f"Processing {fname} with profile: {profile_name}"
                                    #         )
                                    #         df_features = (
                                    #             calculate_physics_features_with_profile(
                                    #                 df_sample, profile_config
                                    #             )
                                    #         )
                                    #     else:
                                    #         # Use default configuration
                                    #         print(
                                    #             f"Processing {fname} with default configuration"
                                    #         )
                                    #         df_features = (
                                    #             calculate_physics_features_flexible(
                                    #                 df_sample, None
                                    #             )
                                    #         )
                                    #     print(
                                    #         f"Feature calculation complete for {fname}"
                                    #     )

                                    #     # Store the column names
                                    #     if "column_names" not in features_store:
                                    #         features_store["column_names"] = list(
                                    #             df_features.columns
                                    #         )
                                    #         print("Updated feature column names")
                                    # except Exception as e:
                                    #     print(
                                    #         f"Error during feature calculation for {fname}: {e}"
                                    #     )
                                    #     import traceback

                                    #     traceback.print_exc()
                                    #     df_features = df_reorg

                                    file_dict = {
                                        "id": next_id,
                                        "filename": fname,
                                        "data": df_reorg.to_json(
                                            date_format="iso", orient="split"
                                        ),
                                        "is_selection": False,  # Flag to indicate this is not a selection file
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
        elif "remove-button" in triggered_prop:
            try:
                print("Handling file removal...")
                triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
                print(f"Triggered ID: {triggered_id}")
                remove_obj = json.loads(triggered_id)
                remove_id = remove_obj.get("index", None)
                print(f"Removing file with ID: {remove_id}")

                # Remove the file from the store
                current_store = [f for f in current_store if f["id"] != remove_id]
                print(f"Files after removal: {len(current_store)}")

                # Create file list UI
                file_list_children = []
                for f in current_store:
                    is_selection = f.get("is_selection", False)
                    selection_label = " (Selection)" if is_selection else ""

                    file_list_children.append(
                        html.Div(
                            [
                                html.Span(
                                    f["filename"] + selection_label,
                                    style={
                                        "color": "#1E88E5" if is_selection else "black"
                                    },
                                ),
                                html.Button(
                                    "×",
                                    id={"type": "remove-button", "index": f["id"]},
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "10px",
                                        "color": "red",
                                        "width": "auto",
                                    },
                                ),
                            ],
                            style={
                                "margin": "5px",
                                "display": "inline-block",
                                "border": "1px solid #ccc",
                                "padding": "5px",
                                "borderRadius": "3px",
                                "backgroundColor": (
                                    "#e3f2fd" if is_selection else "white"
                                ),
                            },
                        )
                    )

                # IMPORTANT: Don't process features when removing files
                # Just create the UI based on existing feature store
                if features_store and "column_names" in features_store:
                    feature_ui_graph1 = create_feature_categories_ui(
                        features_store["column_names"], "graph1"
                    )
                    feature_ui_graph3 = create_feature_categories_ui(
                        features_store["column_names"], "graph3"
                    )
                    feature_ui_graph3_selection = create_feature_categories_ui(
                        features_store["column_names"], "graph3-selection"
                    )
                else:
                    feature_ui_graph1 = [
                        html.Div(
                            "Upload files to see available features",
                            style={"color": "gray"},
                        )
                    ]
                    feature_ui_graph3 = [
                        html.Div(
                            "Upload files to see available features",
                            style={"color": "gray"},
                        )
                    ]
                    feature_ui_graph3_selection = [
                        html.Div(
                            "Upload files to see available features",
                            style={"color": "gray"},
                        )
                    ]

                # Return early to avoid processing all files again
                return (
                    current_store,
                    file_list_children,
                    features_store,
                    feature_ui_graph1,
                    feature_ui_graph3,
                    feature_ui_graph3_selection,
                )

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
            is_selection = f.get("is_selection", False)
            selection_label = " (Selection)" if is_selection else ""

            file_list_children.append(
                html.Div(
                    [
                        html.Span(
                            f["filename"] + selection_label,
                            style={"color": "#1E88E5" if is_selection else "black"},
                        ),
                        html.Button(
                            "×",
                            id={"type": "remove-button", "index": f["id"]},
                            n_clicks=0,
                            style={
                                "marginLeft": "10px",
                                "color": "red",
                                "width": "auto",
                            },
                        ),
                    ],
                    style={
                        "margin": "5px",
                        "display": "inline-block",
                        "border": "1px solid #ccc",
                        "padding": "5px",
                        "borderRadius": "3px",
                        "backgroundColor": "#e3f2fd" if is_selection else "white",
                    },
                )
            )

        # Create feature selection UIs for both graphs
        print("Creating feature selection UIs...")
        if not current_store or "column_names" not in features_store:
            feature_ui_graph1 = [
                html.Div(
                    "Upload files to see available features", style={"color": "gray"}
                )
            ]
            feature_ui_graph3 = [
                html.Div(
                    "Upload files to see available features", style={"color": "gray"}
                )
            ]
            feature_ui_graph3_selection = [
                html.Div(
                    "Upload files to see available features", style={"color": "gray"}
                )
            ]
        else:
            # Create feature selection UI for Graph 1
            feature_ui_graph1 = create_feature_categories_ui(
                features_store["column_names"], "graph1"
            )

            # Create feature selection UI for Graph 3 (identical structure but different IDs)
            feature_ui_graph3 = create_feature_categories_ui(
                features_store["column_names"], "graph3"
            )

            # Create feature selection UI for Graph 3 Selection (identical structure but different IDs)
            feature_ui_graph3_selection = create_feature_categories_ui(
                features_store["column_names"], "graph3-selection"
            )

        print("Callback completed successfully")
        return (
            current_store,
            file_list_children,
            features_store,
            feature_ui_graph1,
            feature_ui_graph3,
            feature_ui_graph3_selection,
        )

    except Exception as e:
        print(f"Fatal error in update_files callback: {e}")
        import traceback

        traceback.print_exc()
        # Return the current values to avoid breaking the app
        empty_ui = [
            html.Div(
                "Error loading features. Check console for details.",
                style={"color": "red"},
            )
        ]
        return (
            current_store or [],
            [],
            features_store or {},
            empty_ui,
            empty_ui,
            empty_ui,
        )


@callback(
    Output("file-selector-graph15", "options"),
    Output("file-selector-graph15", "value"),
    Input("stored-files", "data"),
)
def update_file_selector_graph15(stored_files):
    """Update the file selector dropdown for Graph 1.5 based on uploaded files."""
    if not stored_files:
        return [], []

    options = [{"label": f["filename"], "value": f["id"]} for f in stored_files]
    values = [f["id"] for f in stored_files]  # Select all files by default

    return options, values


# Callback to update the file selector checklist
@callback(
    Output("umap-file-selector", "options"),
    Output("umap-file-selector", "value"),
    Input("stored-files", "data"),
)
def update_umap_selector(stored_files):
    """Update the file selector dropdown based on uploaded files."""
    if not stored_files:
        return [], []

    options = [{"label": f["filename"], "value": f["id"]} for f in stored_files]
    values = [f["id"] for f in stored_files]  # Select all files by default

    return options, values
