from dash import Input, Output, callback, dcc, html

from sculpt.utils.ui import create_feature_categories_ui


# Callback to update the feature dropdowns when data is available
@callback(
    Output("x-axis-feature-graph15", "options"),
    Output("y-axis-feature-graph15", "options"),
    Input("features-data-store", "data"),
    prevent_initial_call=True,
)
def update_feature_dropdowns_graph15(features_data):
    """Update the dropdown options for custom feature plot in Graph 1.5."""
    if not features_data or "column_names" not in features_data:
        return [], []

    # Get all feature columns
    feature_columns = features_data["column_names"]

    # Create dropdown options
    options = [{"label": col, "value": col} for col in feature_columns]

    return options, options


# Callback to update the feature dropdowns when data is available
@callback(
    Output("x-axis-feature", "options"),
    Output("y-axis-feature", "options"),
    Input("features-data-store", "data"),
    prevent_initial_call=True,
)
def update_feature_dropdowns(features_data):
    """Update the dropdown options for custom feature plot."""
    if not features_data or "column_names" not in features_data:
        return [], []

    # Get all feature columns
    feature_columns = features_data["column_names"]

    # Create dropdown options
    options = [{"label": col, "value": col} for col in feature_columns]

    return options, options


# Update the UI for autoencoder feature selection
@callback(
    Output("feature-selection-ui-autoencoder", "children"),
    Input("features-data-store", "data"),
    prevent_initial_call=True,
)
def update_autoencoder_feature_ui(features_data):
    """Update the feature selection UI for the autoencoder."""
    if not features_data or "column_names" not in features_data:
        return [
            html.Div("Upload files to see available features", style={"color": "gray"})
        ]

    # Create feature selection UI for autoencoder
    return create_feature_categories_ui(features_data["column_names"], "autoencoder")


@callback(
    Output("feature-selection-ui-genetic", "children"),
    Input("features-data-store", "data"),
    Input(
        "genetic-features-store", "data"
    ),  # Add this to update UI when genetic features are created
    prevent_initial_call=True,
)
def update_genetic_feature_ui(features_data, genetic_features_store):
    """Update the feature selection UI for genetic programming."""
    feature_columns = []

    # First get standard features
    if features_data and "column_names" in features_data:
        feature_columns = features_data["column_names"]

    # Then check if we have discovered genetic features to add
    if genetic_features_store and "feature_names" in genetic_features_store:
        gp_features = genetic_features_store["feature_names"]
        expressions = genetic_features_store.get("expressions", [])

        # Add a special category for discovered genetic features
        genetic_category = html.Div(
            [
                html.Div(
                    "Discovered Genetic Features",
                    className="feature-category-title",
                    style={"color": "#d32f2f", "fontWeight": "bold"},
                ),
                dcc.Checklist(
                    id={
                        "type": "feature-selector-genetic",
                        "category": "GeneticFeatures",
                    },
                    options=[
                        {
                            "label": f"{feat} ({expressions[i] if i < len(expressions) else ''})",
                            "value": feat,
                        }
                        for i, feat in enumerate(gp_features)
                    ],
                    value=[],  # No default selection
                    labelStyle={"display": "block"},
                ),
            ],
            className="feature-category",
            style={
                "backgroundColor": "#ffebee",
                "padding": "10px",
                "marginBottom": "15px",
            },
        )

        # Create regular feature selection UI without the genetic features
        regular_ui = create_feature_categories_ui(feature_columns, "genetic")

        # Combine genetic features and regular features
        if len(gp_features) > 0:
            return [genetic_category] + regular_ui

    # If no genetic features or standard features, use the default UI
    if not feature_columns:
        return [
            html.Div("Upload files to see available features", style={"color": "gray"})
        ]
    else:
        return create_feature_categories_ui(feature_columns, "genetic")


@callback(
    Output(
        "mi-feature-selection-ui", "children"
    ),  # This should match the ID in your MI tab
    Input("features-data-store", "data"),
    Input("mi-features-store", "data"),
    prevent_initial_call=True,
)
def update_mi_feature_ui(features_data, mi_features_store):
    """Update the feature selection UI for MI analysis."""
    if not features_data or "column_names" not in features_data:
        return [
            html.Div("Upload files to see available features", style={"color": "gray"})
        ]

    # Create feature selection UI
    return create_feature_categories_ui(features_data["column_names"], "mi")
