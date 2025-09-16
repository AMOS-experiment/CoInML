from dash import Input, Output, callback


@callback(
    [Output("heatmap-settings", "style"), Output("scatter-settings", "style")],
    [Input("visualization-type", "value")],
)
def toggle_visualization_settings(visualization_type):
    """Show/hide appropriate settings based on visualization type."""
    if visualization_type == "heatmap":
        return {"display": "block", "marginTop": "10px"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block", "marginTop": "10px"}


# Callback to toggle visualization settings for Graph 1.5
@callback(
    [
        Output("heatmap-settings-graph15", "style"),
        Output("scatter-settings-graph15", "style"),
    ],
    [Input("visualization-type-graph15", "value")],
)
def toggle_visualization_settings_graph15(visualization_type):
    """Show/hide appropriate settings based on visualization type for Graph 1.5."""
    if visualization_type == "heatmap":
        return {"display": "block", "marginTop": "10px"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block", "marginTop": "10px"}


@callback(
    Output("dbscan-params", "style"),
    Output("kmeans-params", "style"),
    Output("agglomerative-params", "style"),
    Input("clustering-method", "value"),
)
def toggle_clustering_params(method):
    """Show/hide clustering parameters based on selected method."""
    dbscan_style = {"display": "block"} if method == "dbscan" else {"display": "none"}
    kmeans_style = {"display": "block"} if method == "kmeans" else {"display": "none"}
    agglomerative_style = (
        {"display": "block"} if method == "agglomerative" else {"display": "none"}
    )
    return dbscan_style, kmeans_style, agglomerative_style
