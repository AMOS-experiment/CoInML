import dash
from dash import Input, Output, callback


# Consolidated callback for status updates - updated function signature and implementation
@callback(
    Output("run-umap-status", "children"),
    Output("show-selected-status", "children"),
    Output("run-umap-selected-run-status", "children"),
    Output("plot-custom-features-status", "children"),
    Output("save-selection-status", "children"),
    Output("show-selected-run-status", "children"),
    Output("save-selection-run-status", "children"),
    Output("run-umap-graph3-selection-status", "children"),
    Output("save-selection-graph3-selection-status", "children"),
    Output("train-autoencoder-status", "children"),
    Output("run-umap-latent-status", "children"),
    Output("save-latent-features-status", "children"),
    Output("genetic-features-status", "children"),
    Output("run-umap-genetic-status", "children", allow_duplicate=True),
    Output("save-genetic-features-status", "children"),
    Output("mi-features-status", "children"),
    Output("run-umap-mi-status", "children"),
    Output("save-mi-features-status", "children"),
    Output("generate-plot-graph15-status", "children"),
    Output("save-selection-graph15-status", "children"),
    Output("show-selected-graph15-status", "children"),
    Output("save-selection-graph25-status", "children"),
    Input("run-umap", "n_clicks"),
    Input("umap-graph", "figure"),
    Input("show-selected", "n_clicks"),
    Input("umap-graph-selected-only", "figure"),
    Input("run-umap-selected-run", "n_clicks"),
    Input("umap-graph-selected-run", "figure"),
    Input("plot-custom-features", "n_clicks"),
    Input("custom-feature-plot", "figure"),
    Input("save-selection-btn", "n_clicks"),
    Input("download-selection", "data"),
    Input("show-selected-run", "n_clicks"),
    Input("save-selection-run-btn", "n_clicks"),
    Input("download-selection-run", "data"),
    Input("run-umap-graph3-selection", "n_clicks"),
    Input("umap-graph-graph3-selection", "figure"),
    Input("save-selection-graph3-selection-btn", "n_clicks"),
    Input("download-selection-graph3-selection", "data"),
    Input("train-autoencoder", "n_clicks"),
    Input("autoencoder-umap-graph", "figure"),
    Input("run-umap-latent", "n_clicks"),
    Input("save-latent-features-btn", "n_clicks"),
    Input("download-latent-features", "data"),
    Input("run-genetic-features", "n_clicks"),
    Input("genetic-features-graph", "figure"),
    Input("run-umap-genetic", "n_clicks"),
    Input("save-genetic-features-btn", "n_clicks"),
    Input("download-genetic-features", "data"),
    Input("run-mi-features", "n_clicks"),
    Input("mi-features-graph", "figure"),
    Input("run-umap-mi", "n_clicks"),
    Input("save-mi-features-btn", "n_clicks"),
    Input("download-mi-features", "data"),
    Input("generate-plot-graph15", "n_clicks"),
    Input("scatter-graph15", "figure"),
    Input("save-selection-graph15-btn", "n_clicks"),
    Input("download-selection-graph15", "data"),
    Input("show-selected-graph15", "n_clicks"),
    Input("graph25", "figure"),
    Input("save-selection-graph25-btn", "n_clicks"),
    Input("download-selection-graph25", "data"),
    prevent_initial_call=True,
)
def update_all_status(
    run_umap_clicks,
    umap_fig,
    show_selected_clicks,
    selected_fig,
    run_umap_sel_clicks,
    sel_run_fig,
    plot_features_clicks,
    custom_plot_fig,
    save_selection_clicks,
    download_data,
    show_selected_run_clicks,
    save_selection_run_clicks,
    download_run_data,
    run_umap_graph3_sel_clicks,
    graph3_sel_fig,
    save_selection_graph3_sel_clicks,
    download_graph3_sel_data,
    train_autoencoder_clicks,
    autoencoder_fig,
    run_umap_latent_clicks,
    save_latent_clicks,
    download_latent_data,
    run_genetic_features_clicks,
    genetic_features_fig,
    run_umap_genetic_clicks,
    save_genetic_features_clicks,
    download_genetic_features_data,
    run_mi_features_clicks,
    mi_features_fig,
    run_umap_mi_clicks,
    save_mi_features_clicks,
    download_mi_features_data,
    generate_plot_graph15_clicks,
    scatter_graph15_fig,
    save_selection_graph15_clicks,
    download_selection_graph15_data,
    show_selected_graph15_clicks,
    graph25_figure,
    save_selection_graph25_clicks,
    download_selection_graph25_data,
):
    """Consolidated callback to update all status messages."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Initialize all status values to dash.no_update
    statuses = [dash.no_update] * 22  # Total number of status outputs

    # Handle run-umap status
    if trigger_id == "run-umap":
        statuses[0] = "Running UMAP analysis..."
    elif trigger_id == "umap-graph":
        statuses[0] = "UMAP analysis complete!"

    # Handle show-selected status
    elif trigger_id == "show-selected":
        statuses[1] = "Processing selection..."
    elif trigger_id == "umap-graph-selected-only":
        statuses[1] = "Selection displayed!"

    # Handle run-umap-selected-run status
    elif trigger_id == "run-umap-selected-run":
        statuses[2] = "Re-running UMAP on selection..."
    elif trigger_id == "umap-graph-selected-run":
        statuses[2] = "UMAP re-run complete!"

    # Handle plot-custom-features status
    elif trigger_id == "plot-custom-features":
        statuses[3] = "Creating custom feature plot..."
    elif trigger_id == "custom-feature-plot":
        statuses[3] = "Custom feature plot complete!"

    # Handle save-selection status
    elif trigger_id == "save-selection-btn":
        statuses[4] = "Preparing selection for download..."
    elif trigger_id == "download-selection":
        statuses[4] = "Selection saved successfully!"

    # Handle show-selected-run status
    elif trigger_id == "show-selected-run":
        statuses[5] = "Processing Graph 3 selection..."

    # Handle save-selection-run status
    elif trigger_id == "save-selection-run-btn":
        statuses[6] = "Preparing Graph 3 selection for download..."
    elif trigger_id == "download-selection-run":
        statuses[6] = "Graph 3 selection saved successfully!"

    # Handle run-umap-graph3-selection status
    elif trigger_id == "run-umap-graph3-selection":
        statuses[7] = "Running UMAP on Graph 3 selection..."
    elif trigger_id == "umap-graph-graph3-selection":
        statuses[7] = "UMAP on Graph 3 selection complete!"

    # Handle save-selection-graph3-selection status
    elif trigger_id == "save-selection-graph3-selection-btn":
        statuses[8] = "Preparing Graph 3 selection UMAP for download..."
    elif trigger_id == "download-selection-graph3-selection":
        statuses[8] = "Graph 3 selection UMAP saved successfully!"

    # Handle autoencoder statuses
    elif trigger_id == "train-autoencoder":
        statuses[9] = "Training autoencoder... This may take a while."
    elif trigger_id == "autoencoder-umap-graph":
        statuses[9] = "Autoencoder training complete!"

    # Handle run-umap-latent status
    elif trigger_id == "run-umap-latent":
        statuses[10] = "Running UMAP on latent space..."

    # Handle save-latent-features status
    elif trigger_id == "save-latent-features-btn":
        statuses[11] = "Preparing latent features for download..."
    elif trigger_id == "download-latent-features":
        statuses[11] = "Latent features saved successfully!"

    # Handle genetic feature statuses
    elif trigger_id == "run-genetic-features":
        statuses[12] = "Running genetic feature discovery... This may take a while."
    elif trigger_id == "genetic-features-graph":
        statuses[12] = "Genetic feature discovery complete!"

    # Handle run-umap-genetic status
    elif trigger_id == "run-umap-genetic":
        statuses[13] = "Running UMAP on genetic features..."

    # Handle save-genetic-features status
    elif trigger_id == "save-genetic-features-btn":
        statuses[14] = "Preparing genetic features for download..."
    elif trigger_id == "download-genetic-features":
        statuses[14] = "Genetic features saved successfully!"

    # Handle MI feature statuses
    elif trigger_id == "run-mi-features":
        statuses[15] = (
            "Running mutual information feature selection... This may take a while."
        )
    elif trigger_id == "mi-features-graph":
        statuses[15] = "Mutual information feature selection complete!"

    # Handle run-umap-mi status
    elif trigger_id == "run-umap-mi":
        statuses[16] = "Running UMAP on MI-selected features..."

    # Handle save-mi-features status
    elif trigger_id == "save-mi-features-btn":
        statuses[17] = "Preparing MI features for download..."
    elif trigger_id == "download-mi-features":
        statuses[17] = "MI features saved successfully!"

    # Handle Graph 1.5 status updates
    elif trigger_id == "generate-plot-graph15":
        statuses[18] = "Generating custom scatter plot..."
    elif trigger_id == "scatter-graph15":
        statuses[18] = "Custom scatter plot generated!"
    elif trigger_id == "save-selection-graph15-btn":
        statuses[19] = "Preparing Graph 1.5 selection for download..."
    elif trigger_id == "download-selection-graph15":
        statuses[19] = "Graph 1.5 selection saved successfully!"

    # Handle Graph 2.5 status updates
    elif trigger_id == "show-selected-graph15":
        statuses[20] = "Processing Graph 1.5 selection..."
    elif trigger_id == "graph25":
        statuses[20] = "Selection displayed!"
    elif trigger_id == "save-selection-graph25-btn":
        statuses[21] = "Preparing Graph 2.5 selection for download..."
    elif trigger_id == "download-selection-graph25":
        statuses[21] = "Graph 2.5 selection saved successfully!"

    return tuple(statuses)
