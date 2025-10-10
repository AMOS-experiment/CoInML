import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html
from matplotlib.path import Path


# Store selected points when selection changes in Graph 1
@callback(
    Output("selected-points-store", "data"),
    Output("selected-points-info", "children"),
    Input("umap-graph", "selectedData"),
    prevent_initial_call=True,
)
def store_selected_points(selectedData):
    """Store the selected points from Graph 1."""
    if not selectedData:
        return [], "No points selected."

    selection_type = ""
    num_points = 0

    # Handle box selection
    if "range" in selectedData:
        x_range = selectedData["range"]["x"]
        y_range = selectedData["range"]["y"]
        selection_type = f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"

    # Handle lasso selection
    elif "lassoPoints" in selectedData:
        selection_type = "Lasso selection"

    # Handle individual point selection
    if "points" in selectedData:
        num_points = len(selectedData["points"])

    info_text = [
        html.Div(f"Selection type: {selection_type}"),
        html.Div(f"Number of points: {num_points}"),
    ]

    return selectedData, info_text


# Store selected points when selection changes in Graph 3
@callback(
    Output("selected-points-run-store", "data"),
    Output("selected-points-run-info", "children"),
    Input("umap-graph-selected-run", "selectedData"),
    prevent_initial_call=True,
)
def store_selected_points_run(selectedData):
    """Store the selected points from Graph 3."""
    if not selectedData:
        return [], "No points selected."

    selection_type = ""
    num_points = 0

    # Handle box selection
    if "range" in selectedData:
        x_range = selectedData["range"]["x"]
        y_range = selectedData["range"]["y"]
        selection_type = f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"

    # Handle lasso selection
    elif "lassoPoints" in selectedData:
        selection_type = "Lasso selection"

    # Handle individual point selection
    if "points" in selectedData:
        num_points = len(selectedData["points"])

    info_text = [
        html.Div(f"Selection type: {selection_type}"),
        html.Div(f"Number of points: {num_points}"),
    ]

    return selectedData, info_text


# Callback to store selected points from Graph 1.5
@callback(
    Output("selected-points-store-graph15", "data"),
    Output("selected-points-info-graph15", "children"),
    Input("scatter-graph15", "selectedData"),
    prevent_initial_call=True,
)
def store_selected_points_graph15(selectedData):
    """Store the selected points from Graph 1.5."""
    if not selectedData:
        return [], "No points selected."

    selection_type = ""
    num_points = 0

    # Handle box selection
    if "range" in selectedData:
        x_range = selectedData["range"]["x"]
        y_range = selectedData["range"]["y"]
        selection_type = f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"

    # Handle lasso selection
    elif "lassoPoints" in selectedData:
        selection_type = "Lasso selection"

    # Handle individual point selection
    if "points" in selectedData:
        num_points = len(selectedData["points"])

    info_text = [
        html.Div(f"Selection type: {selection_type}"),
        html.Div(f"Number of points: {num_points}"),
    ]

    return selectedData, info_text


# Callback for Graph 3 Selected Points
@callback(
    Output("umap-graph-selected-run-only", "figure"),
    Output("debug-output-selected-run-only", "children"),
    Output("graph3-selection-info-viz", "children"),
    Input("show-selected-run", "n_clicks"),
    State("selected-points-run-store", "data"),
    State("umap-graph-selected-run", "figure"),  # Graph 3 figure for reference
    State("umap-graph", "figure"),  # Graph 1 figure for color consistency
    State("combined-data-store", "data"),  # Add this to access graph3_umap_coords
    prevent_initial_call=True,
)
def update_umap_selected_run_only(
    n_clicks, selectedData, graph3_figure, graph1_figure, combined_data_json
):
    """Display the selected points from Graph 3 using geometric selection approach."""
    try:
        debug_text = ""

        # Validate inputs
        if not selectedData:
            return (
                {},
                "No points selected. Use the lasso or box select tool on Graph 3.",
                "No selection made yet.",
            )

        # Check if we have Graph 3 UMAP coordinates in the data store
        if (
            "graph3_umap_coords" not in combined_data_json
            or combined_data_json["graph3_umap_coords"] == "{}"
        ):
            return (
                {},
                "Graph 3 UMAP coordinates not found. Please re-run Graph 3 first.",
                "No Graph 3 data available.",
            )

        # Load the Graph 3 UMAP coordinates
        graph3_umap_coords = pd.read_json(
            combined_data_json["graph3_umap_coords"], orient="split"
        )
        debug_text += (
            f"Found Graph 3 UMAP coordinates with {len(graph3_umap_coords)} points.<br>"
        )

        # Process the selection using geometric operations - similar to Graph 1
        indices = []

        # Handle box selection
        if "range" in selectedData:
            x_range = selectedData["range"]["x"]
            y_range = selectedData["range"]["y"]
            debug_text += f"Box selection: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], y=[{y_range[0]:.2f}, {y_range[1]:.2f}]<br>"

            # Find points inside the box
            selected_mask = (
                (graph3_umap_coords["UMAP1"] >= x_range[0])
                & (graph3_umap_coords["UMAP1"] <= x_range[1])
                & (graph3_umap_coords["UMAP2"] >= y_range[0])
                & (graph3_umap_coords["UMAP2"] <= y_range[1])
            )
            indices = np.where(selected_mask)[0].tolist()
            debug_text += f"Found {len(indices)} points in box selection.<br>"

        # Handle lasso selection
        elif "lassoPoints" in selectedData:
            debug_text += "Lasso selection detected.<br>"

            # Extract lasso polygon coordinates
            lasso_x = selectedData["lassoPoints"]["x"]
            lasso_y = selectedData["lassoPoints"]["y"]

            # Create a Path object from the lasso points
            lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

            # Check which points are within the lasso path
            points_array = np.column_stack(
                [graph3_umap_coords["UMAP1"], graph3_umap_coords["UMAP2"]]
            )
            inside_lasso = lasso_path.contains_points(points_array)

            # Get indices of points inside the lasso
            indices = np.where(inside_lasso)[0].tolist()
            debug_text += f"Found {len(indices)} points in lasso selection.<br>"

        # As a fallback, try to use points directly if available
        elif "points" in selectedData and selectedData["points"]:
            debug_text += "Direct point selection detected.<br>"

            # For direct point selection, we'll create a temporary dataframe with the points from the selection
            points = selectedData["points"]
            debug_text += f"Found {len(points)} points in direct selection.<br>"

            # Extract point data directly
            # This approach ensures we display exactly what was selected visually
            selected_data = []
            for point in points:
                x = point.get("x", 0)
                y = point.get("y", 0)

                # Try to get the label from different sources
                label = "Unknown"

                # From curve number
                curve_num = point.get("curveNumber", -1)
                if (
                    curve_num >= 0
                    and graph3_figure
                    and "data" in graph3_figure
                    and curve_num < len(graph3_figure["data"])
                ):
                    curve = graph3_figure["data"][curve_num]
                    if "name" in curve:
                        label = curve["name"]
                        if " (" in label:
                            label = label.split(" (")[0]

                # From customdata if available
                if "customdata" in point and point["customdata"]:
                    if (
                        isinstance(point["customdata"], list)
                        and len(point["customdata"]) > 0
                    ):
                        # If we have customdata with label information
                        if isinstance(point["customdata"][0], str):
                            label = point["customdata"][0]

                selected_data.append({"UMAP1": x, "UMAP2": y, "file_label": label})

            # Create a temporary selection dataframe
            temp_df = pd.DataFrame(selected_data)
            debug_text += f"Created temporary dataframe with {len(temp_df)} points.<br>"

            # For consistency with other selection methods, we'll still try to find indices
            # in graph3_umap_coords that match these points as closely as possible
            for i, row in temp_df.iterrows():
                # Find the closest point in graph3_umap_coords
                distances = (graph3_umap_coords["UMAP1"] - row["UMAP1"]) ** 2 + (
                    graph3_umap_coords["UMAP2"] - row["UMAP2"]
                ) ** 2
                closest_idx = distances.idxmin()
                indices.append(closest_idx)

            debug_text += (
                f"Mapped {len(indices)} points to Graph 3 UMAP coordinates.<br>"
            )

        if not indices:
            return {}, "No valid points found in the selection.", "No valid selection."

        # Make sure indices are valid
        valid_indices = [i for i in indices if 0 <= i < len(graph3_umap_coords)]
        if len(valid_indices) < len(indices):
            debug_text += f"Warning: {len(indices) - len(valid_indices)} invalid indices removed.<br>"
            indices = valid_indices

        if not indices:
            return {}, "No valid indices after validation.", "No valid selection."

        # Get the selected points from the Graph 3 UMAP coordinates
        selected_df = graph3_umap_coords.iloc[indices].copy().reset_index(drop=True)
        debug_text += (
            f"Selected {len(selected_df)} points from Graph 3 UMAP coordinates.<br>"
        )

        # Extract color information from both figures for consistent visualization
        color_map = {}

        # Get colors from Graph 1 figure
        if graph1_figure and "data" in graph1_figure:
            for trace in graph1_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    name = trace["name"]
                    if " (" in name:  # Clean label if it contains count
                        name = name.split(" (")[0]
                    color_map[name] = trace["marker"]["color"]

        # Get colors from Graph 3 figure (prioritizing these if there's overlap)
        if graph3_figure and "data" in graph3_figure:
            for trace in graph3_figure["data"]:
                if "name" in trace and "marker" in trace and "color" in trace["marker"]:
                    name = trace["name"]
                    if " (" in name:  # Clean label if it contains count
                        name = name.split(" (")[0]
                    color_map[name] = trace["marker"]["color"]

        # Create the visualization
        fig = go.Figure()

        # Add traces for each label
        for label in selected_df["file_label"].unique():
            mask = selected_df["file_label"] == label
            df_subset = selected_df[mask]

            # Get color for this label
            color = color_map.get(label, None)

            fig.add_trace(
                go.Scatter(
                    x=df_subset["UMAP1"],
                    y=df_subset["UMAP2"],
                    mode="markers",
                    marker=dict(size=8, color=color, opacity=0.7),
                    name=f"{label} ({len(df_subset)} pts)",
                    showlegend=True,
                )
            )

        # Update figure properties
        fig.update_layout(
            height=600,
            title=f"Selected Points from Graph 3 ({len(selected_df)} events)",
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            legend_title="Data File",
            showlegend=True,
        )

        # Count points by file for information panel
        file_counts = selected_df["file_label"].value_counts().to_dict()

        # Create info text
        info_text = [
            html.Div(f"Total selected points: {len(selected_df)}"),
            html.Div([html.Br(), html.B("Points by file:")]),
            html.Div(
                [
                    html.Div(f"{file}: {count} events")
                    for file, count in file_counts.items()
                ],
                style={"marginLeft": "10px"},
            ),
        ]

        return fig, debug_text, info_text

    except Exception as e:
        trace = traceback.format_exc()
        return (
            {},
            f"Error processing Graph 3 selection: {str(e)}<br><pre>{trace}</pre>",
            f"Error: {str(e)}",
        )
