import dash
import plotly.graph_objects as go
from dash import Input, Output, callback


@callback(
    Output("filtered-data-graph", "figure", allow_duplicate=True),
    Input("x-axis-feature-graph15", "value"),
    Input("y-axis-feature-graph15", "value"),
    prevent_initial_call=True,
)
def init_filter_graph(x_feature, y_feature):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    # Check if we have features selected
    if not x_feature or not y_feature:
        # Show placeholder
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4, 5],
                y=[1, 2, 3, 4, 5],
                mode="markers",
                marker=dict(size=10, color="blue"),
            )
        )
        fig.update_layout(
            height=600,
            title="Filter Graph - Select features in Graph 1.5 first",
            xaxis_title="X",
            yaxis_title="Y",
        )
        return fig

    # If we have features, show a message to apply a filter
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text="Apply a filter to see results",
        showarrow=False,
        font=dict(size=20),
    )
    fig.update_layout(
        height=600,
        title=f"Ready to filter: {x_feature} vs {y_feature}",
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig


# TODO: Check this duplicated callback
# # Add a proper initialization callback for the filtered-data-graph
# callback(
#     Output("filtered-data-graph", "figure"),
#     Input("x-axis-feature-graph15", "value"),
#     Input("y-axis-feature-graph15", "value"),
#     prevent_initial_call=True,
# )
# def init_filter_graph(x_feature, y_feature):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         raise dash.exceptions.PreventUpdate

#     # Check if we have features selected
#     if not x_feature or not y_feature:
#         # Show placeholder
#         fig = go.Figure()
#         fig.add_annotation(
#             x=0.5,
#             y=0.5,
#             text="Select features in Graph 1.5 first",
#             showarrow=False,
#             font=dict(size=20),
#         )
#         fig.update_layout(
#             height=600,
#             title="Filter Graph - Select features in Graph 1.5 first",
#             xaxis_title="X",
#             yaxis_title="Y",
#             xaxis=dict(range=[0, 1]),
#             yaxis=dict(range=[0, 1]),
#         )
#         return fig

#     # If we have features, show a message to apply a filter
#     fig = go.Figure()
#     fig.add_annotation(
#         x=0.5,
#         y=0.5,
#         text="Apply a filter to see results",
#         showarrow=False,
#         font=dict(size=20),
#     )
#     fig.update_layout(
#         height=600,
#         title=f"Ready to filter: {x_feature} vs {y_feature}",
#         xaxis_title=x_feature,
#         yaxis_title=y_feature,
#         xaxis=dict(range=[0, 1]),
#         yaxis=dict(range=[0, 1]),
#     )
#     return fig


# Initialize UMAP filtered graph
@callback(
    Output("umap-filtered-data-graph", "figure", allow_duplicate=True),
    Input("umap-graph", "figure"),
    prevent_initial_call=True,
)
def init_umap_filter_graph(umap_figure):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # If UMAP graph exists, show a message to apply a filter
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text="Apply a filter to see results",
        showarrow=False,
        font=dict(size=20),
    )
    fig.update_layout(
        height=600,
        title="Ready to filter UMAP visualization",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig
