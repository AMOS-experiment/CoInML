"""
Simple unit conversion utilities for COLTRIMS data visualization.
Apply conversions only at plot time, keeping original data unchanged.
"""

import numpy as np
import pandas as pd

# Conversion constants
AU_TO_EV = 27.2114  # 1 atomic unit of energy = 27.2114 eV
RAD_TO_DEG = 180.0 / np.pi  # Convert radians to degrees


def convert_feature_for_display(data, feature_name):
    """
    Convert feature data to display units based on feature type.

    Parameters:
    -----------
    data : array-like or pd.Series
        The data to convert
    feature_name : str
        Name of the feature to determine conversion type

    Returns:
    --------
    converted_data : array-like
        Data converted to display units
    unit_label : str
        Unit label for axis
    """
    # Energy features - convert to eV
    if feature_name.startswith("energy_") or feature_name in [
        "KER",
        "EESum",
        "TotalEnergy",
    ]:
        converted_data = np.array(data) * AU_TO_EV
        unit_label = "eV"

    # Angle features - convert to degrees
    elif (
        feature_name.startswith("theta_")
        or feature_name.startswith("phi_")
        or "relative_angle" in feature_name
        or "angle_" in feature_name
    ):
        converted_data = np.array(data) * RAD_TO_DEG
        unit_label = "degrees"

    # Momentum features - keep in atomic units
    elif "mom_mag" in feature_name:
        converted_data = data
        unit_label = "a.u."

    # EESharing is a ratio - no units
    elif feature_name == "EESharing":
        converted_data = data
        unit_label = "ratio"

    # Dot products and other features - keep as is
    else:
        converted_data = data
        unit_label = ""

    return converted_data, unit_label


def format_axis_title(feature_name, unit_label):
    """
    Format axis title with proper feature name and units.

    Parameters:
    -----------
    feature_name : str
        Original feature name
    unit_label : str
        Unit label from conversion

    Returns:
    --------
    str
        Formatted axis title
    """
    # Clean up feature name for display
    display_name = feature_name

    # Special formatting for common features
    name_mapping = {
        "KER": "KER (Kinetic Energy Release)",
        "EESum": "Electron Energy Sum",
        "TotalEnergy": "Total Energy",
        "EESharing": "Electron Energy Sharing",
        "energy_ion1": "Ion 1 Energy",
        "energy_ion2": "Ion 2 Energy",
        "energy_electron1": "Electron 1 Energy",
        "energy_electron2": "Electron 2 Energy",
        "energy_neutral1": "Neutral Energy",
        "theta_ion1": "Ion 1 Polar Angle (θ)",
        "theta_ion2": "Ion 2 Polar Angle (θ)",
        "theta_electron1": "Electron 1 Polar Angle (θ)",
        "theta_electron2": "Electron 2 Polar Angle (θ)",
        "phi_ion1": "Ion 1 Azimuthal Angle (φ)",
        "phi_ion2": "Ion 2 Azimuthal Angle (φ)",
        "phi_electron1": "Electron 1 Azimuthal Angle (φ)",
        "phi_electron2": "Electron 2 Azimuthal Angle (φ)",
        "relative_angle_ion1_ion2": "Ion-Ion Relative Angle",
        "relative_angle_electron1_electron2": "Electron-Electron Relative Angle",
        "mom_mag_ion1": "Ion 1 Momentum",
        "mom_mag_ion2": "Ion 2 Momentum",
        "mom_mag_electron1": "Electron 1 Momentum",
        "mom_mag_electron2": "Electron 2 Momentum",
    }

    if feature_name in name_mapping:
        display_name = name_mapping[feature_name]
    else:
        # Generic cleanup - replace underscores with spaces and capitalize
        display_name = feature_name.replace("_", " ").title()

    # Add units if available
    if unit_label and unit_label != "ratio":
        if unit_label == "degrees":
            return f"{display_name} (°)"
        elif unit_label == "eV":
            return f"{display_name} (eV)"
        elif unit_label == "a.u.":
            return f"{display_name} (a.u.)"
        else:
            return f"{display_name} ({unit_label})"
    else:
        return display_name


def apply_unit_conversions_to_plot(fig, x_feature, y_feature, x_data, y_data):
    """
    Apply unit conversions to plotly figure data and update axis titles.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The plotly figure to update
    x_feature : str
        Name of x-axis feature
    y_feature : str
        Name of y-axis feature
    x_data : array-like
        Original x-axis data
    y_data : array-like
        Original y-axis data

    Returns:
    --------
    plotly.graph_objects.Figure
        Updated figure with converted data and proper axis labels
    """
    # Convert data for both axes
    x_converted, x_unit = convert_feature_for_display(x_data, x_feature)
    y_converted, y_unit = convert_feature_for_display(y_data, y_feature)

    # Update the data in the figure
    for trace in fig.data:
        trace.x = x_converted
        trace.y = y_converted

    # Update axis titles
    fig.update_xaxis(title_text=format_axis_title(x_feature, x_unit))
    fig.update_yaxis(title_text=format_axis_title(y_feature, y_unit))

    return fig


# Integration helper for existing callbacks
def integrate_conversions_in_callback(x_feature, y_feature, combined_df):
    """
    Helper function to integrate conversions in existing plot callbacks.

    Example usage in your existing callbacks:

    # In update_scatter_graph15 or similar callbacks:
    from your_module import integrate_conversions_in_callback

    # Get converted data and labels
    x_data_converted, y_data_converted, x_label, y_label = integrate_conversions_in_callback(
        x_feature, y_feature, combined_df
    )

    # Use converted data in plot
    fig = go.Figure(data=go.Scatter(
        x=x_data_converted,
        y=y_data_converted,
        mode='markers'
    ))

    # Set axis labels
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    """
    # Extract original data
    x_data = combined_df[x_feature]
    y_data = combined_df[y_feature]

    # Convert data
    x_converted, x_unit = convert_feature_for_display(x_data, x_feature)
    y_converted, y_unit = convert_feature_for_display(y_data, y_feature)

    # Format labels
    x_label = format_axis_title(x_feature, x_unit)
    y_label = format_axis_title(y_feature, y_unit)

    return x_converted, y_converted, x_label, y_label


# Quick integration for existing scatter plot callbacks
def update_existing_figure_with_units(fig, x_feature, y_feature, combined_df):
    """
    Quick integration for existing plotly figures.
    Just call this after creating your figure.

    Example:
    fig = px.scatter(combined_df, x=x_feature, y=y_feature)
    fig = update_existing_figure_with_units(fig, x_feature, y_feature, combined_df)
    """
    # Get the original data
    x_data = combined_df[x_feature]
    y_data = combined_df[y_feature]

    # Apply conversions
    return apply_unit_conversions_to_plot(fig, x_feature, y_feature, x_data, y_data)
