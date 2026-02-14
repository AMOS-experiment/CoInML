"""
Simple unit conversion utilities for COLTRIMS data visualization.
Apply conversions only at plot time, keeping original data unchanged.
"""

import re

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


def _generate_display_name(feature_name):
    """
    Generate a human-readable display name from a feature name using pattern matching.
    Works with any number of particles, not just the default 2-ion/1-neutral/2-electron config.

    Parameters:
    -----------
    feature_name : str
        The internal feature name (e.g., 'energy_ion3', 'theta_electron4')

    Returns:
    --------
    str
        Human-readable display name
    """
    # Exact-match special names (configuration-independent summary features)
    exact_names = {
        "KER": "KER (Kinetic Energy Release)",
        "EESum": "Electron Energy Sum",
        "TotalEnergy": "Total Energy",
        "EESharing": "Electron Energy Sharing",
    }
    if feature_name in exact_names:
        return exact_names[feature_name]

    # Pattern-based rules: (regex, format_function)
    # Each rule tries to match and produce a nice display name
    patterns = [
        # energy_ion1, energy_D+_ion2, energy_electron3, energy_O_neutral1
        (r"^energy_(?:(.+)_)?(ion|neutral|electron)(\d+)$",
         lambda m: f"{m.group(2).capitalize()} {m.group(3)}{' (' + m.group(1) + ')' if m.group(1) else ''} Energy"),
        # mom_mag_ion1, mom_mag_D+_ion2, mom_mag_electron3
        (r"^mom_mag_(?:(.+)_)?(ion|neutral|electron)(\d+)$",
         lambda m: f"{m.group(2).capitalize()} {m.group(3)}{' (' + m.group(1) + ')' if m.group(1) else ''} Momentum"),
        # theta_ion1, theta_D+_ion2, theta_electron3
        (r"^theta_(?:(.+)_)?(ion|neutral|electron)(\d+)$",
         lambda m: f"{m.group(2).capitalize()} {m.group(3)}{' (' + m.group(1) + ')' if m.group(1) else ''} Polar Angle (θ)"),
        # phi_ion1, phi_D+_ion2, phi_electron3
        (r"^phi_(?:(.+)_)?(ion|neutral|electron)(\d+)$",
         lambda m: f"{m.group(2).capitalize()} {m.group(3)}{' (' + m.group(1) + ')' if m.group(1) else ''} Azimuthal Angle (φ)"),
        # angle_ion1_ion2, angle_electron1_electron2 (named pairs)
        (r"^angle_(\w+)_(\w+)$",
         lambda m: f"{m.group(1).replace('_', ' ').title()}-{m.group(2).replace('_', ' ').title()} Angle"),
        # relative_angle_12, relative_angle_34
        (r"^relative_angle_(\d+)(\d+)$",
         lambda m: f"Relative Angle (Particle {m.group(1)}-{m.group(2)})"),
        # dot_product_12
        (r"^dot_product_(\d+)(\d+)$",
         lambda m: f"Dot Product (Particle {m.group(1)}-{m.group(2)})"),
        # cosine_similarity_12
        (r"^cosine_similarity_(\d+)(\d+)$",
         lambda m: f"Cosine Similarity (Particle {m.group(1)}-{m.group(2)})"),
        # momentum_diff_12
        (r"^momentum_diff_(\d+)(\d+)$",
         lambda m: f"Momentum Difference (Particle {m.group(1)}-{m.group(2)})"),
        # phi_diff_12, theta_diff_12
        (r"^(phi|theta)_diff_(\d+)(\d+)$",
         lambda m: f"{'Azimuthal' if m.group(1) == 'phi' else 'Polar'} Angle Difference (Particle {m.group(2)}-{m.group(3)})"),
        # phi_rel_12, theta_rel_12
        (r"^(phi|theta)_rel_(\d+)(\d+)$",
         lambda m: f"{'Azimuthal' if m.group(1) == 'phi' else 'Polar'} Angle Ratio (Particle {m.group(2)}-{m.group(3)})"),
    ]

    for pattern, formatter in patterns:
        match = re.match(pattern, feature_name)
        if match:
            return formatter(match)

    # Fallback: generic cleanup - replace underscores with spaces and capitalize
    return feature_name.replace("_", " ").title()


def format_axis_title(feature_name, unit_label):
    """
    Format axis title with proper feature name and units.
    Uses pattern-based name generation to handle any particle configuration.

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
    display_name = _generate_display_name(feature_name)

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


def generate_feature_label(feature_name):
    """
    Generate a user-friendly label for a feature name, suitable for dropdown menus.
    Combines the display name with units.

    Parameters:
    -----------
    feature_name : str
        Internal feature name

    Returns:
    --------
    str
        Label with display name and units
    """
    display_name = _generate_display_name(feature_name)
    _, unit_label = convert_feature_for_display([0], feature_name)

    if unit_label and unit_label not in ("", "ratio"):
        unit_str = {"degrees": "°", "eV": "eV", "a.u.": "a.u."}.get(
            unit_label, unit_label
        )
        return f"{display_name} ({unit_str})"
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