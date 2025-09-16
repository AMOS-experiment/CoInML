import dash
from dash import ALL, Input, Output, State, callback, dcc, html


# Initialize with example profiles
@callback(
    Output("configuration-profiles-store", "data"),
    Input("url", "pathname"),  # Triggers on page load
    prevent_initial_call=False,
)
def initialize_profiles(pathname):
    """Initialize with example profiles."""
    return {
        "D2O": {
            "name": "D2O",
            "particle_count": {"ions": 2, "neutrals": 1, "electrons": 2},
            "particles": {
                "ion_0": {
                    "name": "D+",
                    "mass": 2,
                    "charge": 1,
                    "type": "ion",
                    "index": 0,
                },
                "ion_1": {
                    "name": "D+",
                    "mass": 2,
                    "charge": 1,
                    "type": "ion",
                    "index": 1,
                },
                "neutral_0": {
                    "name": "O",
                    "mass": 16,
                    "charge": 0,
                    "type": "neutral",
                    "index": 0,
                },
                "electron_0": {
                    "name": "e-",
                    "mass": 0.000545,
                    "charge": -1,
                    "type": "electron",
                    "index": 0,
                },
                "electron_1": {
                    "name": "e-",
                    "mass": 0.000545,
                    "charge": -1,
                    "type": "electron",
                    "index": 1,
                },
            },
        },
        "HDO": {
            "name": "HDO",
            "particle_count": {"ions": 2, "neutrals": 1, "electrons": 2},
            "particles": {
                "ion_0": {
                    "name": "H+",
                    "mass": 1,
                    "charge": 1,
                    "type": "ion",
                    "index": 0,
                },
                "ion_1": {
                    "name": "D+",
                    "mass": 2,
                    "charge": 1,
                    "type": "ion",
                    "index": 1,
                },
                "neutral_0": {
                    "name": "O",
                    "mass": 16,
                    "charge": 0,
                    "type": "neutral",
                    "index": 0,
                },
                "electron_0": {
                    "name": "e-",
                    "mass": 0.000545,
                    "charge": -1,
                    "type": "electron",
                    "index": 0,
                },
                "electron_1": {
                    "name": "e-",
                    "mass": 0.000545,
                    "charge": -1,
                    "type": "electron",
                    "index": 1,
                },
            },
        },
    }


# Update profile dropdown options
@callback(
    Output("active-profile-dropdown", "options"),
    Input("configuration-profiles-store", "data"),
)
def update_profile_dropdown(profiles_store):
    """Update the profile dropdown options."""
    if not profiles_store:
        return []
    return [{"label": name, "value": name} for name in profiles_store.keys()]


# Generate particle configuration UI
@callback(
    Output("particle-config-container", "children"),
    Input("num-ions", "value"),
    Input("num-neutrals", "value"),
    Input("num-electrons", "value"),
)
def update_particle_config_ui(num_ions, num_neutrals, num_electrons):
    """Generate UI for particle configuration."""
    config_elements = []

    # Create configuration for ions
    if num_ions and num_ions > 0:
        ion_inputs = []
        for i in range(num_ions):
            ion_inputs.append(
                html.Div(
                    [
                        html.Label(
                            f"Ion {i+1}:",
                            style={"width": "80px", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id={"type": "ion-name", "index": i},
                            type="text",
                            placeholder="e.g., H+, D+, O+",
                            value=(
                                "D+" if i < 2 else "Ion"
                            ),  # Default to D+ for first two
                            style={"width": "100px", "marginRight": "10px"},
                        ),
                        html.Label("Mass (amu):", style={"marginLeft": "10px"}),
                        dcc.Input(
                            id={"type": "ion-mass", "index": i},
                            type="number",
                            value=2 if i < 2 else 1,  # Default to 2 for deuterium
                            min=1,
                            style={"width": "80px", "marginLeft": "5px"},
                        ),
                        html.Label("Charge:", style={"marginLeft": "10px"}),
                        dcc.Input(
                            id={"type": "ion-charge", "index": i},
                            type="number",
                            value=1,
                            style={"width": "60px", "marginLeft": "5px"},
                        ),
                    ],
                    style={"marginBottom": "8px"},
                )
            )

        config_elements.append(
            html.Div(
                [
                    html.H5(
                        "Ion Configuration",
                        style={"color": "#1976d2", "marginBottom": "10px"},
                    ),
                    html.Div(
                        ion_inputs,
                        style={
                            "padding": "10px",
                            "backgroundColor": "#e3f2fd",
                            "borderRadius": "5px",
                        },
                    ),
                ]
            )
        )

    # Create configuration for neutrals
    if num_neutrals and num_neutrals > 0:
        neutral_inputs = []
        for i in range(num_neutrals):
            neutral_inputs.append(
                html.Div(
                    [
                        html.Label(
                            f"Neutral {i+1}:",
                            style={"width": "80px", "display": "inline-block"},
                        ),
                        dcc.Input(
                            id={"type": "neutral-name", "index": i},
                            type="text",
                            placeholder="e.g., O, N, C",
                            value=(
                                "O" if i == 0 else "Neutral"
                            ),  # Default to O for first
                            style={"width": "100px", "marginRight": "10px"},
                        ),
                        html.Label("Mass (amu):", style={"marginLeft": "10px"}),
                        dcc.Input(
                            id={"type": "neutral-mass", "index": i},
                            type="number",
                            value=16 if i == 0 else 1,  # Default to 16 for oxygen
                            min=1,
                            style={"width": "80px", "marginLeft": "5px"},
                        ),
                    ],
                    style={"marginBottom": "8px"},
                )
            )

        config_elements.append(
            html.Div(
                [
                    html.H5(
                        "Neutral Configuration",
                        style={
                            "color": "#388e3c",
                            "marginBottom": "10px",
                            "marginTop": "15px",
                        },
                    ),
                    html.Div(
                        neutral_inputs,
                        style={
                            "padding": "10px",
                            "backgroundColor": "#e8f5e9",
                            "borderRadius": "5px",
                        },
                    ),
                ]
            )
        )

    # Note about electrons
    if num_electrons and num_electrons > 0:
        config_elements.append(
            html.Div(
                [
                    html.H5(
                        "Electron Configuration",
                        style={
                            "color": "#f57c00",
                            "marginBottom": "10px",
                            "marginTop": "15px",
                        },
                    ),
                    html.Div(
                        [
                            html.I(f"System includes {num_electrons} electron(s). "),
                            html.I(
                                "Electron mass is fixed at 1/1836 amu (0.000545 amu)"
                            ),
                        ],
                        style={
                            "fontSize": "12px",
                            "color": "gray",
                            "padding": "10px",
                            "backgroundColor": "#fff3e0",
                            "borderRadius": "5px",
                        },
                    ),
                ]
            )
        )

    return config_elements


# Create new profile
@callback(
    Output("configuration-profiles-store", "data", allow_duplicate=True),
    Output("config-profile-name", "value"),
    Output("profile-edit-status", "children"),
    Input("create-profile-btn", "n_clicks"),
    State("config-profile-name", "value"),
    State("configuration-profiles-store", "data"),
    State("num-ions", "value"),
    State("num-neutrals", "value"),
    State("num-electrons", "value"),
    State({"type": "ion-name", "index": ALL}, "value"),
    State({"type": "ion-mass", "index": ALL}, "value"),
    State({"type": "ion-charge", "index": ALL}, "value"),
    State({"type": "neutral-name", "index": ALL}, "value"),
    State({"type": "neutral-mass", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def create_profile(
    n_clicks,
    profile_name,
    profiles_store,
    num_ions,
    num_neutrals,
    num_electrons,
    ion_names,
    ion_masses,
    ion_charges,
    neutral_names,
    neutral_masses,
):
    """Create a new configuration profile with actual particle data."""
    if not n_clicks or not profile_name:
        raise dash.exceptions.PreventUpdate

    # Initialize profiles store if needed
    if profiles_store is None:
        profiles_store = {}

    # Check if profile already exists
    if profile_name in profiles_store:
        return profiles_store, profile_name, "Profile with this name already exists!"

    # Build particles dictionary with actual input values
    particles = {}

    # Process ions with actual input values
    for i in range(num_ions or 0):
        if i < len(ion_names):  # Make sure we have input values
            particles[f"ion_{i}"] = {
                "name": ion_names[i] if ion_names[i] else f"Ion{i+1}",
                "mass": ion_masses[i] if i < len(ion_masses) and ion_masses[i] else 1,
                "charge": (
                    ion_charges[i] if i < len(ion_charges) and ion_charges[i] else 1
                ),
                "type": "ion",
                "index": i,
            }

    # Process neutrals with actual input values
    for i in range(num_neutrals or 0):
        if i < len(neutral_names):  # Make sure we have input values
            particles[f"neutral_{i}"] = {
                "name": neutral_names[i] if neutral_names[i] else f"Neutral{i+1}",
                "mass": (
                    neutral_masses[i]
                    if i < len(neutral_masses) and neutral_masses[i]
                    else 16
                ),
                "charge": 0,
                "type": "neutral",
                "index": i,
            }

    # Add electrons (always with fixed properties)
    for i in range(num_electrons or 0):
        particles[f"electron_{i}"] = {
            "name": "e-",
            "mass": 0.000545,  # Electron mass in amu
            "charge": -1,
            "type": "electron",
            "index": i,
        }

    # Create new profile with actual configuration
    new_profile = {
        "name": profile_name,
        "particle_count": {
            "ions": num_ions or 0,
            "neutrals": num_neutrals or 0,
            "electrons": num_electrons or 0,
        },
        "particles": particles,
    }

    # Add to profiles store
    profiles_store[profile_name] = new_profile

    # Success message
    success_msg = (
        f"Successfully created profile '{profile_name}' with {num_ions} ion(s), {num_neutrals} neutral(s), and "
        "{num_electrons} electron(s)"
    )

    return profiles_store, "", success_msg  # Clear the profile name input


@callback(
    Output("num-ions", "value"),
    Output("num-neutrals", "value"),
    Output("num-electrons", "value"),
    Input("active-profile-dropdown", "value"),
    State("configuration-profiles-store", "data"),
    prevent_initial_call=True,
)
def load_profile_for_editing(selected_profile, profiles_store):
    """Load a profile's particle counts when selected for editing."""
    if (
        not selected_profile
        or not profiles_store
        or selected_profile not in profiles_store
    ):
        raise dash.exceptions.PreventUpdate

    profile = profiles_store[selected_profile]
    particle_count = profile.get("particle_count", {})

    return (
        particle_count.get("ions", 2),
        particle_count.get("neutrals", 1),
        particle_count.get("electrons", 2),
    )


# Update file assignment UI
@callback(
    Output("file-configuration-assignment", "children"),
    Input("stored-files", "data"),
    Input("configuration-profiles-store", "data"),
    State("file-config-assignments-store", "data"),
)
def update_file_assignment_ui(stored_files, profiles_store, assignments_store):
    """Create dropdowns for file-to-profile assignment."""
    if not stored_files:
        return html.Div(
            "Upload files first to assign configurations",
            style={"color": "gray", "fontStyle": "italic"},
        )

    if not profiles_store:
        return html.Div(
            "Create configuration profiles first",
            style={"color": "orange", "fontStyle": "italic"},
        )

    # Create profile options
    profile_options = [{"label": name, "value": name} for name in profiles_store.keys()]
    profile_options.insert(0, {"label": "None (Skip file)", "value": "none"})

    assignment_elements = []

    for f in stored_files:
        if not f.get("is_selection", False):  # Only for data files
            current_value = (
                assignments_store.get(f["filename"], "none")
                if assignments_store
                else "none"
            )

            assignment_elements.append(
                html.Div(
                    [
                        html.Span(
                            f"{f['filename']}: ",
                            style={
                                "fontWeight": "bold",
                                "width": "300px",
                                "display": "inline-block",
                            },
                        ),
                        dcc.Dropdown(
                            id={
                                "type": "file-profile-assignment",
                                "filename": f["filename"],
                            },
                            options=profile_options,
                            value=current_value,
                            style={"width": "250px", "display": "inline-block"},
                        ),
                    ],
                    style={"marginBottom": "10px"},
                )
            )

    return assignment_elements


# Save file assignments
@callback(
    Output("file-config-assignments-store", "data"),
    Output("file-assignment-status", "children"),  # Changed from 'file-config-status'
    Input("apply-file-config-btn", "n_clicks"),
    State({"type": "file-profile-assignment", "filename": ALL}, "value"),
    State({"type": "file-profile-assignment", "filename": ALL}, "id"),
    prevent_initial_call=True,
)
def save_file_assignments(n_clicks, assignments, assignment_ids):
    """Save file-to-profile assignments."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    assignments_dict = {}
    assigned_count = 0

    for i, (assignment, id_info) in enumerate(zip(assignments, assignment_ids)):
        if assignment and assignment != "none":
            filename = id_info["filename"]
            assignments_dict[filename] = assignment
            assigned_count += 1

    return (
        assignments_dict,
        f"Assigned {assigned_count} files to configuration profiles",
    )


# Add this callback after the initialize_profiles callback
@callback(
    Output("active-profiles-display", "children"),
    Input("configuration-profiles-store", "data"),
)
def update_active_profiles_display(profiles_store):
    """Update the display of active profiles."""
    if not profiles_store:
        return html.Div(
            "No profiles created yet", style={"color": "gray", "fontStyle": "italic"}
        )

    profile_cards = []

    for profile_name, profile_data in profiles_store.items():
        particle_count = profile_data.get("particle_count", {})

        # Create a summary of the profile
        particle_summary = []
        if particle_count.get("ions", 0) > 0:
            particle_summary.append(f"{particle_count['ions']} ion(s)")
        if particle_count.get("neutrals", 0) > 0:
            particle_summary.append(f"{particle_count['neutrals']} neutral(s)")
        if particle_count.get("electrons", 0) > 0:
            particle_summary.append(f"{particle_count['electrons']} electron(s)")

        # Get particle details
        particles = profile_data.get("particles", {})
        particle_details = []

        # Group particles by type
        for p_type in ["ion", "neutral", "electron"]:
            type_particles = [p for p in particles.values() if p.get("type") == p_type]
            if type_particles:
                if p_type == "electron":
                    particle_details.append(f"Electrons: {len(type_particles)}")
                else:
                    names = [
                        f"{p.get('name', 'Unknown')} (m={p.get('mass', '?')})"
                        for p in type_particles
                    ]
                    particle_details.append(
                        f"{p_type.capitalize()}s: {', '.join(names)}"
                    )

        # Create profile card
        profile_card = html.Div(
            [
                html.Div(
                    [
                        html.H6(
                            profile_name,
                            style={
                                "margin": "0",
                                "color": "#2e7d32",
                                "fontWeight": "bold",
                            },
                        ),
                        html.Div(
                            ", ".join(particle_summary),
                            style={"fontSize": "14px", "color": "#666"},
                        ),
                        html.Div(
                            particle_details,
                            style={
                                "fontSize": "12px",
                                "color": "#888",
                                "marginTop": "5px",
                            },
                        ),
                    ],
                    style={"padding": "10px"},
                ),
                html.Button(
                    "Delete",
                    id={"type": "delete-profile-btn", "profile": profile_name},
                    n_clicks=0,
                    style={
                        "position": "absolute",
                        "top": "10px",
                        "right": "10px",
                        "padding": "5px 10px",
                        "fontSize": "12px",
                        "backgroundColor": "#f44336",
                        "width": "auto",
                    },
                ),
            ],
            style={
                "position": "relative",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "padding": "10px",
                "marginBottom": "10px",
                "backgroundColor": "white",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            },
        )

        profile_cards.append(profile_card)

    return profile_cards


# Add callback to handle profile deletion
@callback(
    Output("configuration-profiles-store", "data", allow_duplicate=True),
    Input({"type": "delete-profile-btn", "profile": ALL}, "n_clicks"),
    State({"type": "delete-profile-btn", "profile": ALL}, "id"),
    State("configuration-profiles-store", "data"),
    prevent_initial_call=True,
)
def delete_profile(n_clicks_list, button_ids, profiles_store):
    """Delete a profile when delete button is clicked."""
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Extract the profile name from the triggered button
    triggered_id = ctx.triggered[0]["prop_id"]
    if '"profile"' in triggered_id:
        # Parse the profile name from the ID
        import json

        button_id = json.loads(triggered_id.split(".")[0])
        profile_to_delete = button_id["profile"]

        # Delete the profile
        if profile_to_delete in profiles_store:
            del profiles_store[profile_to_delete]

    return profiles_store


@callback(
    [
        Output({"type": "ion-name", "index": ALL}, "value"),
        Output({"type": "ion-mass", "index": ALL}, "value"),
        Output({"type": "ion-charge", "index": ALL}, "value"),
        Output({"type": "neutral-name", "index": ALL}, "value"),
        Output({"type": "neutral-mass", "index": ALL}, "value"),
    ],
    Input("active-profile-dropdown", "value"),
    State("configuration-profiles-store", "data"),
    State("num-ions", "value"),
    State("num-neutrals", "value"),
    State("num-electrons", "value"),
    prevent_initial_call=True,
)
def populate_particle_fields(
    selected_profile, profiles_store, num_ions, num_neutrals, num_electrons
):
    """Populate particle configuration fields when a profile is selected."""
    if (
        not selected_profile
        or not profiles_store
        or selected_profile not in profiles_store
    ):
        raise dash.exceptions.PreventUpdate

    profile = profiles_store[selected_profile]
    particles = profile.get("particles", {})

    # Initialize empty lists for all outputs
    ion_names = []
    ion_masses = []
    ion_charges = []
    neutral_names = []
    neutral_masses = []

    # Populate ion data
    for i in range(num_ions or 0):
        particle_key = f"ion_{i}"
        if particle_key in particles:
            particle = particles[particle_key]
            ion_names.append(particle.get("name", f"Ion{i+1}"))
            ion_masses.append(particle.get("mass", 1))
            ion_charges.append(particle.get("charge", 1))
        else:
            ion_names.append(f"Ion{i+1}")
            ion_masses.append(1)
            ion_charges.append(1)

    # Populate neutral data
    for i in range(num_neutrals or 0):
        particle_key = f"neutral_{i}"
        if particle_key in particles:
            particle = particles[particle_key]
            neutral_names.append(particle.get("name", f"Neutral{i+1}"))
            neutral_masses.append(particle.get("mass", 16))
        else:
            neutral_names.append(f"Neutral{i+1}")
            neutral_masses.append(16)

    return ion_names, ion_masses, ion_charges, neutral_names, neutral_masses


@callback(
    Output("configuration-profiles-store", "data", allow_duplicate=True),
    Output("profile-edit-status", "children", allow_duplicate=True),
    Input("update-profile-btn", "n_clicks"),
    State("active-profile-dropdown", "value"),
    State("configuration-profiles-store", "data"),
    State("num-ions", "value"),
    State("num-neutrals", "value"),
    State("num-electrons", "value"),
    State({"type": "ion-name", "index": ALL}, "value"),
    State({"type": "ion-mass", "index": ALL}, "value"),
    State({"type": "ion-charge", "index": ALL}, "value"),
    State({"type": "neutral-name", "index": ALL}, "value"),
    State({"type": "neutral-mass", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def update_profile(
    n_clicks,
    profile_name,
    profiles_store,
    num_ions,
    num_neutrals,
    num_electrons,
    ion_names,
    ion_masses,
    ion_charges,
    neutral_names,
    neutral_masses,
):
    """Update an existing configuration profile."""
    if (
        not n_clicks
        or not profile_name
        or not profiles_store
        or profile_name not in profiles_store
    ):
        raise dash.exceptions.PreventUpdate

    # Build updated particles dictionary
    particles = {}

    # Process ions
    for i in range(num_ions or 0):
        if i < len(ion_names):
            particles[f"ion_{i}"] = {
                "name": ion_names[i] if ion_names[i] else f"Ion{i+1}",
                "mass": ion_masses[i] if i < len(ion_masses) and ion_masses[i] else 1,
                "charge": (
                    ion_charges[i] if i < len(ion_charges) and ion_charges[i] else 1
                ),
                "type": "ion",
                "index": i,
            }

    # Process neutrals
    for i in range(num_neutrals or 0):
        if i < len(neutral_names):
            particles[f"neutral_{i}"] = {
                "name": neutral_names[i] if neutral_names[i] else f"Neutral{i+1}",
                "mass": (
                    neutral_masses[i]
                    if i < len(neutral_masses) and neutral_masses[i]
                    else 16
                ),
                "charge": 0,
                "type": "neutral",
                "index": i,
            }

    # Add electrons
    for i in range(num_electrons or 0):
        particles[f"electron_{i}"] = {
            "name": "e-",
            "mass": 0.000545,
            "charge": -1,
            "type": "electron",
            "index": i,
        }

    # Update the profile
    profiles_store[profile_name] = {
        "name": profile_name,
        "particle_count": {
            "ions": num_ions or 0,
            "neutrals": num_neutrals or 0,
            "electrons": num_electrons or 0,
        },
        "particles": particles,
    }

    return profiles_store, f"Successfully updated profile '{profile_name}'"
