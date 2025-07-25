import numpy as np


def calculate_physics_features_flexible(df, config=None):
    """Calculate physics features with flexible particle configuration."""
    try:
        # Create a copy of the dataframe
        result_df = df.copy()

        # Get particle configuration
        if config and "particle_count" in config:
            num_ions = config["particle_count"].get("ions", 2)
            num_neutrals = config["particle_count"].get("neutrals", 1)
            num_electrons = config["particle_count"].get("electrons", 2)
        else:
            # Default configuration
            num_ions = 2
            num_neutrals = 1
            num_electrons = 2

        total_particles = num_ions + num_neutrals + num_electrons

        # Get masses from configuration
        particle_masses = {}
        if config and "particles" in config:
            particles_config = config["particles"]

            # Process ions
            for i in range(num_ions):
                mass = particles_config.get(f"ion_{i}", {}).get("mass", 1) * 1836
                particle_masses[f"ion{i+1}"] = mass

            # Process neutrals
            for i in range(num_neutrals):
                mass = particles_config.get(f"neutral_{i}", {}).get("mass", 16) * 1836
                particle_masses[f"neutral{i+1}"] = mass

            # Electrons always have mass 1
            for i in range(num_electrons):
                particle_masses[f"electron{i+1}"] = 1
        else:
            # Default masses
            for i in range(num_ions):
                particle_masses[f"ion{i+1}"] = 2 * 1836  # Default deuterium
            for i in range(num_neutrals):
                particle_masses[f"neutral{i+1}"] = 16 * 1836  # Default oxygen
            for i in range(num_electrons):
                particle_masses[f"electron{i+1}"] = 1

        # Calculate momentum magnitudes for all particles
        particle_idx = 0

        # Process ions
        for i in range(num_ions):
            if particle_idx < total_particles:
                p = df[
                    [
                        f"particle_{particle_idx}_Px",
                        f"particle_{particle_idx}_Py",
                        f"particle_{particle_idx}_Pz",
                    ]
                ].to_numpy()
                result_df[f"mom_mag_ion{i+1}"] = np.linalg.norm(p, axis=1)
                result_df[f"energy_ion{i+1}"] = (
                    result_df[f"mom_mag_ion{i+1}"] ** 2
                ) / (2 * particle_masses[f"ion{i+1}"])
                particle_idx += 1

        # Process neutrals
        for i in range(num_neutrals):
            if particle_idx < total_particles:
                p = df[
                    [
                        f"particle_{particle_idx}_Px",
                        f"particle_{particle_idx}_Py",
                        f"particle_{particle_idx}_Pz",
                    ]
                ].to_numpy()
                result_df[f"mom_mag_neutral{i+1}"] = np.linalg.norm(p, axis=1)
                result_df[f"energy_neutral{i+1}"] = (
                    result_df[f"mom_mag_neutral{i+1}"] ** 2
                ) / (2 * particle_masses[f"neutral{i+1}"])
                particle_idx += 1

        # Process electrons
        for i in range(num_electrons):
            if particle_idx < total_particles:
                p = df[
                    [
                        f"particle_{particle_idx}_Px",
                        f"particle_{particle_idx}_Py",
                        f"particle_{particle_idx}_Pz",
                    ]
                ].to_numpy()
                result_df[f"mom_mag_electron{i+1}"] = np.linalg.norm(p, axis=1)
                result_df[f"energy_electron{i+1}"] = (
                    result_df[f"mom_mag_electron{i+1}"] ** 2
                ) / (2 * particle_masses[f"electron{i+1}"])
                particle_idx += 1

        # Calculate combined energies
        # KER (Kinetic Energy Release) - sum of all ion energies
        ker_cols = [
            f"energy_ion{i+1}"
            for i in range(num_ions)
            if f"energy_ion{i+1}" in result_df.columns
        ]
        if ker_cols:
            result_df["KER"] = result_df[ker_cols].sum(axis=1)

        # Sum of electron energies
        ee_cols = [
            f"energy_electron{i+1}"
            for i in range(num_electrons)
            if f"energy_electron{i+1}" in result_df.columns
        ]
        if ee_cols:
            result_df["EESum"] = result_df[ee_cols].sum(axis=1)

        # Total energy
        all_energy_cols = [
            col for col in result_df.columns if col.startswith("energy_")
        ]
        if all_energy_cols:
            result_df["TotalEnergy"] = result_df[all_energy_cols].sum(axis=1)

        # Calculate angles for each particle
        particle_types = []
        for i in range(num_ions):
            particle_types.append(("ion", i + 1))
        for i in range(num_neutrals):
            particle_types.append(("neutral", i + 1))
        for i in range(num_electrons):
            particle_types.append(("electron", i + 1))

        for idx, (ptype, pnum) in enumerate(particle_types):
            if idx < total_particles:
                particle_name = f"{ptype}{pnum}"
                if f"mom_mag_{particle_name}" in result_df.columns:
                    p_mag = result_df[f"mom_mag_{particle_name}"]
                    p_z = df[f"particle_{idx}_Pz"]

                    # Calculate theta
                    cos_theta = np.clip(p_z / (p_mag + 1e-8), -1.0, 1.0)
                    result_df[f"theta_{particle_name}"] = np.arccos(cos_theta)

                    # Calculate phi
                    p_x = df[f"particle_{idx}_Px"]
                    p_y = df[f"particle_{idx}_Py"]
                    result_df[f"phi_{particle_name}"] = np.arctan2(p_y, p_x)

        # Calculate relative angles between particle pairs
        for i, (ptype1, pnum1) in enumerate(particle_types):
            for j, (ptype2, pnum2) in enumerate(particle_types):
                if i < j and i < total_particles and j < total_particles:
                    p1_name = f"{ptype1}{pnum1}"
                    p2_name = f"{ptype2}{pnum2}"

                    # Extract momentum vectors
                    vec1 = df[
                        [f"particle_{i}_Px", f"particle_{i}_Py", f"particle_{i}_Pz"]
                    ].values
                    vec2 = df[
                        [f"particle_{j}_Px", f"particle_{j}_Py", f"particle_{j}_Pz"]
                    ].values

                    # Calculate dot products
                    dot_products = np.sum(vec1 * vec2, axis=1)

                    # Get magnitudes
                    if (
                        f"mom_mag_{p1_name}" in result_df.columns
                        and f"mom_mag_{p2_name}" in result_df.columns
                    ):
                        mag1 = result_df[f"mom_mag_{p1_name}"].values
                        mag2 = result_df[f"mom_mag_{p2_name}"].values

                        # Calculate angle
                        cos_angle = dot_products / (mag1 * mag2 + 1e-8)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        result_df[f"angle_{p1_name}_{p2_name}"] = np.arccos(cos_angle)
                        result_df[f"dot_product_{p1_name}_{p2_name}"] = dot_products

        return result_df

    except Exception as e:
        print(f"Error calculating physics features: {str(e)}")
        import traceback

        traceback.print_exc()
        return df


def calculate_physics_features_with_profile(df, profile_config):
    """Calculate physics features using a specific configuration profile."""
    try:
        result_df = df.copy()

        if not profile_config:
            # Use default configuration
            return calculate_physics_features_flexible(df, None)

        particle_count = profile_config.get("particle_count", {})
        particles = profile_config.get("particles", {})

        num_ions = particle_count.get("ions", 0)
        num_neutrals = particle_count.get("neutrals", 0)
        num_electrons = particle_count.get("electrons", 0)
        total_particles = num_ions + num_neutrals + num_electrons

        # Build particle list in order
        particle_list = []

        # Add ions
        for i in range(num_ions):
            particle_info = particles.get(f"ion_{i}", {})
            particle_list.append(
                {
                    "type": "ion",
                    "index": i,
                    "name": particle_info.get("name", f"Ion{i+1}"),
                    "mass": particle_info.get("mass", 1)
                    * 1836,  # Convert to electron masses
                    "charge": particle_info.get("charge", 1),
                }
            )

        # Add neutrals
        for i in range(num_neutrals):
            particle_info = particles.get(f"neutral_{i}", {})
            particle_list.append(
                {
                    "type": "neutral",
                    "index": i,
                    "name": particle_info.get("name", f"Neutral{i+1}"),
                    "mass": particle_info.get("mass", 16) * 1836,
                    "charge": 0,
                }
            )

        # Add electrons
        for i in range(num_electrons):
            particle_list.append(
                {
                    "type": "electron",
                    "index": i,
                    "name": "e-",
                    "mass": 1,  # Electron mass
                    "charge": -1,
                }
            )

        # Calculate features for each particle
        for p_idx, particle in enumerate(particle_list):
            if p_idx < total_particles:
                # Get momentum components
                px = df[f"particle_{p_idx}_Px"]
                py = df[f"particle_{p_idx}_Py"]
                pz = df[f"particle_{p_idx}_Pz"]
                p_vec = np.column_stack([px, py, pz])

                # Calculate momentum magnitude
                p_mag = np.linalg.norm(p_vec, axis=1)

                # Create feature names based on particle type and name
                if particle["type"] == "ion":
                    feature_prefix = f"{particle['name']}_ion{particle['index']+1}"
                elif particle["type"] == "neutral":
                    feature_prefix = f"{particle['name']}_neutral{particle['index']+1}"
                else:
                    feature_prefix = f"electron{particle['index']+1}"

                # Store momentum magnitude
                result_df[f"mom_mag_{feature_prefix}"] = p_mag

                # Calculate kinetic energy
                result_df[f"energy_{feature_prefix}"] = (p_mag**2) / (
                    2 * particle["mass"]
                )

                # Calculate angles
                cos_theta = np.clip(pz / (p_mag + 1e-8), -1.0, 1.0)
                result_df[f"theta_{feature_prefix}"] = np.arccos(cos_theta)
                result_df[f"phi_{feature_prefix}"] = np.arctan2(py, px)

        # Calculate combined energies based on particle types
        # KER - sum of ion energies
        ion_energy_cols = [
            col
            for col in result_df.columns
            if col.startswith("energy_") and "_ion" in col
        ]
        if ion_energy_cols:
            result_df["KER"] = result_df[ion_energy_cols].sum(axis=1)

        # Electron energy sum
        electron_energy_cols = [
            col for col in result_df.columns if col.startswith("energy_electron")
        ]
        if electron_energy_cols:
            result_df["EESum"] = result_df[electron_energy_cols].sum(axis=1)

        # Total energy
        all_energy_cols = [
            col for col in result_df.columns if col.startswith("energy_")
        ]
        if all_energy_cols:
            result_df["TotalEnergy"] = result_df[all_energy_cols].sum(axis=1)

        # Calculate relative angles between particles
        for i in range(len(particle_list)):
            for j in range(i + 1, len(particle_list)):
                if i < total_particles and j < total_particles:
                    p1 = particle_list[i]
                    p2 = particle_list[j]

                    # Get feature prefixes
                    if p1["type"] == "ion":
                        prefix1 = f"{p1['name']}_ion{p1['index']+1}"
                    elif p1["type"] == "neutral":
                        prefix1 = f"{p1['name']}_neutral{p1['index']+1}"
                    else:
                        prefix1 = f"electron{p1['index']+1}"

                    if p2["type"] == "ion":
                        prefix2 = f"{p2['name']}_ion{p2['index']+1}"
                    elif p2["type"] == "neutral":
                        prefix2 = f"{p2['name']}_neutral{p2['index']+1}"
                    else:
                        prefix2 = f"electron{p2['index']+1}"

                    # Calculate relative angle
                    vec1 = df[
                        [f"particle_{i}_Px", f"particle_{i}_Py", f"particle_{i}_Pz"]
                    ].values
                    vec2 = df[
                        [f"particle_{j}_Px", f"particle_{j}_Py", f"particle_{j}_Pz"]
                    ].values

                    dot_product = np.sum(vec1 * vec2, axis=1)
                    mag1 = result_df[f"mom_mag_{prefix1}"].values
                    mag2 = result_df[f"mom_mag_{prefix2}"].values

                    cos_angle = dot_product / (mag1 * mag2 + 1e-8)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)

                    result_df[f"angle_{prefix1}_{prefix2}"] = np.arccos(cos_angle)
                    result_df[f"dot_product_{prefix1}_{prefix2}"] = dot_product

        return result_df

    except Exception as e:
        print(f"Error calculating physics features with profile: {str(e)}")
        import traceback

        traceback.print_exc()
        return df


def calculate_physics_features(df, config=None):
    """Wrapper function for backward compatibility."""
    if config and "particles" in config:
        return calculate_physics_features_with_profile(df, config)
    else:
        return calculate_physics_features_flexible(df, config)
