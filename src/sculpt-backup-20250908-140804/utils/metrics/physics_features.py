import numpy as np


# TODO: Define feature patterns only once and reuse
def has_physics_features(df):
    """Check if a dataframe has calculated physics features."""
    # List of feature patterns that indicate physics features were calculated
    feature_patterns = [
        "KER",
        "EESum",
        "EESharing",
        "TotalEnergy",
        "mom_mag_",
        "energy_ion",
        "energy_electron",
        "energy_neutral",
        "theta_",
        "phi_",
        "angle_",
        "relative_angle_",
        "dot_product_",
        "cosine_similarity_",
        "momentum_diff_",
        "mom_diff_",
        "phi_diff_",
        "theta_diff_",
        "phi_rel_",
        "theta_rel_",
    ]

    # Check if any column contains any of these patterns
    for col in df.columns:
        for pattern in feature_patterns:
            if pattern in col:
                return True
    return False


def calculate_physics_features_flexible(df, config=None):
    """Calculate physics features with flexible particle configuration including all missing features."""
    try:
        result_df = df.copy()

        # Default configuration
        if config is None:
            config = {
                "num_ions": 2,
                "num_neutrals": 1,
                "num_electrons": 2,
                "ion_masses": [2, 2],  # In amu
                "neutral_masses": [16],  # In amu
                "ion_charges": [1, 1],
            }

        num_ions = config.get("num_ions", 2)
        num_neutrals = config.get("num_neutrals", 1)
        num_electrons = config.get("num_electrons", 2)
        total_particles = num_ions + num_neutrals + num_electrons

        # Get particle masses
        ion_masses = config.get("ion_masses", [2] * num_ions)
        neutral_masses = config.get("neutral_masses", [16] * num_neutrals)

        # Convert to electron masses
        particle_masses = {}
        for i in range(num_ions):
            mass_amu = ion_masses[i] if i < len(ion_masses) else 2
            particle_masses[f"ion{i+1}"] = mass_amu * 1836

        for i in range(num_neutrals):
            mass_amu = neutral_masses[i] if i < len(neutral_masses) else 16
            particle_masses[f"neutral{i+1}"] = mass_amu * 1836

        for i in range(num_electrons):
            particle_masses[f"electron{i+1}"] = 1

        # Calculate momentum magnitudes and store momentum vectors
        particle_idx = 0
        momentum_vectors = []
        momentum_magnitudes = []
        particle_names = []

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
                momentum_vectors.append(p)
                mag = np.linalg.norm(p, axis=1)
                momentum_magnitudes.append(mag)
                particle_names.append(f"ion{i+1}")
                result_df[f"mom_mag_ion{i+1}"] = mag
                result_df[f"energy_ion{i+1}"] = (mag**2) / (
                    2 * particle_masses[f"ion{i+1}"]
                )
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
                momentum_vectors.append(p)
                mag = np.linalg.norm(p, axis=1)
                momentum_magnitudes.append(mag)
                particle_names.append(f"neutral{i+1}")
                result_df[f"mom_mag_neutral{i+1}"] = mag
                result_df[f"energy_neutral{i+1}"] = (mag**2) / (
                    2 * particle_masses[f"neutral{i+1}"]
                )
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
                momentum_vectors.append(p)
                mag = np.linalg.norm(p, axis=1)
                momentum_magnitudes.append(mag)
                particle_names.append(f"electron{i+1}")
                result_df[f"mom_mag_electron{i+1}"] = mag
                result_df[f"energy_electron{i+1}"] = (mag**2) / (
                    2 * particle_masses[f"electron{i+1}"]
                )
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

            # Calculate EESharing (electron energy sharing)
            if num_electrons >= 2:
                result_df["EESharing"] = result_df["energy_electron1"] / (
                    result_df["EESum"] + 1e-8
                )

        # Total energy
        all_energy_cols = [
            col for col in result_df.columns if col.startswith("energy_")
        ]
        if all_energy_cols:
            result_df["TotalEnergy"] = result_df[all_energy_cols].sum(axis=1)

        # Calculate angles for each particle (theta and phi)
        for idx, name in enumerate(particle_names):
            if idx < len(momentum_vectors):
                p_vec = momentum_vectors[idx]
                p_mag = momentum_magnitudes[idx]

                # Calculate theta (polar angle from z-axis)
                cos_theta = np.clip(p_vec[:, 2] / (p_mag + 1e-8), -1.0, 1.0)
                result_df[f"theta_{name}"] = np.arccos(cos_theta)

                # Calculate phi (azimuthal angle in xy-plane)
                result_df[f"phi_{name}"] = np.arctan2(p_vec[:, 1], p_vec[:, 0])

        # Calculate pairwise features
        for i in range(len(particle_names)):
            for j in range(i + 1, len(particle_names)):
                p1_name = particle_names[i]
                p2_name = particle_names[j]

                # Get vectors and magnitudes
                vec1 = momentum_vectors[i]
                vec2 = momentum_vectors[j]
                mag1 = momentum_magnitudes[i]
                mag2 = momentum_magnitudes[j]

                # Particle indices for naming (1-indexed)
                idx1 = i + 1
                idx2 = j + 1

                # Calculate dot products
                dot_products = np.sum(vec1 * vec2, axis=1)
                result_df[f"dot_product_{idx1}{idx2}"] = dot_products

                # Calculate cosine similarity
                cosine_sim = dot_products / (mag1 * mag2 + 1e-8)
                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                result_df[f"cosine_similarity_{idx1}{idx2}"] = cosine_sim

                # Calculate relative angle from cosine similarity
                result_df[f"relative_angle_{idx1}{idx2}"] = np.arccos(cosine_sim)

                # Calculate angle between particles (same as relative_angle but kept for compatibility)
                result_df[f"angle_{p1_name}_{p2_name}"] = np.arccos(cosine_sim)

                # Calculate momentum magnitude difference
                result_df[f"momentum_diff_{idx1}{idx2}"] = np.abs(mag1 - mag2)

                # Calculate phi difference (Δφ)
                phi1 = result_df[f"phi_{p1_name}"].values
                phi2 = result_df[f"phi_{p2_name}"].values
                phi_diff = np.abs(phi1 - phi2)
                phi_diff = np.arctan2(
                    np.sin(phi_diff), np.cos(phi_diff)
                )  # Ensure proper periodicity
                result_df[f"phi_diff_{idx1}{idx2}"] = phi_diff

                # Calculate theta difference (Δθ)
                theta1 = result_df[f"theta_{p1_name}"].values
                theta2 = result_df[f"theta_{p2_name}"].values
                result_df[f"theta_diff_{idx1}{idx2}"] = np.abs(theta1 - theta2)

                # Calculate phi ratio (φ_rel)
                result_df[f"phi_rel_{idx1}{idx2}"] = phi1 / (phi2 + 1e-8)

                # Calculate theta ratio (θ_rel)
                result_df[f"theta_rel_{idx1}{idx2}"] = theta1 / (theta2 + 1e-8)

        return result_df

    except Exception as e:
        print(f"Error calculating physics features: {str(e)}")
        import traceback

        traceback.print_exc()
        return df


def calculate_physics_features_with_profile(df, profile_config):
    """Calculate physics features using a specific configuration profile including all missing features."""
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

        # Store momentum vectors and magnitudes for later use
        momentum_vectors = []
        momentum_magnitudes = []
        feature_prefixes = []

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

                # Store for later use
                momentum_vectors.append(p_vec)
                momentum_magnitudes.append(p_mag)

                # Create feature names based on particle type and name
                if particle["type"] == "ion":
                    feature_prefix = f"{particle['name']}_ion{particle['index']+1}"
                elif particle["type"] == "neutral":
                    feature_prefix = f"{particle['name']}_neutral{particle['index']+1}"
                else:
                    feature_prefix = f"electron{particle['index']+1}"

                feature_prefixes.append(feature_prefix)

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

            # Calculate EESharing
            if len(electron_energy_cols) >= 2:
                # Assuming first electron for sharing calculation
                first_electron_col = electron_energy_cols[0]
                result_df["EESharing"] = result_df[first_electron_col] / (
                    result_df["EESum"] + 1e-8
                )

        # Total energy
        all_energy_cols = [
            col for col in result_df.columns if col.startswith("energy_")
        ]
        if all_energy_cols:
            result_df["TotalEnergy"] = result_df[all_energy_cols].sum(axis=1)

        # Calculate pairwise features between particles
        for i in range(len(particle_list)):
            for j in range(i + 1, len(particle_list)):
                if i < total_particles and j < total_particles:
                    # TODO: Are these needed?
                    # p1 = particle_list[i]
                    # p2 = particle_list[j]

                    prefix1 = feature_prefixes[i]
                    prefix2 = feature_prefixes[j]

                    # Get vectors and magnitudes
                    vec1 = momentum_vectors[i]
                    vec2 = momentum_vectors[j]
                    mag1 = momentum_magnitudes[i]
                    mag2 = momentum_magnitudes[j]

                    # Particle indices for naming (1-indexed)
                    idx1 = i + 1
                    idx2 = j + 1

                    # Calculate dot product
                    dot_product = np.sum(vec1 * vec2, axis=1)
                    result_df[f"dot_product_{prefix1}_{prefix2}"] = dot_product
                    result_df[f"dot_product_{idx1}{idx2}"] = (
                        dot_product  # Also use numeric indices
                    )

                    # Calculate cosine similarity
                    cos_angle = dot_product / (mag1 * mag2 + 1e-8)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    result_df[f"cosine_similarity_{idx1}{idx2}"] = cos_angle

                    # Calculate relative angle
                    angle = np.arccos(cos_angle)
                    result_df[f"angle_{prefix1}_{prefix2}"] = angle
                    result_df[f"relative_angle_{idx1}{idx2}"] = angle

                    # Calculate momentum magnitude difference
                    result_df[f"momentum_diff_{idx1}{idx2}"] = np.abs(mag1 - mag2)

                    # Get phi and theta values
                    phi1 = result_df[f"phi_{prefix1}"].values
                    phi2 = result_df[f"phi_{prefix2}"].values
                    theta1 = result_df[f"theta_{prefix1}"].values
                    theta2 = result_df[f"theta_{prefix2}"].values

                    # Calculate phi difference (Δφ)
                    phi_diff = np.abs(phi1 - phi2)
                    phi_diff = np.arctan2(
                        np.sin(phi_diff), np.cos(phi_diff)
                    )  # Ensure proper periodicity
                    result_df[f"phi_diff_{idx1}{idx2}"] = phi_diff

                    # Calculate theta difference (Δθ)
                    result_df[f"theta_diff_{idx1}{idx2}"] = np.abs(theta1 - theta2)

                    # Calculate phi ratio (φ_rel)
                    result_df[f"phi_rel_{idx1}{idx2}"] = phi1 / (phi2 + 1e-8)

                    # Calculate theta ratio (θ_rel)
                    result_df[f"theta_rel_{idx1}{idx2}"] = theta1 / (theta2 + 1e-8)

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
