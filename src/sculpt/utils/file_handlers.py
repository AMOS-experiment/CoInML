import base64
import io
import os
import re

import numpy as np
import pandas as pd
from matplotlib.path import Path  # For lasso selection


# =============================================================================
# FLEXIBLE COLUMN DETECTION
# =============================================================================

# Default required columns for backward compatibility (2 ions + 1 neutral + 2 electrons)
DEFAULT_REQUIRED_COLUMNS = [
    "Px_ion1", "Py_ion1", "Pz_ion1",
    "Px_ion2", "Py_ion2", "Pz_ion2",
    "Px_neutral", "Py_neutral", "Pz_neutral",
    "Px_electron1", "Py_electron1", "Pz_electron1",
    "Px_electron2", "Py_electron2", "Pz_electron2",
]


def detect_particle_columns(df):
    """
    Detect particle momentum columns in the dataframe.
    
    Supports multiple naming conventions:
    - Standard: Px_ion1, Py_ion1, Pz_ion1, Px_ion2, ..., Px_neutral, ..., Px_electron1, ...
    - Numbered neutrals: Px_neutral1, Py_neutral1, Pz_neutral1, Px_neutral2, ...
    - Generic: Px_particle1, Py_particle1, Pz_particle1, ...
    
    Returns:
        dict: Dictionary with particle configuration info:
            - 'ions': list of (name, [Px_col, Py_col, Pz_col])
            - 'neutrals': list of (name, [Px_col, Py_col, Pz_col])
            - 'electrons': list of (name, [Px_col, Py_col, Pz_col])
            - 'other': list of (name, [Px_col, Py_col, Pz_col]) for generic particles
            - 'total_particles': int
            - 'column_order': list of all momentum columns in order
    """
    columns = df.columns.tolist()
    
    ions = []
    neutrals = []
    electrons = []
    other_particles = []
    
    # Patterns for different particle types
    # Ion patterns: Px_ion1, Px_ion2, etc.
    ion_pattern = re.compile(r'^P([xyz])_ion(\d+)$', re.IGNORECASE)
    
    # Neutral patterns: Px_neutral, Px_neutral1, Px_neutral2, etc.
    neutral_pattern = re.compile(r'^P([xyz])_neutral(\d*)$', re.IGNORECASE)
    
    # Electron patterns: Px_electron1, Px_electron2, etc.
    electron_pattern = re.compile(r'^P([xyz])_electron(\d+)$', re.IGNORECASE)
    
    # Generic particle pattern: Px_particle1, Px_particle2, etc.
    generic_pattern = re.compile(r'^P([xyz])_particle(\d+)$', re.IGNORECASE)
    
    # Track which particles we've found
    found_ions = {}  # {ion_number: {'Px': col, 'Py': col, 'Pz': col}}
    found_neutrals = {}
    found_electrons = {}
    found_generic = {}
    
    for col in columns:
        # Check for ions
        match = ion_pattern.match(col)
        if match:
            component, num = match.groups()
            num = int(num)
            if num not in found_ions:
                found_ions[num] = {}
            found_ions[num][f'P{component.lower()}'] = col
            continue
        
        # Check for neutrals
        match = neutral_pattern.match(col)
        if match:
            component, num = match.groups()
            # Handle both "neutral" (no number) and "neutral1", "neutral2", etc.
            num = int(num) if num else 1
            if num not in found_neutrals:
                found_neutrals[num] = {}
            found_neutrals[num][f'P{component.lower()}'] = col
            continue
        
        # Check for electrons
        match = electron_pattern.match(col)
        if match:
            component, num = match.groups()
            num = int(num)
            if num not in found_electrons:
                found_electrons[num] = {}
            found_electrons[num][f'P{component.lower()}'] = col
            continue
        
        # Check for generic particles
        match = generic_pattern.match(col)
        if match:
            component, num = match.groups()
            num = int(num)
            if num not in found_generic:
                found_generic[num] = {}
            found_generic[num][f'P{component.lower()}'] = col
            continue
    
    # Build ordered lists of complete particles (must have Px, Py, Pz)
    column_order = []
    
    # Process ions in numerical order
    for num in sorted(found_ions.keys()):
        components = found_ions[num]
        if all(k in components for k in ['Px', 'Py', 'Pz']):
            cols = [components['Px'], components['Py'], components['Pz']]
            ions.append((f'ion{num}', cols))
            column_order.extend(cols)
    
    # Process neutrals in numerical order
    for num in sorted(found_neutrals.keys()):
        components = found_neutrals[num]
        if all(k in components for k in ['Px', 'Py', 'Pz']):
            cols = [components['Px'], components['Py'], components['Pz']]
            name = f'neutral{num}' if len(found_neutrals) > 1 else 'neutral'
            neutrals.append((name, cols))
            column_order.extend(cols)
    
    # Process electrons in numerical order
    for num in sorted(found_electrons.keys()):
        components = found_electrons[num]
        if all(k in components for k in ['Px', 'Py', 'Pz']):
            cols = [components['Px'], components['Py'], components['Pz']]
            electrons.append((f'electron{num}', cols))
            column_order.extend(cols)
    
    # Process generic particles in numerical order
    for num in sorted(found_generic.keys()):
        components = found_generic[num]
        if all(k in components for k in ['Px', 'Py', 'Pz']):
            cols = [components['Px'], components['Py'], components['Pz']]
            other_particles.append((f'particle{num}', cols))
            column_order.extend(cols)
    
    total_particles = len(ions) + len(neutrals) + len(electrons) + len(other_particles)
    
    return {
        'ions': ions,
        'neutrals': neutrals,
        'electrons': electrons,
        'other': other_particles,
        'total_particles': total_particles,
        'column_order': column_order,
        'num_ions': len(ions),
        'num_neutrals': len(neutrals),
        'num_electrons': len(electrons),
        'num_other': len(other_particles),
    }


def validate_coltrims_file(df):
    """
    Validate that the dataframe contains valid COLTRIMS data.
    
    Returns:
        tuple: (is_valid, particle_config, error_message)
    """
    particle_config = detect_particle_columns(df)
    
    # Check if we found any particles
    if particle_config['total_particles'] == 0:
        # Try to check for default columns (backward compatibility)
        missing_cols = [col for col in DEFAULT_REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, None, f"No valid particle columns found. Missing: {missing_cols[:5]}..."
        else:
            # Has default columns
            return True, None, None  # None config means use default
    
    # Validate that we have at least one complete particle
    if particle_config['total_particles'] < 1:
        return False, None, "No complete particle momentum vectors found (need Px, Py, Pz)"
    
    return True, particle_config, None


# =============================================================================
# FILE STORAGE HELPERS
# =============================================================================

def store_file_data(file_id, data):
    """Store file data in a temporary file and return the path."""
    import tempfile

    # Create a temporary directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "dash_coltrims_data")
    os.makedirs(temp_dir, exist_ok=True)

    # Save the data to a temporary file
    temp_path = os.path.join(temp_dir, f"file_{file_id}.json")
    data.to_json(temp_path, orient="split")

    return temp_path


def load_file_data(file_path):
    """Load file data from temporary storage."""
    if os.path.exists(file_path):
        return pd.read_json(file_path, orient="split")
    return None


# =============================================================================
# FILE TYPE DETECTION
# =============================================================================

def is_selection_file(df):
    """Check if the dataframe looks like a saved selection."""
    # Check for UMAP coordinates which would indicate a selection file
    return (
        "UMAP1" in df.columns and "UMAP2" in df.columns and "file_label" in df.columns
    )


# =============================================================================
# MAIN PARSING FUNCTIONS
# =============================================================================

def parse_contents(contents, filename):
    """
    Parse the uploaded file contents.
    
    Supports flexible particle configurations - any number of ions, neutrals, 
    and electrons as long as they follow the naming convention:
    - Px_ion1, Py_ion1, Pz_ion1, Px_ion2, ...
    - Px_neutral, Py_neutral, Pz_neutral (or Px_neutral1, Px_neutral2, ...)
    - Px_electron1, Py_electron1, Pz_electron1, Px_electron2, ...
    
    Returns:
        tuple: (dataframe, is_selection_file) or (None, False) on error
    """
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        print(f"Decoding {filename}...")

        # Get a sample of the file for debugging
        sample = decoded[:1000].decode("utf-8", errors="replace")
        print(f"Sample of file {filename}: {sample[:100]}...")

        # Try to infer the separator (space or comma)
        first_line = decoded.decode("utf-8", errors="replace").split("\n")[0]
        if "," in first_line:
            sep = ","
            print(f"Detected comma separator for {filename}")
        else:
            sep = None  # Let pandas figure it out (handles multiple spaces)
            print(f"Detected whitespace separator for {filename}")

        # Try to read with pandas
        df = None
        parse_errors = []
        
        # Attempt 1: Use detected separator
        try:
            if sep == ",":
                df = pd.read_csv(
                    io.StringIO(decoded.decode("utf-8", errors="replace")), 
                    sep=sep
                )
            else:
                # Use delim_whitespace for space-separated files
                df = pd.read_csv(
                    io.StringIO(decoded.decode("utf-8", errors="replace")),
                    delim_whitespace=True
                )
            print(f"Successfully read {filename}, shape: {df.shape}")
        except Exception as e:
            parse_errors.append(f"Primary parse failed: {e}")
            
        # Attempt 2: Try alternative separator if first attempt failed
        if df is None:
            try:
                alt_sep = " " if sep == "," else ","
                df = pd.read_csv(
                    io.StringIO(decoded.decode("utf-8", errors="replace")),
                    sep=alt_sep
                )
                print(f"Successfully read {filename} with alternative separator, shape: {df.shape}")
            except Exception as e:
                parse_errors.append(f"Alternative parse failed: {e}")
        
        # Attempt 3: Try with flexible whitespace
        if df is None:
            try:
                df = pd.read_csv(
                    io.StringIO(decoded.decode("utf-8", errors="replace")),
                    sep=r'\s+',
                    engine='python'
                )
                print(f"Successfully read {filename} with regex separator, shape: {df.shape}")
            except Exception as e:
                parse_errors.append(f"Regex parse failed: {e}")
        
        if df is None:
            print(f"All parsing attempts failed for {filename}: {parse_errors}")
            return None, False
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Check if this is a saved selection file
        if is_selection_file(df):
            print(f"{filename} is a selection file")
            return df, True
        
        # Validate as COLTRIMS file
        is_valid, particle_config, error_msg = validate_coltrims_file(df)
        
        if not is_valid:
            print(f"Warning: File {filename} validation failed: {error_msg}")
            return None, False
        
        # Store particle configuration in dataframe attributes for later use
        if particle_config:
            print(f"{filename} detected configuration: {particle_config['num_ions']} ions, "
                  f"{particle_config['num_neutrals']} neutrals, {particle_config['num_electrons']} electrons")
        else:
            print(f"{filename} using default 5-particle configuration")
        
        print(f"{filename} is a valid COLTRIMS file")
        return df, False

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def reorganize_df(df, particle_config=None):
    """
    Reorganize the dataframe with standardized column names.
    
    Converts various column naming conventions to standardized format:
    particle_0_Px, particle_0_Py, particle_0_Pz, particle_1_Px, ...
    
    The order is: ions first, then neutrals, then electrons, then other particles.
    
    Args:
        df: Input dataframe with momentum columns
        particle_config: Optional particle configuration dict from detect_particle_columns()
                        If None, will auto-detect or use default columns
    
    Returns:
        DataFrame with standardized column names
    """
    # If no config provided, detect it
    if particle_config is None:
        is_valid, particle_config, error_msg = validate_coltrims_file(df)
        
        if not is_valid or particle_config is None:
            # Fall back to default column handling for backward compatibility
            if all(col in df.columns for col in DEFAULT_REQUIRED_COLUMNS):
                print("Using default column mapping (backward compatibility)")
                df_subset = df[DEFAULT_REQUIRED_COLUMNS].copy()
                new_columns = []
                for i in range(5):
                    new_columns.extend([f"particle_{i}_Px", f"particle_{i}_Py", f"particle_{i}_Pz"])
                new_df = pd.DataFrame(df_subset.values, columns=new_columns)
                
                # Store metadata about particle types
                new_df.attrs['particle_info'] = {
                    'num_ions': 2,
                    'num_neutrals': 1,
                    'num_electrons': 2,
                    'particle_types': ['ion', 'ion', 'neutral', 'electron', 'electron'],
                    'particle_names': ['ion1', 'ion2', 'neutral', 'electron1', 'electron2'],
                }
                return new_df
            else:
                raise ValueError(f"Cannot reorganize dataframe: {error_msg}")
    
    # Build the reorganized dataframe using detected configuration
    new_columns = []
    new_data = []
    particle_types = []
    particle_names = []
    particle_idx = 0
    
    # Process ions
    for name, cols in particle_config['ions']:
        new_columns.extend([f"particle_{particle_idx}_Px", 
                           f"particle_{particle_idx}_Py", 
                           f"particle_{particle_idx}_Pz"])
        new_data.extend([df[cols[0]], df[cols[1]], df[cols[2]]])
        particle_types.append('ion')
        particle_names.append(name)
        particle_idx += 1
    
    # Process neutrals
    for name, cols in particle_config['neutrals']:
        new_columns.extend([f"particle_{particle_idx}_Px", 
                           f"particle_{particle_idx}_Py", 
                           f"particle_{particle_idx}_Pz"])
        new_data.extend([df[cols[0]], df[cols[1]], df[cols[2]]])
        particle_types.append('neutral')
        particle_names.append(name)
        particle_idx += 1
    
    # Process electrons
    for name, cols in particle_config['electrons']:
        new_columns.extend([f"particle_{particle_idx}_Px", 
                           f"particle_{particle_idx}_Py", 
                           f"particle_{particle_idx}_Pz"])
        new_data.extend([df[cols[0]], df[cols[1]], df[cols[2]]])
        particle_types.append('electron')
        particle_names.append(name)
        particle_idx += 1
    
    # Process other/generic particles
    for name, cols in particle_config['other']:
        new_columns.extend([f"particle_{particle_idx}_Px", 
                           f"particle_{particle_idx}_Py", 
                           f"particle_{particle_idx}_Pz"])
        new_data.extend([df[cols[0]], df[cols[1]], df[cols[2]]])
        particle_types.append('other')
        particle_names.append(name)
        particle_idx += 1
    
    # Create new dataframe
    new_df = pd.DataFrame(dict(zip(new_columns, new_data)))
    
    # Store metadata about particle configuration
    new_df.attrs['particle_info'] = {
        'num_ions': particle_config['num_ions'],
        'num_neutrals': particle_config['num_neutrals'],
        'num_electrons': particle_config['num_electrons'],
        'num_other': particle_config['num_other'],
        'total_particles': particle_config['total_particles'],
        'particle_types': particle_types,
        'particle_names': particle_names,
    }
    
    print(f"Reorganized dataframe: {particle_idx} particles, {len(new_columns)} columns")
    
    return new_df


def get_particle_info(df):
    """
    Get particle information from a reorganized dataframe.
    
    Returns:
        dict: Particle configuration info, or default if not available
    """
    if hasattr(df, 'attrs') and 'particle_info' in df.attrs:
        return df.attrs['particle_info']
    
    # Try to infer from column count
    particle_cols = [col for col in df.columns if col.startswith('particle_')]
    num_particles = len(particle_cols) // 3
    
    # Default assumption if no metadata
    return {
        'num_ions': min(2, num_particles),
        'num_neutrals': 1 if num_particles > 2 else 0,
        'num_electrons': max(0, num_particles - 3),
        'total_particles': num_particles,
        'particle_types': None,
        'particle_names': None,
    }


# =============================================================================
# SELECTION HANDLING
# =============================================================================

def extract_selection_indices(selection_data, coords_df):
    """Extract indices from selection data."""
    indices = []

    # Handle box selection
    if "range" in selection_data:
        x_range = selection_data["range"]["x"]
        y_range = selection_data["range"]["y"]

        selected_mask = (
            (coords_df["UMAP1"] >= x_range[0])
            & (coords_df["UMAP1"] <= x_range[1])
            & (coords_df["UMAP2"] >= y_range[0])
            & (coords_df["UMAP2"] <= y_range[1])
        )
        indices = np.where(selected_mask)[0].tolist()

    # Handle lasso selection
    elif "lassoPoints" in selection_data:
        # Extract lasso polygon coordinates
        lasso_x = selection_data["lassoPoints"]["x"]
        lasso_y = selection_data["lassoPoints"]["y"]

        # Create a Path object from the lasso points
        lasso_path = Path(np.column_stack([lasso_x, lasso_y]))

        # Check which points are within the lasso path
        points_array = np.column_stack([coords_df["UMAP1"], coords_df["UMAP2"]])
        inside_lasso = lasso_path.contains_points(points_array)

        # Get indices of points inside the lasso
        indices = np.where(inside_lasso)[0].tolist()

    # Handle direct point selection
    elif "points" in selection_data:
        indices = [pt["pointIndex"] for pt in selection_data["points"]]

    return indices


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def get_original_column_names(num_ions, num_neutrals, num_electrons):
    """
    Generate original column names for export based on particle counts.
    
    Returns:
        list: Column names in the original format
    """
    columns = []
    
    # Ions
    for i in range(1, num_ions + 1):
        columns.extend([f"Px_ion{i}", f"Py_ion{i}", f"Pz_ion{i}"])
    
    # Neutrals
    if num_neutrals == 1:
        columns.extend(["Px_neutral", "Py_neutral", "Pz_neutral"])
    else:
        for i in range(1, num_neutrals + 1):
            columns.extend([f"Px_neutral{i}", f"Py_neutral{i}", f"Pz_neutral{i}"])
    
    # Electrons
    for i in range(1, num_electrons + 1):
        columns.extend([f"Px_electron{i}", f"Py_electron{i}", f"Pz_electron{i}"])
    
    return columns


def convert_to_original_format(df, particle_info=None):
    """
    Convert a dataframe with standardized column names back to original format.
    
    Args:
        df: DataFrame with particle_X_Px/Py/Pz columns
        particle_info: Optional particle info dict. If None, will try to get from df.attrs
    
    Returns:
        DataFrame with original column names
    """
    if particle_info is None:
        particle_info = get_particle_info(df)
    
    # Get momentum columns
    momentum_columns = sorted([col for col in df.columns if col.startswith("particle_")])
    
    if not momentum_columns:
        return df  # No conversion needed
    
    num_particles = len(momentum_columns) // 3
    
    # Generate original column names
    original_columns = get_original_column_names(
        particle_info.get('num_ions', 2),
        particle_info.get('num_neutrals', 1),
        particle_info.get('num_electrons', 2)
    )
    
    # Make sure we have the right number of columns
    if len(original_columns) != len(momentum_columns):
        # Fall back to default 5-particle format
        original_columns = get_original_column_names(2, 1, 2)
        
        # If still doesn't match, use generic names
        if len(original_columns) != len(momentum_columns):
            original_columns = []
            for i in range(num_particles):
                original_columns.extend([f"Px_particle{i+1}", f"Py_particle{i+1}", f"Pz_particle{i+1}"])
    
    # Create new dataframe with original column names
    original_df = pd.DataFrame()
    for i, col in enumerate(momentum_columns):
        if i < len(original_columns):
            original_df[original_columns[i]] = df[col]
    
    # Add any non-particle columns
    for col in df.columns:
        if not col.startswith("particle_") and col not in ["UMAP1", "UMAP2", "file_label"]:
            original_df[col] = df[col]
    
    return original_df