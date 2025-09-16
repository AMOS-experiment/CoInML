import base64
import io
import os

import numpy as np
import pandas as pd
from matplotlib.path import Path  # For lasso selection

# Define the required columns as they appear in the CSV.
required_columns = [
    "Px_ion1",
    "Py_ion1",
    "Pz_ion1",
    "Px_ion2",
    "Py_ion2",
    "Pz_ion2",
    "Px_neutral",
    "Py_neutral",
    "Pz_neutral",
    "Px_electron1",
    "Py_electron1",
    "Pz_electron1",
    "Px_electron2",
    "Py_electron2",
    "Pz_electron2",
]


# Add this helper function before the update_files callback
def store_file_data(file_id, data):
    """Store file data in a temporary file and return the path."""
    import os
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


def is_selection_file(df):
    """Check if the dataframe looks like a saved selection."""
    # Check for UMAP coordinates which would indicate a selection file
    return (
        "UMAP1" in df.columns and "UMAP2" in df.columns and "file_label" in df.columns
    )


def parse_contents(contents, filename):
    """Parse the uploaded file contents."""
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
            sep = " "
            print(f"Detected space separator for {filename}")

        # Try to read with pandas
        print(f"Reading {filename} with separator '{sep}'")
        try:
            df = pd.read_csv(
                io.StringIO(decoded.decode("utf-8", errors="replace")), sep=sep
            )
            print(f"Successfully read {filename}, shape: {df.shape}")

            # Check if this is a saved selection file
            if is_selection_file(df):
                print(f"{filename} is a selection file")
                return df, True  # Return df and flag indicating it's a selection file

            # Otherwise check if it's a standard COLTRIMS file
            print(f"Checking if {filename} is a COLTRIMS file...")
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: File {filename} is missing columns: {missing_cols}")
                return None, False

            print(f"{filename} is a valid COLTRIMS file")
            return df, False  # Return df and flag indicating it's not a selection file

        except Exception as e:
            print(f"Error in pd.read_csv for {filename}: {e}")
            # Try alternative parsing approaches
            if sep == ",":
                try:
                    print(f"Trying to read {filename} with space separator as fallback")
                    df = pd.read_csv(
                        io.StringIO(decoded.decode("utf-8", errors="replace")), sep=" "
                    )
                    print(
                        f"Successfully read {filename} with space separator, shape: {df.shape}"
                    )
                    # Check columns as before
                    if is_selection_file(df):
                        return df, True

                    missing_cols = [
                        col for col in required_columns if col not in df.columns
                    ]
                    if missing_cols:
                        print(
                            f"Warning: File {filename} is missing columns: {missing_cols}"
                        )
                        return None, False

                    return df, False
                except Exception as e2:
                    print(f"Error in fallback parsing for {filename}: {e2}")

            # Try with delim_whitespace for fixed-width files
            try:
                print(
                    f"Trying to read {filename} with delim_whitespace=True as second fallback"
                )
                df = pd.read_csv(
                    io.StringIO(decoded.decode("utf-8", errors="replace")),
                    delim_whitespace=True,
                )
                print(
                    f"Successfully read {filename} with delim_whitespace, shape: {df.shape}"
                )
                # Check columns as before
                if is_selection_file(df):
                    return df, True

                missing_cols = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_cols:
                    print(
                        f"Warning: File {filename} is missing columns: {missing_cols}"
                    )
                    return None, False

                return df, False
            except Exception as e3:
                print(f"Error in second fallback for {filename}: {e3}")
                raise
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback

        traceback.print_exc()
        return None, False


def reorganize_df(df):
    """Reorganize the dataframe with standardized column names."""
    df = df[required_columns].copy()
    new_columns = []
    for i in range(5):
        new_columns.extend([f"particle_{i}_Px", f"particle_{i}_Py", f"particle_{i}_Pz"])
    new_df = pd.DataFrame(df.values, columns=new_columns)
    return new_df


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
