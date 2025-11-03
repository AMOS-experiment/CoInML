# MUST BE AT THE VERY TOP OF THE FILE - BEFORE ANY IMPORTS!
import os
import sys

# Detect if running in Docker and set environment
IS_DOCKER = (
    os.path.exists('/.dockerenv') or 
    os.getenv('DOCKER_CONTAINER') == 'true' or
    os.getenv('PREFECT_API_URL', '').startswith('http://prefect-server')
)

if IS_DOCKER:
    os.environ['PREFECT_API_URL'] = 'http://prefect-server:4200/api'
    os.environ['DOCKER_CONTAINER'] = 'true'  # Ensure this is set for callbacks
    print("🐳 Running in Docker environment")
else:
    os.environ['PREFECT_API_URL'] = 'http://localhost:4200/api'
    print("💻 Running in local environment")

print(f"Prefect API URL: {os.environ['PREFECT_API_URL']}")

# Remove any cloud configuration
if 'PREFECT_API_KEY' in os.environ:
    del os.environ['PREFECT_API_KEY']

# Import Dash
import dash
import diskcache
from dash import DiskcacheManager

# Initialize cache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
print("Successfully initialized DiskcacheManager")

# Import all callbacks (in the same order as before)
from sculpt.callbacks.autoencoder_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.config_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.custom_plot_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.download_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.feature_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.file_management_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.filtering_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.genetic_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.initialization_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.mutual_information_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.selection_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.status_callbacks import *  # noqa: F401, F403
from sculpt.callbacks.umap_callbacks import (  # noqa: F401
    # update_umap,  # Commented out - using Prefect version
    update_umap_graph3_selection,
    update_umap_selected_only,
    update_umap_selected_run,
)

# Import the unified UMAP Prefect callbacks (handles both Docker and local)
from sculpt.callbacks.umap_prefect_callbacks import (  # noqa: F401
    run_umap_analysis,
    monitor_umap_flow_progress
)
print("✅ UMAP Prefect callbacks loaded")

from sculpt.callbacks.visualization_callbacks import *  # noqa: F401, F403
from sculpt.components.layout import create_layout

# Constants
mass_ion = 2 * 1836
mass_neutral = 16 * 1836
mass_electron = 1

# Create app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    background_callback_manager=background_callback_manager,
)
app.title = "Supervised Clustering and Uncovering Latent Patterns with Training SCULPT"
app.layout = create_layout()

if __name__ == "__main__":
    if IS_DOCKER:
        print("🚀 Starting SCULPT in Docker on http://0.0.0.0:9000")
        app.run(debug=True, host='0.0.0.0', port=9000)
    else:
        print("🚀 Starting SCULPT locally on http://localhost:9000")
        app.run(debug=True, port=9000)