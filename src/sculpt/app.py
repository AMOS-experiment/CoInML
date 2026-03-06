# MUST BE AT THE VERY TOP OF THE FILE - BEFORE ANY IMPORTS!
import os

# Import Dash
import dash
import diskcache
from dash import DiskcacheManager

from sculpt.callbacks.autoencoder_callbacks import (  # noqa: F401
    calculate_features_after_assignment,
    run_umap_on_mi_features,
    train_autoencoder_and_run_umap,
    train_mi_autoencoder,
    update_autoencoder_genetic_features,
    update_mi_feature_importance_table,
    update_mi_scatter_plot,
    update_mi_status,
)
from sculpt.callbacks.config_callbacks import (  # noqa: F401
    create_profile,
    delete_profile,
    initialize_profiles,
    load_profile_for_editing,
    populate_particle_fields,
    save_file_assignments,
    update_active_profiles_display,
    update_file_assignment_ui,
    update_particle_config_ui,
    update_profile,
    update_profile_dropdown,
)
from sculpt.callbacks.custom_plot_callbacks import (  # noqa: F401
    update_custom_feature_plot,
    update_graph25,
    update_scatter_graph15,
)
from sculpt.callbacks.download_callbacks import (  # noqa: F401
    download_filtered_data,
    download_genetic_features,
    download_graph3_selection_points,
    download_latent_features,
    download_mi_features,
    download_selected_points,
    download_selected_points_graph15,
    download_selected_points_graph25,
    download_selected_points_run,
    download_umap_filtered_data,
)
from sculpt.callbacks.feature_callbacks import (  # noqa: F401
    update_autoencoder_feature_ui,
    update_feature_dropdowns,
    update_feature_dropdowns_graph15,
    update_genetic_feature_ui,
    update_mi_feature_ui,
)
from sculpt.callbacks.file_management_callbacks import (  # noqa: F401
    update_file_selector_graph15,
    update_files,
    update_umap_selector,
)
from sculpt.callbacks.filtering_callbacks import (  # noqa: F401
    apply_density_filter,
    apply_parameter_filter,
    apply_umap_density_filter,
    apply_umap_parameter_filter,
    update_parameter_filter_controls,
    update_parameter_filter_controls_visibility,
    update_parameter_filter_range,
    update_physics_filter_dropdowns,
    update_umap_parameter_filter_controls,
    update_umap_physics_filter_dropdowns,
)

# NOTE: update_physics_filter_dropdowns and update_umap_physics_filter_dropdowns
# are new callbacks that dynamically populate the physics parameter filter dropdowns
# in the filtering tab, replacing the previously hardcoded options in selection.py.
from sculpt.callbacks.genetic_callbacks import (  # noqa: F401
    run_genetic_feature_discovery_and_umap,
    update_genetic_umap_status,
)
from sculpt.callbacks.initialization_callbacks import (  # noqa: F401
    init_filter_graph,
    init_umap_filter_graph,
)
from sculpt.callbacks.mutual_information_callbacks import (  # noqa: F401
    run_mi_feature_selection,
    update_feature_importance_table,
)
from sculpt.callbacks.selection_callbacks import (  # noqa: F401
    store_selected_points,
    store_selected_points_graph15,
    store_selected_points_run,
    update_umap_selected_run_only,
)
from sculpt.callbacks.status_callbacks import update_all_status  # noqa: F401

# Import the unified UMAP Prefect callbacks (handles both Docker and local)
from sculpt.callbacks.umap_prefect_callbacks import (  # noqa: F401
    monitor_umap_flow_progress,
    run_umap_analysis,
)
from sculpt.callbacks.visualization_callbacks import *  # noqa: F401, F403
from sculpt.components.layout import create_layout

# Detect if running in Docker and set environment
IS_DOCKER = (
    os.path.exists("/.dockerenv")
    or os.getenv("DOCKER_CONTAINER") == "true"
    or os.getenv("PREFECT_API_URL", "").startswith("http://prefect-server")
)

if IS_DOCKER:
    os.environ["PREFECT_API_URL"] = "http://prefect-server:4200/api"
    os.environ["DOCKER_CONTAINER"] = "true"  # Ensure this is set for callbacks
    print("🐳 Running in Docker environment")
else:
    os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"
    print("💻 Running in local environment")

print(f"Prefect API URL: {os.environ['PREFECT_API_URL']}")

# Remove any cloud configuration
if "PREFECT_API_KEY" in os.environ:
    del os.environ["PREFECT_API_KEY"]

# Initialize cache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
print("Successfully initialized DiskcacheManager")

print("Successfully initialized DiskcacheManager for background callbacks")

print("✅ UMAP Prefect callbacks loaded")

# NOTE: Mass constants (mass_ion, mass_neutral, mass_electron) previously defined here
# have been removed. All physics calculations now use the flexible configuration-aware
# functions in physics_features.py which get masses from the assigned particle profile.

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
        app.run(debug=True, host="0.0.0.0", port=9000)
    else:
        print("🚀 Starting SCULPT locally on http://localhost:9000")
        app.run(debug=True, port=9000)
