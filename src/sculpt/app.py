import dash
import diskcache
from dash import DiskcacheManager

# Initialize cache and background callback manager for Dash 3.2.0
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

print("Successfully initialized DiskcacheManager for background callbacks")

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
    update_umap_parameter_filter_controls,
)
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
from sculpt.callbacks.umap_callbacks import (  # noqa: F401
    update_umap,
    update_umap_graph3_selection,
    update_umap_selected_only,
    update_umap_selected_run,
)
from sculpt.callbacks.visualization_callbacks import (  # noqa: F401
    toggle_clustering_params,
    toggle_visualization_settings,
    toggle_visualization_settings_graph15,
)
from sculpt.components.layout import create_layout

# print("Is CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("CUDA device name:", torch.cuda.get_device_name(0))


# Constants for physics calculations
mass_ion = 2 * 1836  # Deuterium ion (D+)
mass_neutral = 16 * 1836  # Neutral Oxygen atom
mass_electron = 1  # Electron mass

app = dash.Dash(__name__, suppress_callback_exceptions=True, background_callback_manager=background_callback_manager)
app.title = "Supervised Clustering and Uncovering Latent Patterns with Training SCULPT"

app.layout = create_layout()


if __name__ == "__main__":
    app.run(debug=True, port=9000)
    #app.run_server(debug=True, port=9000)