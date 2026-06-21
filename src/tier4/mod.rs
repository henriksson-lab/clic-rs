//! Tier 4 — operations composing tier1-tier3 primitives.
//!
//! Mirrors CLIc's `clic/src/tier4/` directory.

mod centroids_of_labels;
mod filter_label_by_values;
mod label_bounding_box;
mod mean_squared_error;
mod parametrics_map;
mod relabel_sequential;
mod spots_to_pointlist;
mod std_of_all_pixels;
mod threshold_functions;

pub use centroids_of_labels::centroids_of_labels;
pub use filter_label_by_values::{
    exclude_labels_with_map_values_out_of_range, exclude_labels_with_map_values_within_range,
    remove_labels_with_map_values_out_of_range, remove_labels_with_map_values_within_range,
};
pub use label_bounding_box::label_bounding_box;
pub use mean_squared_error::mean_squared_error;
pub use parametrics_map::{
    extension_ratio_map, label_mean_intensity_map, label_pixel_count_map, maximum_extension_map,
    maximum_intensity_map, maximum_of_touching_neighbors_map, mean_extension_map,
    mean_intensity_map, mean_of_touching_neighbors_map, median_of_touching_neighbors_map,
    minimum_intensity_map, minimum_of_touching_neighbors_map, mode_of_touching_neighbors_map,
    parametric_map, pixel_count_map, standard_deviation_intensity_map,
    standard_deviation_of_touching_neighbors_map, touching_neighbor_count_map,
};
pub use relabel_sequential::relabel_sequential;
pub use spots_to_pointlist::spots_to_pointlist;
pub use std_of_all_pixels::standard_deviation_of_all_pixels;
pub use threshold_functions::{percentile, threshold_mean, threshold_otsu, threshold_yen};
