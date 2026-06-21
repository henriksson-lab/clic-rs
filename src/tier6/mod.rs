//! Tier 6 — higher-level operations composing lower tiers.
//!
//! Mirrors CLIc's `clic/src/tier6/` directory.

mod dilate_labels;
mod erode_labels;
mod gauss_otsu_labeling;
mod masked_voronoi_labeling;
mod proximal_neighbor_map;
mod remove_objects;
mod voronoi_labeling;

pub use dilate_labels::dilate_labels;
pub use erode_labels::erode_labels;
pub use gauss_otsu_labeling::gauss_otsu_labeling;
pub use masked_voronoi_labeling::masked_voronoi_labeling;
pub use proximal_neighbor_map::proximal_neighbor_count_map;
pub use remove_objects::{
    exclude_large_labels, exclude_small_labels, remove_large_labels, remove_small_labels,
};
pub use voronoi_labeling::voronoi_labeling;
