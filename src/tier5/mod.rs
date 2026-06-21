//! Tier 5 — higher-level operations composing lower tiers.

mod array_equal;
mod combine_labels;
mod connected_component_labeling;
mod filter_label_by_size;
mod merge_touching_labels;
mod normalize;
mod proximal_neighbor;
mod reduce_labels_to_centroids;

pub use array_equal::array_equal;
pub use combine_labels::combine_labels;
pub use connected_component_labeling::connected_component_labeling;
#[allow(deprecated)]
pub use connected_component_labeling::connected_components_labeling;
pub use filter_label_by_size::{exclude_labels_outside_size_range, filter_label_by_size};
pub use merge_touching_labels::merge_touching_labels;
pub use normalize::normalize;
pub use proximal_neighbor::proximal_neighbor_count;
pub use reduce_labels_to_centroids::reduce_labels_to_centroids;
