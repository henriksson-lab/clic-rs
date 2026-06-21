//! Tier 4 — operations composing tier1-tier3 primitives.
//!
//! Mirrors CLIc's `clic/src/tier4/` directory.

mod mean_squared_error;
mod std_of_all_pixels;
mod threshold_functions;

pub use mean_squared_error::mean_squared_error;
pub use std_of_all_pixels::standard_deviation_of_all_pixels;
pub use threshold_functions::threshold_mean;
