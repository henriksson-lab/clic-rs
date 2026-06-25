/// Mathematical pi constant matching CLIc's `M_PI` fallback.
pub const PI: f64 = std::f64::consts::PI;

/// Quiet NaN matching CLIc's `NaN` constant.
pub const NAN: f32 = f32::NAN;

/// Positive infinity matching CLIc's `pINF` constant.
pub const P_INF: f32 = f32::INFINITY;

/// Negative infinity matching CLIc's `nINF` constant.
pub const N_INF: f32 = f32::NEG_INFINITY;

/// Compute kernel half-size from a sigma value (mirrors CLIc's `sigma2kernelsize`).
pub fn sigma2kernelsize(sigma: f32) -> i32 {
    let rad = (sigma * 8.0) as i32;
    if rad % 2 == 0 {
        rad + 1
    } else {
        rad
    }
}

/// Compute kernel size from a radius value (mirrors CLIc's `radius2kernelsize`).
pub fn radius2kernelsize(radius: f32) -> i32 {
    (radius * 2.0 + 1.0) as i32
}

/// Infer array dimensionality from shape (mirrors CLIc's `shape_to_dimension`).
pub fn shape_to_dimension(_width: usize, height: usize, depth: usize) -> usize {
    if depth > 1 {
        3
    } else if height > 1 {
        2
    } else {
        1
    }
}

/// Find the next "smooth" number ≥ x whose prime factors are only {2,3,5,7}.
/// Used to pick FFT-friendly sizes (mirrors CLIc's `next_smooth`).
pub fn next_smooth(x: usize) -> usize {
    let z = (10.0 * (x as f64).log2()) as usize;
    let delta = 0.000001_f64;
    let mut a = vec![0.0_f64; z];

    for p in [2_usize, 3, 5, 7] {
        handle_prime(x, z, &mut a, p);
    }

    let log_x = (x as f64).ln();
    for (i, &val) in a.iter().enumerate() {
        if val >= log_x - delta {
            return x + i;
        }
    }
    usize::MAX
}

fn handle_prime(x: usize, z: usize, a: &mut [f64], p: usize) {
    let log_p = (p as f64).ln();
    let mut power = p;

    while power <= x + z {
        let mut j = x % power;
        if j > 0 {
            j = power - j;
        }

        while j < z {
            a[j] += log_p;
            j += power;
        }

        power *= p;
    }
}

/// Compute an FFT-friendly shape (each dim rounded up to next smooth number).
pub fn fft_smooth_shape(shape: [usize; 3]) -> [usize; 3] {
    shape.map(|v| if v > 1 { next_smooth(v) } else { 1 })
}

/// Replace `{KEY}` placeholders in a template string (mirrors CLIc's `renderTemplate`).
pub fn render_template(tmpl: &str, vars: &[(&str, &str)]) -> String {
    let mut result = String::with_capacity(tmpl.len() * 2);
    let bytes = tmpl.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(end) = tmpl[i + 1..].find('}') {
                let key = &tmpl[i + 1..i + 1 + end];
                if let Some(&(_, val)) = vars.iter().find(|(k, _)| *k == key) {
                    result.push_str(val);
                    i += 1 + end + 1;
                    continue;
                }
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Load a file into a string.
pub fn load_file(file_path: impl AsRef<std::path::Path>) -> std::io::Result<String> {
    std::fs::read_to_string(file_path)
}

/// Save a string into a file.
pub fn save_file(file_path: impl AsRef<std::path::Path>, source: &str) -> std::io::Result<()> {
    std::fs::write(file_path, source)
}

/// Correct a start/stop/step range against an axis size.
///
/// This mirrors CLIc's `correct_range`, but returns the corrected tuple instead
/// of mutating pointer arguments.
pub fn correct_range(
    start: Option<i32>,
    stop: Option<i32>,
    step: Option<i32>,
    size: i32,
) -> (i32, i32, i32) {
    let step = step.unwrap_or(1);
    let mut start = start.unwrap_or(if step >= 0 { 0 } else { size - 1 });
    let mut stop = stop.unwrap_or(if step >= 0 { size } else { -1 });

    if start >= size {
        start = if step >= 0 { size } else { size - 1 };
    }
    if start < -size + 1 {
        start = -size + 1;
    }
    if stop > size {
        stop = size;
    }
    if stop < -size {
        stop = if start > 0 { -1 } else { -size };
    }
    if start < 0 {
        start = size - start;
    }
    if (start > stop && step > 0) || (start < stop && step < 0) {
        stop = start;
    }

    (start, stop, step)
}

/// Convert a string to lowercase.
pub fn to_lower(s: &str) -> String {
    s.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigma2kernelsize() {
        assert_eq!(sigma2kernelsize(0.0), 1);
        assert_eq!(sigma2kernelsize(1.0), 9);
        assert_eq!(sigma2kernelsize(2.0), 17);
    }

    #[test]
    fn test_clic_numeric_constants() {
        assert_eq!(PI, std::f64::consts::PI);
        assert!(NAN.is_nan());
        assert!(P_INF.is_infinite() && P_INF.is_sign_positive());
        assert!(N_INF.is_infinite() && N_INF.is_sign_negative());
    }

    #[test]
    fn test_shape_to_dimension() {
        assert_eq!(shape_to_dimension(10, 1, 1), 1);
        assert_eq!(shape_to_dimension(10, 10, 1), 2);
        assert_eq!(shape_to_dimension(10, 10, 10), 3);
    }

    #[test]
    fn test_render_template() {
        let s = render_template("Hello {name}!", &[("name", "world")]);
        assert_eq!(s, "Hello world!");
    }

    #[test]
    fn test_next_smooth() {
        assert_eq!(next_smooth(8), 8);
        assert_eq!(next_smooth(9), 9);
        assert_eq!(next_smooth(11), 12);
        assert_eq!(next_smooth(127), 128);
    }

    #[test]
    fn handle_prime_marks_offsets_for_prime_powers() {
        let mut a = vec![0.0; 6];
        handle_prime(10, 6, &mut a, 2);

        let log_2 = 2.0_f64.ln();
        assert_eq!(a[0], log_2);
        assert_eq!(a[2], log_2 * 2.0);
        assert_eq!(a[4], log_2);
    }

    #[test]
    fn test_correct_range_defaults() {
        assert_eq!(correct_range(None, None, None, 10), (0, 10, 1));
        assert_eq!(correct_range(None, None, Some(-1), 10), (9, -1, -1));
    }

    #[test]
    fn test_correct_range_clamps() {
        assert_eq!(correct_range(Some(12), Some(20), Some(1), 10), (10, 10, 1));
        assert_eq!(correct_range(Some(-20), Some(5), Some(1), 10), (19, 19, 1));
    }

    #[test]
    fn test_to_lower() {
        assert_eq!(to_lower("AbC"), "abc");
    }

    #[test]
    fn test_load_save_file() {
        let path = std::env::temp_dir().join(format!("clic_rs_utils_{}.txt", std::process::id()));
        save_file(&path, "hello").unwrap();
        assert_eq!(load_file(&path).unwrap(), "hello");
        let _ = std::fs::remove_file(path);
    }
}
