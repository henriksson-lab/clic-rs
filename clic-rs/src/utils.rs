/// Compute kernel half-size from a sigma value (mirrors CLIc's `sigma2kernelsize`).
pub fn sigma2kernelsize(sigma: f32) -> i32 {
    let rad = (sigma * 8.0) as i32;
    if rad % 2 == 0 { rad + 1 } else { rad }
}

/// Compute kernel size from a radius value (mirrors CLIc's `radius2kernelsize`).
pub fn radius2kernelsize(radius: f32) -> i32 {
    (radius * 2.0 + 1.0) as i32
}

/// Infer array dimensionality from shape (mirrors CLIc's `shape_to_dimension`).
pub fn shape_to_dimension(width: usize, height: usize, depth: usize) -> usize {
    if depth > 1 { 3 } else if height > 1 { 2 } else { 1 }
}

/// Find the next "smooth" number ≥ x whose prime factors are only {2,3,5,7}.
/// Used to pick FFT-friendly sizes (mirrors CLIc's `next_smooth`).
pub fn next_smooth(x: usize) -> usize {
    let z = (10.0 * (x as f64).log2()) as usize;
    let delta = 0.000001_f64;
    let mut a = vec![0.0_f64; z];

    for &p in &[2_i32, 3, 5, 7] {
        let log_p = (p as f64).ln();
        let mut power = p as usize;
        while power <= x + z {
            let mut j = x % power;
            if j > 0 { j = power - j; }
            while j < z {
                a[j] += log_p;
                j += power;
            }
            power *= p as usize;
        }
    }

    let log_x = (x as f64).ln();
    for (i, &val) in a.iter().enumerate() {
        if val >= log_x - delta {
            return x + i;
        }
    }
    usize::MAX
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
        // 8 = 2^3 is already smooth
        let s = next_smooth(8);
        assert!(s >= 8);
        // 9 = 3^2
        assert!(next_smooth(9) >= 9);
    }
}
