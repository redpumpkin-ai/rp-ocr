use image::RgbImage;
use rusto_core::DetectionRunOptions;

/// Smaller detection raster for clear, sparse pages. Recognition still reads
/// the source pixels. Weak contrast, small inputs and dense pages keep the
/// original detector resolution.
pub(crate) fn fast_options(image: &RgbImage) -> Option<DetectionRunOptions> {
    let (width, height) = image.dimensions();
    if width.min(height) < 640 {
        return None;
    }
    let white = |x, y| image.get_pixel(x, y).0.iter().all(|&c| c >= 240);
    if !(0..width).all(|x| white(x, 0) && white(x, height - 1))
        || !(0..height).all(|y| white(0, y) && white(width - 1, y))
    {
        return None;
    }
    let mut rows = vec![false; height as usize];
    let mut columns = vec![false; width as usize];
    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel.0.iter().all(|&channel| channel < 150) {
            rows[y as usize] = true;
            columns[x as usize] = true;
        }
    }
    let row_bands = bands(&rows);
    let column_bands = bands(&columns);
    if row_bands == 0 || column_bands == 0 || row_bands > 24 || column_bands > 24 {
        return None;
    }
    Some(DetectionRunOptions {
        limit_type: Some("max".into()),
        limit_side_len: Some(640),
        ..Default::default()
    })
}

fn bands(projection: &[bool]) -> usize {
    let mut count = 0;
    let mut previous = None;
    for (index, &ink) in projection.iter().enumerate() {
        if ink {
            if previous.is_none_or(|last| index - last > 3) {
                count += 1;
            }
            previous = Some(index);
        }
    }
    count
}
