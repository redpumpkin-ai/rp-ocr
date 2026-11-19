use image::RgbImage;

/// Stretch a narrow luminance range on washed-out scans. Retain clean and blank pages.
pub fn restore(image: &mut RgbImage) {
    let mut histogram = [0; 256];
    let mut samples = 0;
    for pixel in image.pixels().step_by(4) {
        let [r, g, b] = pixel.0.map(u32::from);
        let luminance = (77 * r + 150 * g + 29 * b + 128) / 256;
        histogram[luminance as usize] += 1;
        samples += 1;
    }
    let dark = percentile(&histogram, samples / 200);
    let light = percentile(&histogram, samples * 199 / 200);
    let range = light - dark;
    if !(12..=105).contains(&range) {
        return;
    }

    let lookup: [u8; 256] = std::array::from_fn(|level| {
        let clamped = level.clamp(dark, light);
        ((clamped - dark) * 255 / range) as u8
    });
    for pixel in image.pixels_mut() {
        pixel.0 = pixel.0.map(|level| lookup[level as usize]);
    }
}

fn percentile(histogram: &[usize; 256], rank: usize) -> usize {
    let mut count = 0;
    for (level, frequency) in histogram.iter().enumerate() {
        count += frequency;
        if count > rank {
            return level;
        }
    }
    255
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::restore;

    #[test]
    fn expands_washed_out_ink_and_paper() {
        let mut image = RgbImage::from_fn(100, 20, |x, _| Rgb([if x < 10 { 182 } else { 225 }; 3]));
        restore(&mut image);
        assert_eq!(image.get_pixel(0, 0).0, [0; 3]);
        assert_eq!(image.get_pixel(99, 0).0, [255; 3]);
    }

    #[test]
    fn preserves_blank_and_high_contrast_pages() {
        for level in [180, 10] {
            let mut image =
                RgbImage::from_fn(100, 20, |x, _| Rgb([if x < 10 { level } else { 180 }; 3]));
            let original = image.clone();
            restore(&mut image);
            assert_eq!(image, original);
        }
    }
}
