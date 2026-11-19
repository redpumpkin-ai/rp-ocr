use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits, RgbImage};
use rusto_core::{InitializeConfig, Mat, OcrRunOptions, Orientation, RustO};
use serde::Serialize;

mod contrast;
mod cpu;
mod detection;
pub mod models;

/// Select the model precision and detector resolution for CPU inference.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    #[default]
    Balanced,
    /// Full-precision detection and integer English/Indonesian recognition.
    Fast,
    Compact,
}

const MAX_DIMENSION: u32 = 8_192;
const MAX_PIXELS: u64 = 32_000_000;

#[derive(Debug, Serialize)]
pub struct Line {
    pub text: String,
    pub confidence: f32,
    pub polygon: [(f32, f32); 4],
}

#[derive(Debug, Serialize)]
pub struct Page {
    pub width: u32,
    pub height: u32,
    pub rotation_clockwise_degrees: Option<i32>,
    pub lines: Vec<Line>,
}

impl Page {
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Reuse one CPU engine for a sequence of local images.
pub struct Engine {
    ocr: RustO,
    fast: bool,
}

impl Engine {
    /// Load an existing model bundle. This operation does not use the network.
    pub fn open(model_dir: &Path) -> Result<Self> {
        Self::open_profile(model_dir, Profile::Balanced)
    }

    /// Load a selected CPU profile from a verified local model bundle.
    pub fn open_profile(model_dir: &Path, profile: Profile) -> Result<Self> {
        models::verify(model_dir, profile)?;
        let (detector, recognizer, dictionary) = match profile {
            Profile::Balanced => ("det.rten", "rec.rten", "dict.txt"),
            Profile::Fast => ("det.rten", "rec-latin.rten", "dict-latin.txt"),
            Profile::Compact => ("det-compact.mnn", "rec-compact.mnn", "dict.txt"),
        };
        let mut config = InitializeConfig::ppv6(
            model_dir.join(detector),
            model_dir.join(recognizer),
            model_dir.join(dictionary),
        )
        .with_orientation(model_dir.join("orient.rten"));
        if matches!(profile, Profile::Compact) {
            config.det.limit_type = "max".to_string();
            config.det.limit_side_len = 640;
        }
        let ocr = if matches!(profile, Profile::Balanced | Profile::Fast) {
            config.rec.rec_workers = Some(1);
            let threads = std::env::var("RUSTO_NUM_THREADS")
                .or_else(|_| std::env::var("RAYON_NUM_THREADS"))
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(4);
            let pool = std::sync::Arc::new(rten::ThreadPool::with_num_threads(threads));
            let det = cpu::CpuSession::open(&config.det.model_path, pool.clone())?;
            let rec = cpu::CpuSession::open(&config.rec.model_path, pool.clone())?;
            let orient = cpu::CpuSession::open(&model_dir.join("orient.rten"), pool)?;
            config.orient = None;
            let mut ocr = RustO::with_custom_engines(config, det, rec)?;
            ocr.pipeline.orient = Some(Box::new(orient));
            Ok(ocr)
        } else {
            RustO::initialize(config)
        }
        .context("could not load the OCR models")?;
        Ok(Self {
            ocr,
            fast: matches!(profile, Profile::Fast),
        })
    }

    /// Read one JPEG or PNG, honoring its EXIF orientation and alpha channel.
    pub fn scan_file(&mut self, path: &Path) -> Result<Page> {
        let image =
            load_image(path).with_context(|| format!("could not read {}", path.display()))?;
        self.scan_rgb(image)
            .with_context(|| format!("OCR failed for {}", path.display()))
    }

    /// Read an RGB image. Polygon coordinates refer to the supplied image.
    pub fn scan_rgb(&mut self, mut image: RgbImage) -> Result<Page> {
        let (width, height) = image.dimensions();
        check_dimensions(width, height)?;
        let detection = if self.fast {
            detection::fast_options(&image)
        } else {
            None
        };
        contrast::restore(&mut image);

        let mut options = OcrRunOptions::default()
            .with_orientation_detection(true)
            .with_layout_reading(true);
        options.detection = detection;
        let input = Mat::from_rgb_stride(width, height, width as usize * 3, image.as_raw())?;
        let output = self.ocr.run_on_mat_with_options(&input, &options)?;
        let results = match output.orientation {
            // Preserve detection order from the corrected raster. The returned polygons
            // have already been mapped back onto the source image.
            Some(Orientation::Rotate90 | Orientation::Rotate180 | Orientation::Rotate270) => {
                output.to_text_results()
            }
            _ => output.to_text_results_with_options(&options),
        };
        let lines = results
            .into_iter()
            .filter(|result| !result.text.trim().is_empty())
            .map(|result| Line {
                text: result.text,
                confidence: result.score,
                polygon: result
                    .box_points
                    .map(|(x, y)| (x.clamp(0.0, width as f32), y.clamp(0.0, height as f32))),
            })
            .collect();

        Ok(Page {
            width,
            height,
            rotation_clockwise_degrees: output
                .orientation
                .map(|value| value.degrees().rem_euclid(360)),
            lines,
        })
    }
}

fn load_image(path: &Path) -> Result<RgbImage> {
    let mut reader = ImageReader::open(path)?.with_guessed_format()?;
    if !matches!(reader.format(), Some(ImageFormat::Jpeg | ImageFormat::Png)) {
        bail!("expected a JPEG or PNG image");
    }

    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_DIMENSION);
    limits.max_image_height = Some(MAX_DIMENSION);
    reader.limits(limits);
    let mut decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    check_dimensions(width, height)?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);

    if !image.color().has_alpha() {
        return Ok(image.into_rgb8());
    }

    let rgba = image.into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(RgbImage::from_fn(width, height, |x, y| {
        let [r, g, b, a] = rgba.get_pixel(x, y).0;
        let alpha = u32::from(a);
        image::Rgb(
            [r, g, b].map(|channel| {
                ((u32::from(channel) * alpha + 255 * (255 - alpha) + 127) / 255) as u8
            }),
        )
    }))
}

fn check_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0
        || height == 0
        || width.max(height) > MAX_DIMENSION
        || u64::from(width) * u64::from(height) > MAX_PIXELS
    {
        bail!(
            "image dimensions exceed {MAX_DIMENSION} pixels per side or {MAX_PIXELS} total pixels"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_dimensions;

    #[test]
    fn bounds_raster_size() {
        assert!(check_dimensions(2_480, 3_508).is_ok());
        assert!(check_dimensions(8_192, 8_192).is_err());
        assert!(check_dimensions(8_193, 1).is_err());
        assert!(check_dimensions(0, 800).is_err());
    }
}
