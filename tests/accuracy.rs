use std::fs;
use std::path::Path;

use rp_ocr::{Engine, Profile};

/// Frozen before the CPU runtime change, including the difficult/rotated pages.
#[test]
#[ignore = "requires the prepared model bundle"]
fn preserves_balanced_transcripts() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "../benchmarks/cpu-comparison-macos-arm64.json"
    ))?;
    let pages = baseline["runs"]["rust-balanced"][0]["pages"]
        .as_array()
        .expect("frozen baseline pages");
    let mut engine = Engine::open_profile(&root.join("models"), Profile::Balanced)?;
    for reference in pages {
        let filename = Path::new(reference["image"].as_str().unwrap())
            .file_name()
            .unwrap();
        let image = root.join("benchmarks/fixtures").join(filename);
        assert!(fs::metadata(&image)?.len() > 0);
        let page = engine.scan_file(&image)?;
        assert_eq!(
            page.text(),
            reference["text"].as_str().unwrap(),
            "transcript changed for {}",
            image.display()
        );
        assert!(page.lines.iter().all(|line| {
            line.confidence.is_finite()
                && line.polygon.iter().all(|&(x, y)| {
                    x.is_finite()
                        && y.is_finite()
                        && (0.0..=page.width as f32).contains(&x)
                        && (0.0..=page.height as f32).contains(&y)
                })
        }));
    }
    Ok(())
}

#[test]
#[ignore = "requires the prepared model bundle"]
fn fast_profile_does_not_increase_any_pages_character_errors() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "../benchmarks/cpu-comparison-macos-arm64.json"
    ))?;
    let mut engine = Engine::open_profile(&root.join("models"), Profile::Fast)?;
    for reference in baseline["runs"]["rust-balanced"][0]["pages"]
        .as_array()
        .unwrap()
    {
        let filename = Path::new(reference["image"].as_str().unwrap())
            .file_name()
            .unwrap();
        let image = root.join("benchmarks/fixtures").join(filename);
        let expected = normalize(&fs::read_to_string(image.with_extension("txt"))?);
        let actual = normalize(&engine.scan_file(&image)?.text());
        let edits = edit_distance(&expected, &actual);
        assert!(
            edits <= reference["character_edits"].as_u64().unwrap() as usize,
            "accuracy regressed for {}: {} edits; baseline {}",
            image.display(),
            edits,
            reference["character_edits"]
        );
    }
    Ok(())
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn edit_distance(expected: &str, actual: &str) -> usize {
    let actual: Vec<_> = actual.chars().collect();
    let mut previous: Vec<_> = (0..=actual.len()).collect();
    for (row, expected) in expected.chars().enumerate() {
        let mut current = vec![row + 1];
        for (column, &actual) in actual.iter().enumerate() {
            current.push(
                (previous[column + 1] + 1)
                    .min(current[column] + 1)
                    .min(previous[column] + usize::from(expected != actual)),
            );
        }
        previous = current;
    }
    previous[actual.len()]
}
