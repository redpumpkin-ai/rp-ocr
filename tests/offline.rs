use std::fs;
use std::path::Path;

use rp_ocr::Engine;

#[test]
#[ignore = "requires the prepared model bundle"]
fn reads_a_sideways_memo_in_order_with_source_coordinates() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("benchmarks/fixtures/rotated-90.png");
    let reference = fs::read_to_string(fixture.with_extension("txt"))?;

    let mut engine = Engine::open(&root.join("models"))?;
    let page = engine.scan_file(&fixture)?;

    assert_eq!(page.rotation_clockwise_degrees, Some(90));
    assert_eq!(page.text(), reference.trim());
    let title = &page.lines[0].polygon;
    assert!((title[0].0 - title[1].0).abs() < (title[0].1 - title[1].1).abs());
    assert!(page.lines.iter().all(|line| {
        line.polygon.iter().all(|&(x, y)| {
            (0.0..=page.width as f32).contains(&x) && (0.0..=page.height as f32).contains(&y)
        })
    }));
    Ok(())
}
