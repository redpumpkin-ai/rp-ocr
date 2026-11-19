use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use rp_ocr::{Engine, Profile};

#[derive(Parser)]
struct Arguments {
    #[arg(long, value_enum, default_value = "balanced")]
    profile: Profile,
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let start = Instant::now();
    let mut engine = Engine::open_profile(Path::new("models"), args.profile)?;
    println!(
        "{}",
        serde_json::json!({
            "event": "model_load",
            "profile": args.profile,
            "milliseconds": start.elapsed().as_secs_f64() * 1000.0,
            "threads": std::env::var("RUSTO_NUM_THREADS").ok(),
        })
    );
    for input in args.inputs {
        let start = Instant::now();
        let page = engine.scan_file(Path::new(&input))?;
        println!(
            "{}",
            serde_json::json!({
                "event": "page",
                "image": input,
                "milliseconds": start.elapsed().as_secs_f64() * 1000.0,
                "text": page.text(),
                "lines": page.lines.len(),
            })
        );
    }
    Ok(())
}
