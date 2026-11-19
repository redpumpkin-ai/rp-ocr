use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rp_ocr::{Engine, Page, models};
use serde::Serialize;

#[derive(Parser)]
#[command(version, about = "Read printed document scans with local CPU OCR")]
struct Cli {
    /// Directory that holds the checked OCR models
    #[arg(long, global = true, default_value = "models")]
    models: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch and verify the pinned model bundle
    Prepare,
    /// Read one or more JPEG or PNG files
    Scan {
        /// Emit page data, line confidence and polygons as JSON
        #[arg(long)]
        json: bool,
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
}

#[derive(Serialize)]
struct Output {
    source: PathBuf,
    #[serde(flatten)]
    page: Page,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    match args.command {
        Command::Prepare => {
            models::prepare(&args.models)?;
            eprintln!("Verified OCR models in {}", args.models.display());
        }
        Command::Scan { json, inputs } => {
            let mut engine = Engine::open(&args.models)?;
            let pages = inputs
                .into_iter()
                .map(|source| {
                    Ok(Output {
                        page: engine.scan_file(&source)?,
                        source,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut stdout = io::stdout().lock();
            if json {
                serde_json::to_writer_pretty(&mut stdout, &pages)?;
                writeln!(stdout)?;
            } else {
                for (index, output) in pages.iter().enumerate() {
                    if index > 0 {
                        writeln!(stdout, "\x0c")?;
                    }
                    writeln!(stdout, "{}", output.page.text())?;
                }
            }
        }
    }
    Ok(())
}
