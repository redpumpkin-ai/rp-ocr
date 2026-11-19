use std::fs::{self, File};
use std::io;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

const SOURCE: &str =
    "https://huggingface.co/byrizki/rusto-models/resolve/34812c324fa258436fbcf1959eaf4ace7af6e5d4";

struct Asset {
    upstream: &'static str,
    filename: &'static str,
    sha256: &'static str,
}

const ASSETS: [Asset; 4] = [
    Asset {
        upstream: "ppocrv6-tiny/det.rten",
        filename: "det.rten",
        sha256: "ca1f0196d2a2da2b485e2cda6aaf5ba02258ac71127d577c8a77a75e89b67605",
    },
    Asset {
        upstream: "ppocrv6-tiny/rec.rten",
        filename: "rec.rten",
        sha256: "6c94c6b42c43a39d11cf3dd1267199ba04a11afe9de43632daa6f3983b365a9b",
    },
    Asset {
        upstream: "ppocrv6-tiny/dict.txt",
        filename: "dict.txt",
        sha256: "c5cbe34ef40c29c4df07ed012bf96569cb69a2d2a01a07027e9f13cb832bd9cd",
    },
    Asset {
        upstream: "modules/orient.rten",
        filename: "orient.rten",
        sha256: "422fa370f05d871f7510b0b4aa453a98a9ed7f7d5284d0f7966bbe9602d82dfa",
    },
];

/// Provision the pinned model assets. Only this operation uses the network.
pub fn prepare(directory: &Path) -> Result<()> {
    fs::create_dir_all(directory)?;
    for asset in &ASSETS {
        let destination = directory.join(asset.filename);
        if matches_digest(&destination, asset.sha256)? {
            continue;
        }

        let staged = directory.join(format!(".{}.{}.part", asset.filename, std::process::id()));
        let download = || -> Result<()> {
            let url = format!("{SOURCE}/{}?download=true", asset.upstream);
            let status = Command::new("curl")
                .args([
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--retry",
                    "3",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    "90",
                    "--output",
                ])
                .arg(&staged)
                .arg(url)
                .status()
                .context("install curl to download OCR models")?;
            ensure!(status.success(), "download failed for {}", asset.filename);
            ensure!(
                matches_digest(&staged, asset.sha256)?,
                "SHA-256 mismatch for {}",
                asset.filename
            );
            fs::rename(&staged, &destination)?;
            Ok(())
        };

        if let Err(error) = download() {
            let _ = fs::remove_file(&staged);
            return Err(error);
        }
    }
    Ok(())
}

pub(crate) fn verify(directory: &Path) -> Result<()> {
    for asset in &ASSETS {
        let path = directory.join(asset.filename);
        ensure!(
            matches_digest(&path, asset.sha256)?,
            "{} is missing or failed verification. Run rp-ocr --models {} prepare.",
            path.display(),
            directory.display()
        );
    }
    Ok(())
}

fn matches_digest(path: &Path, expected: &str) -> io::Result<bool> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    let mut hash = Sha256::new();
    io::copy(&mut file, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()) == expected)
}
