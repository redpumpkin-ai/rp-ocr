use std::fs::{self, File};
use std::io;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

use crate::Profile;

mod latin;

const SOURCE: &str =
    "https://huggingface.co/byrizki/rusto-models/resolve/34812c324fa258436fbcf1959eaf4ace7af6e5d4";

#[derive(Clone, Copy)]
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

const COMPACT_ASSETS: [Asset; 4] = [
    Asset {
        upstream: "ppocrv6-tiny-fp16/det.mnn",
        filename: "det-compact.mnn",
        sha256: "5100ad4adffc406385fcb7478b01a3fa11fb9fd417e9a39789c355fc15e949e8",
    },
    Asset {
        upstream: "ppocrv6-tiny-fp16/rec.mnn",
        filename: "rec-compact.mnn",
        sha256: "e825329be1c21c5d22b9eb4afc946d8bc70545297476b76938efbe9dfd4ec500",
    },
    ASSETS[2],
    ASSETS[3],
];

const INT8_READER: Asset = Asset {
    upstream: "ppocrv6-tiny-int8/rec.rten",
    filename: "rec-int8.rten",
    sha256: "6c8db072d7606054a9682194eb2db581eccb8490099382d63a0af67dc80b9499",
};

const FAST_ASSETS: [Asset; 4] = [
    ASSETS[0],
    Asset {
        upstream: "", // Derived locally from INT8_READER.
        filename: "rec-latin.rten",
        sha256: "6ff6e6544121c225b503f049136b7cdb23a813e26ab3c5e74c094a95f15c9ac4",
    },
    Asset {
        upstream: "", // Derived locally from ASSETS[2].
        filename: "dict-latin.txt",
        sha256: "0a4450f64631c386333cc0774edacd4b90b0c765c2fde376503349ee708f9aa9",
    },
    ASSETS[3],
];

fn assets(profile: Profile) -> &'static [Asset] {
    match profile {
        Profile::Balanced => &ASSETS,
        Profile::Fast => &FAST_ASSETS,
        Profile::Compact => &COMPACT_ASSETS,
    }
}

/// Provision the pinned model assets. Only this operation uses the network.
pub fn prepare(directory: &Path) -> Result<()> {
    prepare_profile(directory, Profile::Balanced)
}

/// Download and verify the models required by the selected CPU profile.
pub fn prepare_profile(directory: &Path, profile: Profile) -> Result<()> {
    fs::create_dir_all(directory)?;
    for asset in assets(profile)
        .iter()
        .filter(|asset| !asset.upstream.is_empty())
    {
        prepare_asset(directory, asset)?;
    }
    if matches!(profile, Profile::Fast)
        && (!matches_digest(
            &directory.join(FAST_ASSETS[1].filename),
            FAST_ASSETS[1].sha256,
        )? || !matches_digest(
            &directory.join(FAST_ASSETS[2].filename),
            FAST_ASSETS[2].sha256,
        )?)
    {
        prepare_asset(directory, &INT8_READER)?;
        prepare_asset(directory, &ASSETS[2])?;
        let (reader, dictionary) = latin::derive(
            &fs::read(directory.join(INT8_READER.filename))?,
            &fs::read_to_string(directory.join(ASSETS[2].filename))?,
        )?;
        for (asset, data) in [
            (FAST_ASSETS[1], reader.as_slice()),
            (FAST_ASSETS[2], dictionary.as_bytes()),
        ] {
            ensure!(
                format!("{:x}", Sha256::digest(data)) == asset.sha256,
                "derived model SHA-256 mismatch"
            );
            let staged = directory.join(format!(".{}.{}.part", asset.filename, std::process::id()));
            fs::write(&staged, data)?;
            fs::rename(staged, directory.join(asset.filename))?;
        }
    }
    verify(directory, profile)
}

fn prepare_asset(directory: &Path, asset: &Asset) -> Result<()> {
    let destination = directory.join(asset.filename);
    if matches_digest(&destination, asset.sha256)? {
        return Ok(());
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
    Ok(())
}

pub(crate) fn verify(directory: &Path, profile: Profile) -> Result<()> {
    let profile_name = match profile {
        Profile::Balanced => "balanced",
        Profile::Fast => "fast",
        Profile::Compact => "compact",
    };
    for asset in assets(profile) {
        let path = directory.join(asset.filename);
        ensure!(
            matches_digest(&path, asset.sha256)?,
            "{} is missing or failed verification. Run rp-ocr --profile {profile_name} --models {} prepare.",
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
