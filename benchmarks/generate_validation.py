"""Generate additional synthetic accuracy checks, separate from the original 12."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fonts", type=Path, default=Path("/System/Library/Fonts/Supplemental")
    )
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / "validation"
    destination.mkdir(exist_ok=True)
    documents = [
        (
            "agreement",
            "Times New Roman.ttf",
            36,
            [
                "SERVICE AGREEMENT",
                "Agreement ID: SA-2026-7319",
                "Supplier: Surya Digital Nusantara",
                "Service period: 01 December 2026 to 28 February 2027",
                "Monthly fee: Rp 12.750.000",
                "Payment is due within 21 calendar days.",
                "Keep one signed copy with the purchase order.",
                "Approved by: Maya Hartono",
            ],
        ),
        (
            "purchase-order",
            "Arial.ttf",
            34,
            [
                "PESANAN PEMBELIAN",
                "Nomor pesanan: PO-2026-58317",
                "Tanggal: 23 November 2026",
                "Pemasok: PT Citra Mandiri",
                "25 kotak kertas A4 x Rp 58.000",
                "14 map arsip x Rp 17.500",
                "Subtotal: Rp 1.695.000",
                "Pajak 11%: Rp 186.450",
                "Total pembayaran: Rp 1.881.450",
                "Kirim sebelum pukul 14:45.",
                "Petugas penerima: Bima Prakoso",
            ],
        ),
        (
            "asset-register",
            "Courier New.ttf",
            24,
            [
                "ASSET VERIFICATION REGISTER",
                "Location: Bandung Warehouse 04",
                "Checked on 24 November 2026",
                "CODE      QTY  CONDITION",
                *[f"ITM-{i:04d}  {i * 7:3d}  VERIFIED" for i in range(1, 27)],
                "Checked by: R. S. Wijaya",
            ],
        ),
    ]
    for name, filename, size, lines in documents:
        image = Image.new("RGB", (1280, 1600), "white")
        font = ImageFont.truetype(str(args.fonts / filename), size)
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            assert draw.textlength(line, font=font) <= 1080
            draw.text((100, 100 + index * 44), line, font=font, fill="black")
        if name == "agreement":
            draw.text((100, 1530), "Page 1 of 1", font=font, fill="black")
            lines = [*lines, "Page 1 of 1"]
        variants = [(name, image)]
        rotated = image.transpose(Image.Transpose.ROTATE_180)
        variants.append((name + "-rotated", rotated))
        for variant, raster in variants:
            raster.save(destination / (variant + ".png"))
            (destination / (variant + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
