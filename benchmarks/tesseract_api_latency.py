"""Measure persistent Tesseract CPU inference with fixed document presets."""

import argparse
import ctypes as c
import ctypes.util
import json
import math
import time
from pathlib import Path


def load_library(name, explicit=None):
    path = str(explicit) if explicit else ctypes.util.find_library(name)
    if path is None:
        raise RuntimeError(f"Shared library {name!r} is unavailable")
    return c.CDLL(path)


def configure(lib, name, arguments, result):
    function = getattr(lib, name)
    function.argtypes = arguments
    function.restype = result
    return function


class Tesseract:
    def __init__(self, preset, tesseract_library=None, leptonica_library=None):
        self.preset = preset
        self.lept = load_library("leptonica", leptonica_library)
        self.tess = load_library("tesseract", tesseract_library)
        pointer = c.c_void_p
        configure(self.lept, "pixRead", [c.c_char_p], pointer)
        configure(self.lept, "pixDestroy", [c.POINTER(pointer)], None)
        configure(self.lept, "pixGetWidth", [pointer], c.c_int)
        configure(self.lept, "pixGetHeight", [pointer], c.c_int)
        configure(self.lept, "pixScale", [pointer, c.c_float, c.c_float], pointer)
        configure(self.lept, "pixSetResolution", [pointer, c.c_int, c.c_int], c.c_int)
        configure(self.lept, "pixConvertTo8", [pointer, c.c_int], pointer)
        configure(self.lept, "pixGetGrayHistogram", [pointer, c.c_int], pointer)
        configure(
            self.lept,
            "numaGetFValue",
            [pointer, c.c_int, c.POINTER(c.c_float)],
            c.c_int,
        )
        configure(self.lept, "numaDestroy", [c.POINTER(pointer)], None)
        configure(
            self.lept,
            "pixGammaTRC",
            [pointer, pointer, c.c_float, c.c_int, c.c_int],
            pointer,
        )
        configure(self.lept, "pixConvertTo1", [pointer, c.c_int], pointer)
        configure(
            self.lept,
            "pixFindSkewSweepAndSearch",
            [
                pointer,
                c.POINTER(c.c_float),
                c.POINTER(c.c_float),
                c.c_int,
                c.c_int,
                c.c_float,
                c.c_float,
                c.c_float,
            ],
            c.c_int,
        )
        configure(
            self.lept,
            "pixRotate",
            [pointer, c.c_float, c.c_int, c.c_int, c.c_int, c.c_int],
            pointer,
        )
        configure(self.tess, "TessBaseAPICreate", [], pointer)
        configure(
            self.tess, "TessBaseAPIInit3", [pointer, c.c_char_p, c.c_char_p], c.c_int
        )
        configure(self.tess, "TessBaseAPISetPageSegMode", [pointer, c.c_int], None)
        configure(
            self.tess,
            "TessBaseAPISetVariable",
            [pointer, c.c_char_p, c.c_char_p],
            c.c_int,
        )
        configure(self.tess, "TessBaseAPISetImage2", [pointer, pointer], None)
        configure(self.tess, "TessBaseAPIRecognize", [pointer, pointer], c.c_int)
        configure(self.tess, "TessBaseAPIGetUTF8Text", [pointer], pointer)
        configure(self.tess, "TessDeleteText", [pointer], None)
        configure(self.tess, "TessBaseAPIClear", [pointer], None)
        configure(self.tess, "TessBaseAPIDelete", [pointer], None)
        configure(self.tess, "TessVersion", [], c.c_char_p)
        self.api = self.tess.TessBaseAPICreate()
        if not self.api:
            raise RuntimeError("TessBaseAPICreate failed")
        if self.tess.TessBaseAPIInit3(self.api, None, b"eng+ind") != 0:
            self.close()
            raise RuntimeError("Tesseract could not initialize eng+ind")
        self.tess.TessBaseAPISetPageSegMode(self.api, 1)
        method = b"0" if preset == "otsu" else b"2"
        if not self.tess.TessBaseAPISetVariable(
            self.api, b"thresholding_method", method
        ):
            self.close()
            raise RuntimeError("Tesseract rejected the thresholding method")

    def close(self):
        self.tess.TessBaseAPIDelete(self.api)
        self.api = None

    def restore_contrast(self, pix, own):
        gray = own(self.lept.pixConvertTo8(pix, 0))
        histogram = c.c_void_p(self.lept.pixGetGrayHistogram(gray, 2))
        if not histogram:
            raise RuntimeError("Could not compute the luminance histogram")
        counts = []
        try:
            value = c.c_float()
            for index in range(256):
                if self.lept.numaGetFValue(histogram, index, c.byref(value)):
                    raise RuntimeError("Could not read the luminance histogram")
                counts.append(value.value)
        finally:
            self.lept.numaDestroy(c.byref(histogram))
        count = sum(counts)
        cumulative = 0
        dark = light = None
        for index, frequency in enumerate(counts):
            cumulative += frequency
            if dark is None and cumulative > count * 0.005:
                dark = index
            if cumulative > count * 0.995:
                light = index
                break
        applied = dark is not None and light is not None and 12 <= light - dark <= 105
        if applied:
            pix = own(self.lept.pixGammaTRC(None, gray, 1.0, dark, light))
        return pix, {
            "contrast_stretched": applied,
            "dark_level": dark,
            "light_level": light,
        }

    def scan(self, path):
        owned = []
        angle = c.c_float()
        confidence = c.c_float()

        def own(pointer):
            if not pointer:
                raise RuntimeError(f"Leptonica image processing failed for {path}")
            value = c.c_void_p(pointer)
            owned.append(value)
            return value

        try:
            pix = own(self.lept.pixRead(str(path).encode()))
            scale = 1.0
            contrast = {"contrast_stretched": False}
            if self.preset == "document-contrast":
                pix, contrast = self.restore_contrast(pix, own)
            if self.preset in ("document", "document-contrast"):
                longest = max(self.lept.pixGetWidth(pix), self.lept.pixGetHeight(pix))
                scale = max(1.0, 1000 / longest)
                if scale > 1:
                    pix = own(self.lept.pixScale(pix, scale, scale))
                binary = own(self.lept.pixConvertTo1(pix, 130))
                found = self.lept.pixFindSkewSweepAndSearch(
                    binary, c.byref(angle), c.byref(confidence), 4, 2, 15.0, 1.0, 0.01
                )
                if (
                    found == 0
                    and confidence.value >= 2.0
                    and 0.5 <= abs(angle.value) <= 15.0
                ):
                    # Leptonica rotation enum values select area mapping and white fill.
                    pix = own(
                        self.lept.pixRotate(pix, math.radians(angle.value), 1, 1, 0, 0)
                    )
                if self.lept.pixSetResolution(pix, 300, 300) != 0:
                    raise RuntimeError("Could not set the document resolution")
            self.tess.TessBaseAPISetImage2(self.api, pix)
            if self.tess.TessBaseAPIRecognize(self.api, None) != 0:
                raise RuntimeError(f"Tesseract recognition failed for {path}")
            pointer = self.tess.TessBaseAPIGetUTF8Text(self.api)
            if not pointer:
                raise RuntimeError(f"Tesseract returned a null text buffer for {path}")
            try:
                text = c.string_at(pointer).decode("utf-8")
            finally:
                self.tess.TessDeleteText(pointer)
            return text, {
                "scale": scale,
                "deskew_angle_degrees": angle.value,
                "deskew_confidence": confidence.value,
                **contrast,
            }
        finally:
            self.tess.TessBaseAPIClear(self.api)
            for pix in reversed(owned):
                self.lept.pixDestroy(c.byref(pix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["otsu", "sauvola", "document", "document-contrast"],
        required=True,
    )
    parser.add_argument("--tesseract-library", type=Path)
    parser.add_argument("--leptonica-library", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    start = time.perf_counter()
    engine = Tesseract(args.preset, args.tesseract_library, args.leptonica_library)
    print(
        json.dumps(
            {
                "event": "model_load",
                "milliseconds": (time.perf_counter() - start) * 1000,
                "version": engine.tess.TessVersion().decode(),
                "preset": args.preset,
            }
        ),
        flush=True,
    )
    try:
        for path in args.inputs:
            start = time.perf_counter()
            text, preprocessing = engine.scan(path)
            milliseconds = (time.perf_counter() - start) * 1000
            print(
                json.dumps(
                    {
                        "event": "page",
                        "image": str(path),
                        "milliseconds": milliseconds,
                        "text": text,
                        "preprocessing": preprocessing,
                    }
                ),
                flush=True,
            )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
