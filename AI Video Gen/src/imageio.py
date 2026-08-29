"""Minimal RGB PNG read/write — stdlib only.

The project deliberately avoids Pillow/numpy so that testing and review run anywhere,
including CI containers with nothing installed. We only ever handle 8-bit RGB and RGBA,
which is what `ffmpeg -pix_fmt rgb24` produces and what we write ourselves.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


class ImageError(RuntimeError):
    """Raised for PNGs outside the narrow subset we support."""


@dataclass
class Image:
    """An 8-bit RGB image. `pixels` is row-major, 3 bytes per pixel."""

    width: int
    height: int
    pixels: bytes

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        offset = (y * self.width + x) * 3
        return tuple(self.pixels[offset : offset + 3])  # type: ignore[return-value]

    def resized(self, width: int, height: int) -> "Image":
        """Nearest-neighbour resize. Good enough for contact sheets."""
        out = bytearray(width * height * 3)
        for y in range(height):
            src_y = min(self.height - 1, y * self.height // height)
            for x in range(width):
                src_x = min(self.width - 1, x * self.width // width)
                src = (src_y * self.width + src_x) * 3
                dst = (y * width + x) * 3
                out[dst : dst + 3] = self.pixels[src : src + 3]
        return Image(width, height, bytes(out))


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_png(image: Image, level: int = 6) -> bytes:
    """Encode 8-bit RGB PNG bytes using filter type 0 for every row."""
    stride = image.width * 3
    raw = b"".join(
        b"\x00" + image.pixels[y * stride : (y + 1) * stride] for y in range(image.height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", image.width, image.height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, level))
        + _chunk(b"IEND", b"")
    )


def write_png(path: Path, image: Image) -> None:
    """Write 8-bit RGB PNG to disk."""
    path.write_bytes(encode_png(image))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path: Path) -> Image:
    """Read an 8-bit RGB/RGBA PNG, applying all five scanline filters."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageError(f"{path} is not a PNG")

    width = height = 0
    channels = 3
    idat = bytearray()
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if tag == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            if depth != 8 or colour not in (2, 6):
                raise ImageError(
                    f"{path}: only 8-bit RGB/RGBA supported (depth={depth}, colour={colour}). "
                    "Extract frames with `ffmpeg -pix_fmt rgb24`."
                )
            channels = 3 if colour == 2 else 4
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        offset += 12 + length

    if not width or not height:
        raise ImageError(f"{path}: no IHDR chunk")

    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = bytearray(height * stride)
    previous = bytearray(stride)

    position = 0
    for y in range(height):
        filter_type = raw[position]
        position += 1
        line = bytearray(raw[position : position + stride])
        position += stride

        if filter_type == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + previous[i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                upper_left = previous[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(left, previous[i], upper_left)) & 0xFF
        elif filter_type != 0:
            raise ImageError(f"{path}: unknown PNG filter type {filter_type}")

        out[y * stride : (y + 1) * stride] = line
        previous = line

    if channels == 4:  # drop alpha
        rgb = bytearray(width * height * 3)
        for i in range(width * height):
            rgb[i * 3 : i * 3 + 3] = out[i * 4 : i * 4 + 3]
        return Image(width, height, bytes(rgb))
    return Image(width, height, bytes(out))
