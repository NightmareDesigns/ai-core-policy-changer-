#!/usr/bin/env python3
"""
GGUF Core Policy Changer
========================
A tool for reading, modifying policy-related metadata, and rebuilding GGUF
model files.  GGUF (GPT-Generated Unified Format) stores model weights and
metadata in a single binary file.  This tool lets you:

  * Inspect all metadata key-value pairs inside a GGUF file.
  * Change policy-relevant fields such as system prompts, chat templates,
    and instruct templates.
  * Optionally decrypt a file that has been XOR-obfuscated before parsing.
  * Write a fully valid GGUF file with the updated metadata.

Supported GGUF versions: 1, 2, 3.

Usage
-----
  python gguf_policy_changer.py info      <model.gguf>
  python gguf_policy_changer.py get       <model.gguf> <key>
  python gguf_policy_changer.py set       <model.gguf> <key> <value> [-o output.gguf]
  python gguf_policy_changer.py decrypt   <model.gguf> [-k xor_key] [-o output.gguf]
  python gguf_policy_changer.py rebuild   <model.gguf> [-o output.gguf]
"""

import argparse
import os
import struct
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GGUF constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # b"GGUF" in little-endian
GGUF_SUPPORTED_VERSIONS = {1, 2, 3}
GGUF_DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# Struct format strings for each scalar value type
_SCALAR_FMT: Dict[GGUFValueType, str] = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.BOOL: "<B",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
}

# Maximum number of characters shown when displaying a string metadata value.
_INFO_TRUNCATE_LEN = 120

# Policy-relevant metadata keys (non-exhaustive)
POLICY_KEYS = {
    "tokenizer.chat_template",
    "tokenizer.ggml.instruct_template",
    "llama.system_prompt",
    "general.description",
    "general.tags",
    "general.license",
    "general.author",
    "general.name",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GGUFTensorInfo:
    name: str
    dimensions: List[int]
    ggml_type: int
    offset: int


@dataclass
class GGUFMetadataKV:
    key: str
    value_type: GGUFValueType
    value: Any


@dataclass
class GGUFFile:
    version: int
    metadata: List[GGUFMetadataKV] = field(default_factory=list)
    tensors: List[GGUFTensorInfo] = field(default_factory=list)
    tensor_data: bytes = b""
    alignment: int = GGUF_DEFAULT_ALIGNMENT


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class GGUFReader:
    """Parses a GGUF binary stream into a :class:`GGUFFile` object."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    # ---- low-level helpers ------------------------------------------------

    def _read(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise ValueError(
                f"Unexpected end of file at offset {self._pos} "
                f"(need {n} bytes, have {len(self._data) - self._pos})"
            )
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def _unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._read(size))[0]

    def _read_string(self) -> str:
        length = self._unpack("<Q")
        return self._read(length).decode("utf-8", errors="replace")

    # ---- value readers ----------------------------------------------------

    def _read_scalar(self, vtype: GGUFValueType) -> Any:
        fmt = _SCALAR_FMT[vtype]
        value = self._unpack(fmt)
        if vtype == GGUFValueType.BOOL:
            return bool(value)
        return value

    def _read_value(self, vtype: GGUFValueType) -> Any:
        if vtype == GGUFValueType.STRING:
            return self._read_string()
        if vtype == GGUFValueType.ARRAY:
            elem_type = GGUFValueType(self._unpack("<I"))
            count = self._unpack("<Q")
            return [self._read_value(elem_type) for _ in range(count)]
        return self._read_scalar(vtype)

    # ---- public API -------------------------------------------------------

    def read(self) -> GGUFFile:
        magic = self._unpack("<I")
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Not a GGUF file (magic 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}). "
                "Try decrypting the file first with the 'decrypt' subcommand."
            )
        version = self._unpack("<I")
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported GGUF version {version}. Supported: {GGUF_SUPPORTED_VERSIONS}")

        tensor_count = self._unpack("<Q")
        metadata_kv_count = self._unpack("<Q")

        metadata: List[GGUFMetadataKV] = []
        for _ in range(metadata_kv_count):
            key = self._read_string()
            vtype = GGUFValueType(self._unpack("<I"))
            value = self._read_value(vtype)
            metadata.append(GGUFMetadataKV(key=key, value_type=vtype, value=value))

        # Determine alignment from metadata (GGUF v3+)
        alignment = GGUF_DEFAULT_ALIGNMENT
        for kv in metadata:
            if kv.key == "general.alignment" and isinstance(kv.value, int):
                alignment = kv.value
                break

        tensors: List[GGUFTensorInfo] = []
        for _ in range(tensor_count):
            name = self._read_string()
            n_dims = self._unpack("<I")
            dims = [self._unpack("<Q") for _ in range(n_dims)]
            ggml_type = self._unpack("<I")
            offset = self._unpack("<Q")
            tensors.append(GGUFTensorInfo(name=name, dimensions=dims, ggml_type=ggml_type, offset=offset))

        # Tensor data begins at the next aligned boundary after the header
        pad = (alignment - (self._pos % alignment)) % alignment
        self._pos += pad
        tensor_data = self._data[self._pos :]

        return GGUFFile(
            version=version,
            metadata=metadata,
            tensors=tensors,
            tensor_data=tensor_data,
            alignment=alignment,
        )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class GGUFWriter:
    """Serialises a :class:`GGUFFile` back to bytes."""

    def __init__(self, gguf: GGUFFile) -> None:
        self._gguf = gguf
        self._buf = bytearray()

    # ---- low-level helpers ------------------------------------------------

    def _pack(self, fmt: str, value: Any) -> None:
        self._buf += struct.pack(fmt, value)

    def _write_string(self, s: str) -> None:
        encoded = s.encode("utf-8")
        self._pack("<Q", len(encoded))
        self._buf += encoded

    # ---- value writers ----------------------------------------------------

    def _write_scalar(self, vtype: GGUFValueType, value: Any) -> None:
        fmt = _SCALAR_FMT[vtype]
        if vtype == GGUFValueType.BOOL:
            value = int(bool(value))
        self._pack(fmt, value)

    def _write_value(self, vtype: GGUFValueType, value: Any) -> None:
        if vtype == GGUFValueType.STRING:
            self._write_string(value)
        elif vtype == GGUFValueType.ARRAY:
            if not value:
                # Empty array – UINT8 is a safe placeholder: readers that care
                # about the element type will see a zero-length array, so the
                # type byte does not affect the decoded value.
                self._pack("<I", int(GGUFValueType.UINT8))
                self._pack("<Q", 0)
            else:
                # Infer element type from first element.
                # NOTE: bool must be checked before int because bool is a
                # subclass of int in Python (isinstance(True, int) is True).
                first = value[0]
                if isinstance(first, str):
                    elem_type = GGUFValueType.STRING
                elif isinstance(first, bool):
                    elem_type = GGUFValueType.BOOL
                elif isinstance(first, float):
                    elem_type = GGUFValueType.FLOAT32
                elif isinstance(first, int):
                    elem_type = GGUFValueType.INT32
                else:
                    elem_type = GGUFValueType.STRING
                self._pack("<I", int(elem_type))
                self._pack("<Q", len(value))
                for elem in value:
                    self._write_value(elem_type, elem)
        else:
            self._write_scalar(vtype, value)

    # ---- public API -------------------------------------------------------

    def write(self) -> bytes:
        g = self._gguf
        # Header
        self._pack("<I", GGUF_MAGIC)
        self._pack("<I", g.version)
        self._pack("<Q", len(g.tensors))
        self._pack("<Q", len(g.metadata))

        # Metadata KV
        for kv in g.metadata:
            self._write_string(kv.key)
            self._pack("<I", int(kv.value_type))
            self._write_value(kv.value_type, kv.value)

        # Tensor info
        for ti in g.tensors:
            self._write_string(ti.name)
            self._pack("<I", len(ti.dimensions))
            for d in ti.dimensions:
                self._pack("<Q", d)
            self._pack("<I", ti.ggml_type)
            self._pack("<Q", ti.offset)

        # Padding to alignment boundary
        pad = (g.alignment - (len(self._buf) % g.alignment)) % g.alignment
        self._buf += b"\x00" * pad

        # Tensor data
        self._buf += g.tensor_data

        return bytes(self._buf)


# ---------------------------------------------------------------------------
# Decryption
# ---------------------------------------------------------------------------

def xor_decrypt(data: bytes, key: int) -> bytes:
    """Apply a single-byte XOR key to every byte of *data*.

    This reverses a common lightweight obfuscation applied to GGUF files
    (XOR is its own inverse, so encrypt == decrypt).
    """
    if not 0 <= key <= 255:
        raise ValueError(f"XOR key must be a single byte (0–255), got {key}")
    if key == 0:
        return data  # XOR with 0 is a no-op
    return bytes(b ^ key for b in data)


def detect_xor_key(data: bytes) -> Optional[int]:
    """Heuristically detect a single-byte XOR key by looking for the GGUF
    magic bytes in the decrypted stream.

    Returns the key (0–255) if found, else ``None``.
    """
    magic_bytes = struct.pack("<I", GGUF_MAGIC)  # b"GGUF"
    if len(data) < 4:
        return None
    for key in range(256):
        candidate = bytes(b ^ key for b in data[:4])
        if candidate == magic_bytes:
            return key
    return None


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def get_metadata(gguf: GGUFFile, key: str) -> Optional[GGUFMetadataKV]:
    """Return the :class:`GGUFMetadataKV` entry for *key*, or ``None``."""
    for kv in gguf.metadata:
        if kv.key == key:
            return kv
    return None


def set_metadata(gguf: GGUFFile, key: str, value: str) -> None:
    """Set *key* to string *value*, updating in-place or appending."""
    for kv in gguf.metadata:
        if kv.key == key:
            kv.value_type = GGUFValueType.STRING
            kv.value = value
            return
    gguf.metadata.append(GGUFMetadataKV(key=key, value_type=GGUFValueType.STRING, value=value))


def remove_metadata(gguf: GGUFFile, key: str) -> bool:
    """Remove *key* from metadata.  Returns ``True`` if it was present."""
    before = len(gguf.metadata)
    gguf.metadata = [kv for kv in gguf.metadata if kv.key != key]
    return len(gguf.metadata) < before


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_gguf(path: str, xor_key: Optional[int] = None) -> GGUFFile:
    """Read *path* and parse it as a GGUF file.

    If *xor_key* is provided the raw bytes are XOR-decrypted before parsing.
    """
    with open(path, "rb") as fh:
        data = fh.read()

    if xor_key is not None:
        data = xor_decrypt(data, xor_key)

    return GGUFReader(data).read()


def save_gguf(gguf: GGUFFile, path: str) -> None:
    """Serialise *gguf* and write it to *path*."""
    data = GGUFWriter(gguf).write()
    with open(path, "wb") as fh:
        fh.write(data)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    """Print all metadata in a GGUF file."""
    try:
        gguf = load_gguf(args.input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"GGUF version : {gguf.version}")
    print(f"Alignment    : {gguf.alignment}")
    print(f"Tensors      : {len(gguf.tensors)}")
    print(f"Metadata KVs : {len(gguf.metadata)}")
    print()
    print("Metadata:")
    for kv in gguf.metadata:
        tag = " [POLICY]" if kv.key in POLICY_KEYS else ""
        if kv.value_type == GGUFValueType.ARRAY:
            display = f"[array, {len(kv.value)} elements]"
        elif kv.value_type == GGUFValueType.STRING:
            truncated = kv.value[:_INFO_TRUNCATE_LEN].replace("\n", "\\n")
            display = f'"{truncated}{"…" if len(kv.value) > _INFO_TRUNCATE_LEN else ""}"'
        else:
            display = str(kv.value)
        print(f"  {kv.key}{tag} = {display}")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    """Print the value of a single metadata key."""
    try:
        gguf = load_gguf(args.input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    kv = get_metadata(gguf, args.key)
    if kv is None:
        print(f"Key '{args.key}' not found.", file=sys.stderr)
        return 1
    print(kv.value)
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    """Set a metadata key to a new string value and rebuild the file."""
    try:
        gguf = load_gguf(args.input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    set_metadata(gguf, args.key, args.value)

    output = args.output or args.input
    save_gguf(gguf, output)
    print(f"Set '{args.key}' → written to '{output}'")
    return 0


def cmd_decrypt(args: argparse.Namespace) -> int:
    """Decrypt a XOR-obfuscated GGUF file and write the plain GGUF."""
    with open(args.input, "rb") as fh:
        raw = fh.read()

    if args.xor_key is not None:
        key = args.xor_key
    else:
        key = detect_xor_key(raw)
        if key is None:
            print("Error: Could not detect XOR key automatically. "
                  "Specify it with -k / --xor-key.", file=sys.stderr)
            return 1
        print(f"Detected XOR key: 0x{key:02X} ({key})")

    decrypted = xor_decrypt(raw, key)

    # Validate that the result is a valid GGUF file
    try:
        GGUFReader(decrypted).read()
    except ValueError as exc:
        print(f"Error: Decrypted data is not a valid GGUF file: {exc}", file=sys.stderr)
        return 1

    output = args.output or args.input
    with open(output, "wb") as fh:
        fh.write(decrypted)
    print(f"Decrypted → '{output}'")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Parse and re-serialise a GGUF file (useful to fix alignment/padding)."""
    try:
        gguf = load_gguf(args.input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = args.output or args.input
    save_gguf(gguf, output)
    print(f"Rebuilt → '{output}'")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gguf_policy_changer",
        description="Inspect, modify policy metadata, decrypt, and rebuild GGUF model files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = sub.add_parser("info", help="Print all metadata in a GGUF file.")
    p_info.add_argument("input", help="Path to the input GGUF file.")

    # get
    p_get = sub.add_parser("get", help="Print the value of a single metadata key.")
    p_get.add_argument("input", help="Path to the input GGUF file.")
    p_get.add_argument("key", help="Metadata key to retrieve.")

    # set
    p_set = sub.add_parser("set", help="Set a metadata key to a new string value.")
    p_set.add_argument("input", help="Path to the input GGUF file.")
    p_set.add_argument("key", help="Metadata key to set.")
    p_set.add_argument("value", help="New string value.")
    p_set.add_argument("-o", "--output", default=None,
                       help="Output path (default: overwrite input).")

    # decrypt
    p_dec = sub.add_parser("decrypt",
                            help="Decrypt a XOR-obfuscated GGUF file.")
    p_dec.add_argument("input", help="Path to the obfuscated GGUF file.")
    p_dec.add_argument("-k", "--xor-key", dest="xor_key", type=lambda x: int(x, 0),
                       default=None, help="Single-byte XOR key (auto-detected if omitted).")
    p_dec.add_argument("-o", "--output", default=None,
                       help="Output path (default: overwrite input).")

    # rebuild
    p_rb = sub.add_parser("rebuild", help="Parse and re-serialise a GGUF file.")
    p_rb.add_argument("input", help="Path to the input GGUF file.")
    p_rb.add_argument("-o", "--output", default=None,
                      help="Output path (default: overwrite input).")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "info": cmd_info,
        "get": cmd_get,
        "set": cmd_set,
        "decrypt": cmd_decrypt,
        "rebuild": cmd_rebuild,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
