"""
Unit tests for gguf_policy_changer.py
"""

import io
import struct
import sys
import unittest

# ---------------------------------------------------------------------------
# We import the module directly (no install needed)
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from gguf_policy_changer import (  # noqa: E402
    GGUF_MAGIC,
    GGUFFile,
    GGUFMetadataKV,
    GGUFReader,
    GGUFTensorInfo,
    GGUFValueType,
    GGUFWriter,
    detect_xor_key,
    get_metadata,
    remove_metadata,
    set_metadata,
    xor_decrypt,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal GGUF byte-strings for testing
# ---------------------------------------------------------------------------

def _encode_string(s: str) -> bytes:
    enc = s.encode("utf-8")
    return struct.pack("<Q", len(enc)) + enc


def _encode_kv(key: str, vtype: GGUFValueType, value) -> bytes:
    buf = _encode_string(key) + struct.pack("<I", int(vtype))
    if vtype == GGUFValueType.STRING:
        buf += _encode_string(value)
    elif vtype == GGUFValueType.UINT32:
        buf += struct.pack("<I", value)
    elif vtype == GGUFValueType.FLOAT32:
        buf += struct.pack("<f", value)
    elif vtype == GGUFValueType.BOOL:
        buf += struct.pack("<B", int(bool(value)))
    elif vtype == GGUFValueType.UINT64:
        buf += struct.pack("<Q", value)
    elif vtype == GGUFValueType.INT32:
        buf += struct.pack("<i", value)
    elif vtype == GGUFValueType.ARRAY:
        # value = (elem_type, [elements])
        elem_type, elements = value
        buf += struct.pack("<I", int(elem_type))
        buf += struct.pack("<Q", len(elements))
        for elem in elements:
            if elem_type == GGUFValueType.STRING:
                buf += _encode_string(elem)
            elif elem_type == GGUFValueType.UINT32:
                buf += struct.pack("<I", elem)
    return buf


def _build_gguf(
    version: int = 3,
    kvs=None,
    tensors=None,
    tensor_data: bytes = b"",
    alignment: int = 32,
) -> bytes:
    """Build a minimal GGUF binary for testing."""
    if kvs is None:
        kvs = []
    if tensors is None:
        tensors = []

    metadata_bytes = b"".join(_encode_kv(*kv) for kv in kvs)
    tensor_info_bytes = b""
    for name, dims, ggml_type, offset in tensors:
        tensor_info_bytes += _encode_string(name)
        tensor_info_bytes += struct.pack("<I", len(dims))
        for d in dims:
            tensor_info_bytes += struct.pack("<Q", d)
        tensor_info_bytes += struct.pack("<I", ggml_type)
        tensor_info_bytes += struct.pack("<Q", offset)

    header = struct.pack("<I", GGUF_MAGIC)
    header += struct.pack("<I", version)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(kvs))
    header += metadata_bytes
    header += tensor_info_bytes

    pad = (alignment - (len(header) % alignment)) % alignment
    return header + b"\x00" * pad + tensor_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGGUFReader(unittest.TestCase):

    def test_parse_minimal_gguf(self):
        data = _build_gguf()
        gguf = GGUFReader(data).read()
        self.assertEqual(gguf.version, 3)
        self.assertEqual(len(gguf.metadata), 0)
        self.assertEqual(len(gguf.tensors), 0)

    def test_parse_string_metadata(self):
        kvs = [("tokenizer.chat_template", GGUFValueType.STRING, "Hello {user}")]
        data = _build_gguf(kvs=kvs)
        gguf = GGUFReader(data).read()
        self.assertEqual(len(gguf.metadata), 1)
        self.assertEqual(gguf.metadata[0].key, "tokenizer.chat_template")
        self.assertEqual(gguf.metadata[0].value, "Hello {user}")

    def test_parse_numeric_metadata(self):
        kvs = [("llama.context_length", GGUFValueType.UINT32, 4096)]
        data = _build_gguf(kvs=kvs)
        gguf = GGUFReader(data).read()
        self.assertEqual(gguf.metadata[0].value, 4096)

    def test_parse_bool_metadata(self):
        kvs = [("tokenizer.ggml.add_bos_token", GGUFValueType.BOOL, True)]
        data = _build_gguf(kvs=kvs)
        gguf = GGUFReader(data).read()
        self.assertTrue(gguf.metadata[0].value)

    def test_parse_array_metadata(self):
        kvs = [("tokenizer.ggml.tokens", GGUFValueType.ARRAY,
                (GGUFValueType.STRING, ["<unk>", "<s>", "</s>"]))]
        data = _build_gguf(kvs=kvs)
        gguf = GGUFReader(data).read()
        self.assertEqual(gguf.metadata[0].value, ["<unk>", "<s>", "</s>"])

    def test_parse_tensor_info(self):
        tensors = [("token_embd.weight", [32000, 4096], 0, 0)]
        data = _build_gguf(tensors=tensors, tensor_data=b"\xAB" * 64)
        gguf = GGUFReader(data).read()
        self.assertEqual(len(gguf.tensors), 1)
        ti = gguf.tensors[0]
        self.assertEqual(ti.name, "token_embd.weight")
        self.assertEqual(ti.dimensions, [32000, 4096])
        self.assertEqual(ti.ggml_type, 0)

    def test_parse_tensor_data(self):
        tensor_data = b"\xDE\xAD\xBE\xEF" * 4
        data = _build_gguf(tensor_data=tensor_data)
        gguf = GGUFReader(data).read()
        self.assertEqual(gguf.tensor_data, tensor_data)

    def test_invalid_magic_raises(self):
        bad = struct.pack("<I", 0xDEADBEEF) + struct.pack("<I", 3) + struct.pack("<QQ", 0, 0)
        with self.assertRaises(ValueError) as ctx:
            GGUFReader(bad).read()
        self.assertIn("Not a GGUF file", str(ctx.exception))

    def test_unsupported_version_raises(self):
        data = _build_gguf(version=99)
        # Patch the magic to match so we get to the version check
        bad = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 99) + struct.pack("<QQ", 0, 0)
        with self.assertRaises(ValueError) as ctx:
            GGUFReader(bad).read()
        self.assertIn("Unsupported GGUF version", str(ctx.exception))

    def test_truncated_file_raises(self):
        data = _build_gguf()
        with self.assertRaises(ValueError) as ctx:
            GGUFReader(data[:4]).read()
        self.assertIn("Unexpected end of file", str(ctx.exception))

    def test_version_1_and_2_parse(self):
        for ver in (1, 2):
            data = _build_gguf(version=ver)
            gguf = GGUFReader(data).read()
            self.assertEqual(gguf.version, ver)

    def test_alignment_read_from_metadata(self):
        kvs = [("general.alignment", GGUFValueType.UINT32, 64)]
        data = _build_gguf(kvs=kvs, alignment=64)
        gguf = GGUFReader(data).read()
        self.assertEqual(gguf.alignment, 64)

    def test_multiple_metadata_keys(self):
        kvs = [
            ("general.name", GGUFValueType.STRING, "MyModel"),
            ("llama.context_length", GGUFValueType.UINT32, 2048),
            ("tokenizer.chat_template", GGUFValueType.STRING, "{prompt}"),
        ]
        data = _build_gguf(kvs=kvs)
        gguf = GGUFReader(data).read()
        self.assertEqual(len(gguf.metadata), 3)
        keys = [kv.key for kv in gguf.metadata]
        self.assertIn("general.name", keys)
        self.assertIn("tokenizer.chat_template", keys)


class TestGGUFWriter(unittest.TestCase):

    def _roundtrip(self, gguf: GGUFFile) -> GGUFFile:
        data = GGUFWriter(gguf).write()
        return GGUFReader(data).read()

    def test_roundtrip_empty(self):
        gguf = GGUFFile(version=3)
        rt = self._roundtrip(gguf)
        self.assertEqual(rt.version, 3)
        self.assertEqual(rt.metadata, [])
        self.assertEqual(rt.tensors, [])
        self.assertEqual(rt.tensor_data, b"")

    def test_roundtrip_string_metadata(self):
        gguf = GGUFFile(version=3, metadata=[
            GGUFMetadataKV("tokenizer.chat_template", GGUFValueType.STRING, "<s>{msg}</s>"),
        ])
        rt = self._roundtrip(gguf)
        self.assertEqual(rt.metadata[0].value, "<s>{msg}</s>")

    def test_roundtrip_numeric_metadata(self):
        gguf = GGUFFile(version=3, metadata=[
            GGUFMetadataKV("llama.context_length", GGUFValueType.UINT32, 8192),
            GGUFMetadataKV("llama.embedding_length", GGUFValueType.UINT32, 4096),
        ])
        rt = self._roundtrip(gguf)
        self.assertEqual(rt.metadata[0].value, 8192)
        self.assertEqual(rt.metadata[1].value, 4096)

    def test_roundtrip_bool_metadata(self):
        gguf = GGUFFile(version=3, metadata=[
            GGUFMetadataKV("tokenizer.ggml.add_bos_token", GGUFValueType.BOOL, True),
            GGUFMetadataKV("tokenizer.ggml.add_eos_token", GGUFValueType.BOOL, False),
        ])
        rt = self._roundtrip(gguf)
        self.assertTrue(rt.metadata[0].value)
        self.assertFalse(rt.metadata[1].value)

    def test_roundtrip_array_metadata(self):
        gguf = GGUFFile(version=3, metadata=[
            GGUFMetadataKV("tokenizer.ggml.tokens", GGUFValueType.ARRAY,
                           ["<unk>", "<s>", "</s>"]),
        ])
        rt = self._roundtrip(gguf)
        self.assertEqual(rt.metadata[0].value, ["<unk>", "<s>", "</s>"])

    def test_roundtrip_tensor_data(self):
        tensor_data = b"\x01\x02\x03\x04" * 8
        gguf = GGUFFile(version=3, tensor_data=tensor_data)
        rt = self._roundtrip(gguf)
        self.assertEqual(rt.tensor_data, tensor_data)

    def test_roundtrip_tensor_info(self):
        gguf = GGUFFile(version=3, tensors=[
            GGUFTensorInfo("token_embd.weight", [32000, 4096], ggml_type=0, offset=0),
        ], tensor_data=b"\xAB" * 32)
        rt = self._roundtrip(gguf)
        self.assertEqual(len(rt.tensors), 1)
        self.assertEqual(rt.tensors[0].name, "token_embd.weight")
        self.assertEqual(rt.tensors[0].dimensions, [32000, 4096])

    def test_magic_written_correctly(self):
        gguf = GGUFFile(version=3)
        data = GGUFWriter(gguf).write()
        magic = struct.unpack("<I", data[:4])[0]
        self.assertEqual(magic, GGUF_MAGIC)

    def test_alignment_padding(self):
        """Header bytes should be padded to alignment boundary before tensor data."""
        tensor_data = b"\xFF" * 16
        gguf = GGUFFile(version=3, alignment=32, tensor_data=tensor_data)
        data = GGUFWriter(gguf).write()
        # Find the tensor data by searching for our sentinel
        idx = data.index(b"\xFF" * 16)
        self.assertEqual(idx % 32, 0, "Tensor data must start at alignment boundary")


class TestXorDecrypt(unittest.TestCase):

    def test_xor_identity(self):
        plaintext = b"Hello, GGUF!"
        key = 42
        encrypted = xor_decrypt(plaintext, key)
        decrypted = xor_decrypt(encrypted, key)
        self.assertEqual(decrypted, plaintext)

    def test_xor_zero_key_noop(self):
        data = b"\x01\x02\x03"
        self.assertEqual(xor_decrypt(data, 0), data)

    def test_xor_key_255(self):
        data = b"\x00\xFF\xAA"
        result = xor_decrypt(data, 0xFF)
        self.assertEqual(result, bytes([0xFF, 0x00, 0x55]))

    def test_xor_invalid_key_raises(self):
        with self.assertRaises(ValueError):
            xor_decrypt(b"data", 256)
        with self.assertRaises(ValueError):
            xor_decrypt(b"data", -1)

    def test_detect_xor_key_correct(self):
        plain = _build_gguf()
        key = 77
        encrypted = xor_decrypt(plain, key)
        detected = detect_xor_key(encrypted)
        self.assertEqual(detected, key)

    def test_detect_xor_key_zero(self):
        """Key 0 means no encryption – detection should still return 0."""
        plain = _build_gguf()
        detected = detect_xor_key(plain)
        self.assertEqual(detected, 0)

    def test_detect_xor_key_none_for_garbage(self):
        # Random bytes that won't XOR to GGUF magic for any single-byte key
        # (we craft bytes so that none of the 256 XOR keys produce the magic)
        magic = struct.pack("<I", GGUF_MAGIC)
        # Build bytes that are NOT xor(magic, k) for any k (impossible to guarantee
        # in 4 bytes, so instead check detect returns something 0-255 or None)
        result = detect_xor_key(b"\x00\x00\x00\x00" * 2)
        # For all-zero bytes, xor(0, k)=k for each byte.
        # magic = b"GGUF" = 0x47 0x47 0x55 0x46
        # So the detected key should match only if all four equal the same k,
        # which they don't (0x47 != 0x55), so None is expected.
        self.assertIsNone(result)

    def test_detect_xor_key_short_data_returns_none(self):
        self.assertIsNone(detect_xor_key(b"\x01\x02"))


class TestMetadataHelpers(unittest.TestCase):

    def _make_gguf(self):
        return GGUFFile(version=3, metadata=[
            GGUFMetadataKV("general.name", GGUFValueType.STRING, "TestModel"),
            GGUFMetadataKV("tokenizer.chat_template", GGUFValueType.STRING, "old_template"),
        ])

    def test_get_existing_key(self):
        gguf = self._make_gguf()
        kv = get_metadata(gguf, "general.name")
        self.assertIsNotNone(kv)
        self.assertEqual(kv.value, "TestModel")

    def test_get_missing_key_returns_none(self):
        gguf = self._make_gguf()
        self.assertIsNone(get_metadata(gguf, "nonexistent.key"))

    def test_set_existing_key(self):
        gguf = self._make_gguf()
        set_metadata(gguf, "tokenizer.chat_template", "new_template")
        kv = get_metadata(gguf, "tokenizer.chat_template")
        self.assertEqual(kv.value, "new_template")
        # Count must not have grown
        self.assertEqual(sum(1 for kv in gguf.metadata
                             if kv.key == "tokenizer.chat_template"), 1)

    def test_set_new_key_appends(self):
        gguf = self._make_gguf()
        set_metadata(gguf, "llama.system_prompt", "You are a helpful assistant.")
        self.assertEqual(len(gguf.metadata), 3)
        kv = get_metadata(gguf, "llama.system_prompt")
        self.assertEqual(kv.value, "You are a helpful assistant.")

    def test_remove_existing_key(self):
        gguf = self._make_gguf()
        removed = remove_metadata(gguf, "general.name")
        self.assertTrue(removed)
        self.assertIsNone(get_metadata(gguf, "general.name"))
        self.assertEqual(len(gguf.metadata), 1)

    def test_remove_missing_key_returns_false(self):
        gguf = self._make_gguf()
        removed = remove_metadata(gguf, "nonexistent.key")
        self.assertFalse(removed)
        self.assertEqual(len(gguf.metadata), 2)


class TestRoundtripIntegration(unittest.TestCase):
    """End-to-end: build GGUF bytes → parse → modify → re-serialise → parse again."""

    def test_end_to_end_policy_change(self):
        """Simulate a full policy-change workflow."""
        # 1. Build a minimal GGUF with an original chat template
        original_template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        kvs = [("tokenizer.chat_template", GGUFValueType.STRING, original_template)]
        raw = _build_gguf(kvs=kvs)

        # 2. Parse
        gguf = GGUFReader(raw).read()
        self.assertEqual(get_metadata(gguf, "tokenizer.chat_template").value,
                         original_template)

        # 3. Modify policy
        new_template = "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}{% endfor %}"
        set_metadata(gguf, "tokenizer.chat_template", new_template)
        set_metadata(gguf, "llama.system_prompt", "Be concise.")

        # 4. Re-serialise
        rebuilt = GGUFWriter(gguf).write()

        # 5. Parse again and verify
        gguf2 = GGUFReader(rebuilt).read()
        self.assertEqual(get_metadata(gguf2, "tokenizer.chat_template").value, new_template)
        self.assertEqual(get_metadata(gguf2, "llama.system_prompt").value, "Be concise.")

    def test_end_to_end_decrypt_and_rebuild(self):
        """Simulate decrypt → modify → rebuild workflow."""
        # 1. Build a GGUF and XOR-encrypt it
        kvs = [("general.name", GGUFValueType.STRING, "SecretModel")]
        raw = _build_gguf(kvs=kvs)
        key = 0xA5
        encrypted = xor_decrypt(raw, key)

        # 2. Detect key and decrypt
        detected = detect_xor_key(encrypted)
        self.assertEqual(detected, key)
        decrypted = xor_decrypt(encrypted, detected)

        # 3. Parse and modify
        gguf = GGUFReader(decrypted).read()
        set_metadata(gguf, "general.name", "ModifiedModel")

        # 4. Rebuild
        rebuilt = GGUFWriter(gguf).write()
        gguf2 = GGUFReader(rebuilt).read()
        self.assertEqual(get_metadata(gguf2, "general.name").value, "ModifiedModel")

    def test_tensor_data_preserved_after_rebuild(self):
        """Tensor binary data must survive a parse-modify-rebuild cycle unchanged."""
        tensor_data = bytes(range(256)) * 4
        kvs = [("general.name", GGUFValueType.STRING, "TensorModel")]
        raw = _build_gguf(kvs=kvs, tensor_data=tensor_data)

        gguf = GGUFReader(raw).read()
        set_metadata(gguf, "general.name", "Modified")
        rebuilt = GGUFWriter(gguf).write()

        gguf2 = GGUFReader(rebuilt).read()
        self.assertEqual(gguf2.tensor_data, tensor_data)


if __name__ == "__main__":
    unittest.main()
