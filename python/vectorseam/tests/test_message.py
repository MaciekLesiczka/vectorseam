"""Tests for vector query binary message packing."""

import struct
import unittest
import zlib

from vectorseam import DType, encode_vector_message, encode_vector_message_le_bytes


def _unpack_u32(frame: bytes, offset: int) -> int:
    return struct.unpack_from("<I", frame, offset)[0]


def _parse_frame(frame: bytes) -> dict[str, object]:
    name_len = _unpack_u32(frame, 20)
    vector_len = _unpack_u32(frame, 28)
    name_start = 32
    vector_start = name_start + name_len

    return {
        "frame_len": _unpack_u32(frame, 0),
        "magic": frame[4:8],
        "version": _unpack_u32(frame, 8),
        "crc32": _unpack_u32(frame, 12),
        "dtype": _unpack_u32(frame, 16),
        "name_len": name_len,
        "dimension": _unpack_u32(frame, 24),
        "vector_len": vector_len,
        "name_bytes": frame[name_start:vector_start],
        "vector_bytes": frame[vector_start : vector_start + vector_len],
    }


def _crc_matches(frame: bytes) -> bool:
    expected_crc = _unpack_u32(frame, 12)
    actual_crc = zlib.crc32(frame[16:]) & 0xFFFFFFFF
    return expected_crc == actual_crc


class MessagePackingTest(unittest.TestCase):
    """Verifies protocol v1 frame encoding."""

    def test_encodes_small_f32_vector(self) -> None:
        frame = encode_vector_message("prod", [1.0, -2.5, 3.25])
        parsed = _parse_frame(frame)

        self.assertEqual(len(frame) - 4, parsed["frame_len"])
        self.assertEqual(b"VQS1", parsed["magic"])
        self.assertEqual(1, parsed["version"])
        self.assertEqual(DType.F32.value, parsed["dtype"])
        self.assertEqual(4, parsed["name_len"])
        self.assertEqual(3, parsed["dimension"])
        self.assertEqual(12, parsed["vector_len"])
        self.assertEqual(b"prod", parsed["name_bytes"])
        self.assertEqual(
            struct.pack("<fff", 1.0, -2.5, 3.25),
            parsed["vector_bytes"],
        )

    def test_encodes_small_f64_vector(self) -> None:
        frame = encode_vector_message("prod", [1.0, -2.5], dtype=DType.F64)
        parsed = _parse_frame(frame)

        self.assertEqual(DType.F64.value, parsed["dtype"])
        self.assertEqual(2, parsed["dimension"])
        self.assertEqual(16, parsed["vector_len"])
        self.assertEqual(struct.pack("<dd", 1.0, -2.5), parsed["vector_bytes"])

    def test_encodes_non_ascii_name_as_utf8(self) -> None:
        name = "caf\u00e9"
        frame = encode_vector_message(name, [0.5])
        parsed = _parse_frame(frame)

        self.assertEqual(name.encode("utf-8"), parsed["name_bytes"])
        self.assertEqual(len(name.encode("utf-8")), parsed["name_len"])

    def test_encodes_bytes_from_memoryview(self) -> None:
        vector = memoryview(struct.pack("<ff", 4.0, 8.0))
        frame = encode_vector_message_le_bytes("raw", DType.F32, 2, vector)
        parsed = _parse_frame(frame)

        self.assertEqual(DType.F32.value, parsed["dtype"])
        self.assertEqual(2, parsed["dimension"])
        self.assertEqual(8, parsed["vector_len"])
        self.assertEqual(vector.tobytes(), parsed["vector_bytes"])

    def test_dtype_tracks_byte_size(self) -> None:
        self.assertEqual(4, DType.F32.byte_size)
        self.assertEqual(8, DType.F64.byte_size)
        self.assertEqual(2, DType.F16.byte_size)
        self.assertEqual(2, DType.BF16.byte_size)
        self.assertEqual(1, DType.I8.byte_size)
        self.assertEqual(1, DType.U8.byte_size)

    def test_crc_matches_bytes_after_crc_field(self) -> None:
        frame = encode_vector_message("search", [1.0, 2.0])
        parsed = _parse_frame(frame)

        self.assertEqual(zlib.crc32(frame[16:]) & 0xFFFFFFFF, parsed["crc32"])
        self.assertTrue(_crc_matches(frame))

    def test_corrupted_frame_fails_crc_check(self) -> None:
        frame = bytearray(encode_vector_message("search", [1.0, 2.0]))
        frame[-1] ^= 0x01

        self.assertFalse(_crc_matches(bytes(frame)))

    def test_rejects_empty_vector(self) -> None:
        with self.assertRaises(ValueError):
            encode_vector_message("empty", [])

    def test_rejects_non_string_name(self) -> None:
        with self.assertRaises(TypeError):
            encode_vector_message(123, [1.0])  # type: ignore[arg-type]

    def test_rejects_non_dtype(self) -> None:
        with self.assertRaises(TypeError):
            encode_vector_message("bad", [1.0], dtype=1)  # type: ignore[arg-type]

    def test_rejects_unsupported_dtype(self) -> None:
        with self.assertRaises(NotImplementedError):
            encode_vector_message("future", [1.0], dtype=DType.F16)

    def test_rejects_wrong_bytes_length(self) -> None:
        with self.assertRaises(ValueError):
            encode_vector_message_le_bytes("raw", DType.F32, 2, b"\x00\x00\x00\x00")


if __name__ == "__main__":
    unittest.main()
