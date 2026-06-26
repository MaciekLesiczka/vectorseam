"""Tests for vector query binary message packing."""

import struct
import unittest

from vectorseam import (
    DType,
    encode_vector_message,
    encode_vector_message_from_list,
)


def _unpack_u32(frame: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<I", frame, offset)[0]


def _parse_frame(frame: bytes) -> dict[str, object]:
    name_len = _unpack_u32(frame, 16)
    vector_len = _unpack_u32(frame, 24)
    name_start = 28
    vector_start = name_start + name_len

    return {
        "frame_len": _unpack_u32(frame, 0),
        "magic": frame[4:8],
        "version": _unpack_u32(frame, 8),
        "dtype": _unpack_u32(frame, 12),
        "name_len": name_len,
        "dimension": _unpack_u32(frame, 20),
        "vector_len": vector_len,
        "name_bytes": frame[name_start:vector_start],
        "vector_bytes": frame[vector_start : vector_start + vector_len],
    }


class MessagePackingTest(unittest.TestCase):
    """Verifies protocol v1 frame encoding."""

    def test_encodes_packed_f32_memoryview(self) -> None:
        vector = memoryview(struct.pack("<ff", 4.0, 8.0))
        frame = encode_vector_message("raw", DType.F32, 2, vector)
        parsed = _parse_frame(frame)

        self.assertIsInstance(frame, bytes)
        self.assertEqual(len(frame) - 4, parsed["frame_len"])
        self.assertEqual(b"VQS1", parsed["magic"])
        self.assertEqual(1, parsed["version"])
        self.assertEqual(DType.F32.value, parsed["dtype"])
        self.assertEqual(3, parsed["name_len"])
        self.assertEqual(2, parsed["dimension"])
        self.assertEqual(8, parsed["vector_len"])
        self.assertEqual(b"raw", parsed["name_bytes"])
        self.assertEqual(vector.tobytes(), parsed["vector_bytes"])

    def test_encodes_packed_memoryview_for_all_dtypes(self) -> None:
        cases = (
            (DType.F32, 2, struct.pack("<ff", 1.0, -2.5)),
            (DType.F64, 2, struct.pack("<dd", 1.0, -2.5)),
            (DType.F16, 3, b"\x00<\x00@\x00B"),
            (DType.BF16, 3, b"\x80?\x00@\x40@"),
            (DType.I8, 4, b"\xff\x00\x01\x7f"),
            (DType.U8, 4, b"\x00\x01\x80\xff"),
        )

        for dtype, dimension, vector_bytes in cases:
            with self.subTest(dtype=dtype):
                frame = encode_vector_message(
                    "raw",
                    dtype,
                    dimension,
                    memoryview(vector_bytes),
                )
                parsed = _parse_frame(frame)

                self.assertEqual(dtype.value, parsed["dtype"])
                self.assertEqual(dimension, parsed["dimension"])
                self.assertEqual(dimension * dtype.byte_size, parsed["vector_len"])
                self.assertEqual(vector_bytes, parsed["vector_bytes"])

    def test_encoded_frame_is_immutable_after_source_mutation(self) -> None:
        vector = bytearray(struct.pack("<ff", 4.0, 8.0))
        frame = encode_vector_message("raw", DType.F32, 2, vector)
        vector[-1] ^= 0x01
        parsed = _parse_frame(frame)

        self.assertEqual(struct.pack("<ff", 4.0, 8.0), parsed["vector_bytes"])

    def test_encodes_small_f32_vector_from_list(self) -> None:
        frame = encode_vector_message_from_list("prod", [1.0, -2.5, 3.25])
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

    def test_encodes_non_ascii_name_as_utf8(self) -> None:
        name = "caf\u00e9"
        frame = encode_vector_message_from_list(name, [0.5])
        parsed = _parse_frame(frame)

        self.assertEqual(name.encode("utf-8"), parsed["name_bytes"])
        self.assertEqual(len(name.encode("utf-8")), parsed["name_len"])

    def test_accepts_name_at_1024_byte_limit(self) -> None:
        name = "x" * 1024
        frame = encode_vector_message(name, DType.U8, 1, b"\x01")
        parsed = _parse_frame(frame)

        self.assertEqual(1024, parsed["name_len"])
        self.assertEqual(name.encode("utf-8"), parsed["name_bytes"])

    def test_rejects_name_over_1024_bytes(self) -> None:
        with self.assertRaises(ValueError):
            encode_vector_message("x" * 1025, DType.U8, 1, b"\x01")

    def test_rejects_multibyte_name_over_1024_bytes(self) -> None:
        with self.assertRaises(ValueError):
            encode_vector_message("\u00e9" * 513, DType.U8, 1, b"\x01")

    def test_list_path_rejects_name_over_1024_bytes(self) -> None:
        with self.assertRaises(ValueError):
            encode_vector_message_from_list("x" * 1025, [1.0])

    def test_dtype_tracks_byte_size(self) -> None:
        self.assertEqual(4, DType.F32.byte_size)
        self.assertEqual(8, DType.F64.byte_size)
        self.assertEqual(2, DType.F16.byte_size)
        self.assertEqual(2, DType.BF16.byte_size)
        self.assertEqual(1, DType.I8.byte_size)
        self.assertEqual(1, DType.U8.byte_size)

    def test_list_path_rejects_empty_vector(self) -> None:
        with self.assertRaises(ValueError):
            encode_vector_message_from_list("empty", [])

    def test_rejects_non_string_name(self) -> None:
        with self.assertRaises(TypeError):
            encode_vector_message(123, DType.U8, 1, b"\x01")  # type: ignore[arg-type]

    def test_rejects_non_dtype(self) -> None:
        with self.assertRaises(TypeError):
            encode_vector_message("bad", 1, 1, b"\x01")  # type: ignore[arg-type]

    def test_list_path_rejects_non_dtype(self) -> None:
        with self.assertRaises(TypeError):
            encode_vector_message_from_list("bad", [1.0], dtype=1)  # type: ignore[arg-type]

    def test_list_path_rejects_unsupported_dtype(self) -> None:
        with self.assertRaises(NotImplementedError):
            encode_vector_message_from_list("future", [1.0], dtype=DType.F16)

    def test_list_path_rejects_f64_dtype(self) -> None:
        with self.assertRaises(NotImplementedError):
            encode_vector_message_from_list("future", [1.0], dtype=DType.F64)

    def test_rejects_wrong_bytes_length(self) -> None:
        with self.assertRaises(ValueError):
            encode_vector_message("raw", DType.F32, 2, b"\x00\x00\x00\x00")


if __name__ == "__main__":
    unittest.main()
