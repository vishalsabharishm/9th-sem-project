import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temporal_clip_buffer import Clip, TemporalBufferError, TemporalClipBuffer


def frame(value: int = 0, shape=(4, 4, 3), dtype=np.uint8) -> np.ndarray:
    return np.full(shape, fill_value=value, dtype=dtype)


class TemporalClipBufferConstructionTests(unittest.TestCase):
    def test_rejects_non_positive_clip_length(self):
        with self.assertRaises(TemporalBufferError):
            TemporalClipBuffer(clip_length=0)

    def test_rejects_non_positive_stride(self):
        with self.assertRaises(TemporalBufferError):
            TemporalClipBuffer(clip_length=4, stride=0)

    def test_rejects_malformed_expected_shape(self):
        with self.assertRaises(TemporalBufferError):
            TemporalClipBuffer(clip_length=4, expected_shape=(4, 4))


class TemporalClipBufferInsufficientFramesTests(unittest.TestCase):
    def test_not_ready_before_clip_length_reached(self):
        buffer = TemporalClipBuffer(clip_length=4)
        for i in range(3):
            buffer.add_frame(frame(i), i)
            self.assertFalse(buffer.is_ready())

    def test_get_clip_raises_when_not_ready(self):
        buffer = TemporalClipBuffer(clip_length=4)
        buffer.add_frame(frame(0), 0)
        with self.assertRaises(TemporalBufferError):
            buffer.get_clip()

    def test_len_reflects_buffered_count(self):
        buffer = TemporalClipBuffer(clip_length=4)
        self.assertEqual(len(buffer), 0)
        buffer.add_frame(frame(0), 0)
        buffer.add_frame(frame(1), 1)
        self.assertEqual(len(buffer), 2)


class TemporalClipBufferExactClipTests(unittest.TestCase):
    def test_ready_exactly_at_clip_length(self):
        buffer = TemporalClipBuffer(clip_length=4)
        for i in range(4):
            buffer.add_frame(frame(i), i)
        self.assertTrue(buffer.is_ready())

    def test_clip_preserves_frame_order_and_numbers(self):
        buffer = TemporalClipBuffer(clip_length=4)
        for i in range(4):
            buffer.add_frame(frame(i), frame_number=100 + i)

        clip = buffer.get_clip()
        self.assertIsInstance(clip, Clip)
        self.assertEqual(clip.frame_numbers, [100, 101, 102, 103])
        self.assertEqual(len(clip), 4)
        for i, buffered_frame in enumerate(clip.frames):
            self.assertTrue(np.array_equal(buffered_frame, frame(i)))

    def test_clip_frames_are_not_copied(self):
        buffer = TemporalClipBuffer(clip_length=2)
        first = frame(0)
        second = frame(1)
        buffer.add_frame(first, 0)
        buffer.add_frame(second, 1)

        clip = buffer.get_clip()
        self.assertIs(clip.frames[0], first)
        self.assertIs(clip.frames[1], second)

    def test_as_array_stacks_into_expected_shape(self):
        buffer = TemporalClipBuffer(clip_length=3, expected_shape=(4, 4, 3))
        for i in range(3):
            buffer.add_frame(frame(i), i)

        clip_array = buffer.get_clip().as_array()
        self.assertEqual(clip_array.shape, (3, 4, 4, 3))


class TemporalClipBufferRollingWindowTests(unittest.TestCase):
    def test_stride_one_produces_overlapping_sliding_windows(self):
        buffer = TemporalClipBuffer(clip_length=3, stride=1)
        for i in range(3):
            buffer.add_frame(frame(i), i)
        first_clip = buffer.get_clip()
        self.assertEqual(first_clip.frame_numbers, [0, 1, 2])

        self.assertFalse(buffer.is_ready())
        buffer.add_frame(frame(3), 3)
        self.assertTrue(buffer.is_ready())
        second_clip = buffer.get_clip()
        self.assertEqual(second_clip.frame_numbers, [1, 2, 3])

    def test_stride_equal_to_clip_length_produces_disjoint_windows(self):
        buffer = TemporalClipBuffer(clip_length=2, stride=2)
        buffer.add_frame(frame(0), 0)
        buffer.add_frame(frame(1), 1)
        self.assertTrue(buffer.is_ready())
        first_clip = buffer.get_clip()
        self.assertEqual(first_clip.frame_numbers, [0, 1])

        buffer.add_frame(frame(2), 2)
        self.assertFalse(buffer.is_ready())
        buffer.add_frame(frame(3), 3)
        self.assertTrue(buffer.is_ready())
        second_clip = buffer.get_clip()
        self.assertEqual(second_clip.frame_numbers, [2, 3])

    def test_ring_buffer_evicts_oldest_frame_beyond_clip_length(self):
        buffer = TemporalClipBuffer(clip_length=2, stride=1)
        for i in range(5):
            buffer.add_frame(frame(i), i)
        self.assertEqual(len(buffer), 2)
        clip = buffer.get_clip()
        self.assertEqual(clip.frame_numbers, [3, 4])

    def test_stride_greater_than_clip_length_discards_intervening_frames(self):
        # With clip_length=4 and stride=6, frames 0-5 are pushed. The
        # buffer is documented to discard frames that fall between emitted
        # windows rather than subsampling them: frames 0-1 are evicted
        # before the first clip becomes ready, so they never appear in any
        # clip this buffer returns.
        buffer = TemporalClipBuffer(clip_length=4, stride=6)
        for i in range(5):
            buffer.add_frame(frame(i), i)
            self.assertFalse(buffer.is_ready())

        buffer.add_frame(frame(5), 5)
        self.assertTrue(buffer.is_ready())
        clip = buffer.get_clip()
        self.assertEqual(clip.frame_numbers, [2, 3, 4, 5])


class TemporalClipBufferClearTests(unittest.TestCase):
    def test_clear_clears_buffered_frames(self):
        buffer = TemporalClipBuffer(clip_length=3)
        for i in range(3):
            buffer.add_frame(frame(i), i)
        self.assertTrue(buffer.is_ready())

        buffer.clear()
        self.assertFalse(buffer.is_ready())
        self.assertEqual(len(buffer), 0)

    def test_clear_allows_new_frame_shape_when_auto_detected(self):
        buffer = TemporalClipBuffer(clip_length=2)
        buffer.add_frame(frame(0, shape=(4, 4, 3)), 0)
        buffer.clear()
        # A different shape is accepted post-clear because the original
        # shape constraint was auto-detected, not explicitly configured.
        buffer.add_frame(frame(0, shape=(8, 8, 3)), 0)

    def test_clear_restores_explicit_shape_constraint(self):
        buffer = TemporalClipBuffer(clip_length=2, expected_shape=(4, 4, 3))
        buffer.add_frame(frame(0, shape=(4, 4, 3)), 0)
        buffer.clear()
        with self.assertRaises(TemporalBufferError):
            buffer.add_frame(frame(0, shape=(8, 8, 3)), 0)

    def test_clear_restarts_auto_frame_numbering(self):
        buffer = TemporalClipBuffer(clip_length=2)
        buffer.add_frame(frame(0))
        buffer.add_frame(frame(1))
        buffer.clear()
        buffer.add_frame(frame(2))
        buffer.add_frame(frame(3))
        clip = buffer.get_clip()
        self.assertEqual(clip.frame_numbers, [0, 1])


class TemporalClipBufferInvalidInputTests(unittest.TestCase):
    def test_rejects_non_ndarray_frame(self):
        buffer = TemporalClipBuffer(clip_length=2)
        with self.assertRaises(TemporalBufferError):
            buffer.add_frame([[0, 0], [0, 0]], 0)

    def test_rejects_empty_frame(self):
        buffer = TemporalClipBuffer(clip_length=2)
        with self.assertRaises(TemporalBufferError):
            buffer.add_frame(np.empty((0, 0, 3), dtype=np.uint8), 0)

    def test_rejects_wrong_number_of_dimensions(self):
        buffer = TemporalClipBuffer(clip_length=2)
        with self.assertRaises(TemporalBufferError):
            buffer.add_frame(np.zeros((4, 4), dtype=np.uint8), 0)

    def test_rejects_inconsistent_shape_after_first_frame(self):
        buffer = TemporalClipBuffer(clip_length=2)
        buffer.add_frame(frame(0, shape=(4, 4, 3)), 0)
        with self.assertRaises(TemporalBufferError):
            buffer.add_frame(frame(1, shape=(6, 6, 3)), 1)

    def test_rejects_inconsistent_dtype_after_first_frame(self):
        buffer = TemporalClipBuffer(clip_length=2)
        buffer.add_frame(frame(0, dtype=np.uint8), 0)
        with self.assertRaises(TemporalBufferError):
            buffer.add_frame(frame(1, dtype=np.float32), 1)

    def test_rejects_non_int_frame_number(self):
        buffer = TemporalClipBuffer(clip_length=2)
        with self.assertRaises(TemporalBufferError):
            buffer.add_frame(frame(0), "0")

    def test_rejected_first_call_does_not_corrupt_shape_or_dtype_state(self):
        # Regression test: a failed add_frame() call must have no partial
        # side effects. Previously, an invalid frame_number on the very
        # first call still let _validate_frame() auto-detect and lock in
        # the rejected frame's shape/dtype, even though nothing was ever
        # actually buffered.
        buffer = TemporalClipBuffer(clip_length=2)
        with self.assertRaises(TemporalBufferError):
            buffer.add_frame(frame(0, shape=(4, 4, 3)), "not-an-int")

        self.assertEqual(len(buffer), 0)
        self.assertIsNone(buffer._expected_shape)
        self.assertIsNone(buffer._expected_dtype)

        # A subsequent, legitimate first frame with a *different* shape
        # must be accepted normally -- nothing should have been locked in
        # by the rejected call above.
        buffer.add_frame(frame(0, shape=(8, 8, 3)), 0)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer._expected_shape, (8, 8, 3))


class TemporalClipBufferAutoFrameNumberTests(unittest.TestCase):
    def test_add_frame_without_frame_number_auto_assigns_sequence(self):
        buffer = TemporalClipBuffer(clip_length=3)
        buffer.add_frame(frame(0))
        buffer.add_frame(frame(1))
        buffer.add_frame(frame(2))
        clip = buffer.get_clip()
        self.assertEqual(clip.frame_numbers, [0, 1, 2])

    def test_explicit_frame_number_does_not_disturb_auto_sequence(self):
        buffer = TemporalClipBuffer(clip_length=3)
        buffer.add_frame(frame(0))
        buffer.add_frame(frame(1), frame_number=999)
        buffer.add_frame(frame(2))
        clip = buffer.get_clip()
        self.assertEqual(clip.frame_numbers, [0, 999, 1])


class TemporalClipBufferRealisticFrameShapeTests(unittest.TestCase):
    """Exercises the buffer with frames shaped like real OpenCV output.

    ``cv2.VideoCapture.read()`` yields BGR uint8 arrays of shape
    (height, width, channels), e.g. (240, 320, 3) for this project's
    ``data/sample_video.mp4`` fixture. The other tests use a tiny
    synthetic (4, 4, 3) shape for speed; this test uses a shape and dtype
    representative of the real pipeline.
    """

    def test_accepts_and_stacks_opencv_shaped_bgr_frames(self):
        buffer = TemporalClipBuffer(clip_length=4, stride=2)
        opencv_shape = (240, 320, 3)
        rng = np.random.default_rng(seed=0)

        clip = None
        for i in range(4):
            random_bgr_frame = rng.integers(
                0, 256, size=opencv_shape, dtype=np.uint8
            )
            buffer.add_frame(random_bgr_frame, i)
            if buffer.is_ready():
                clip = buffer.get_clip()

        self.assertIsNotNone(clip)
        self.assertEqual(clip.frame_numbers, [0, 1, 2, 3])
        stacked = clip.as_array()
        self.assertEqual(stacked.shape, (4, 240, 320, 3))
        self.assertEqual(stacked.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
