# The default stft-transformation doesn't work with tflite, so a replacement is needed. Taken from:
# https://github.com/antonyharfield/tflite-models-audioset-yamnet/blob/master/features_tflite.py
# With updates from: https://github.com/tensorflow/tensorflow/issues/27303#issuecomment-675008946
# And from: https://github.com/tensorflow/tensorflow/issues/27303#issuecomment-577361337

import numpy as np
import tensorflow as tf

# ==================================================================================================


def _dft_matrix(dft_length):
    """Calculate the full DFT matrix in numpy."""
    omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
    # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
    return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))


# ==================================================================================================


def _naive_rdft(signal_tensor, fft_length, padding="center"):
    """Implement real-input Fourier Transform by matmul."""
    # We are right-multiplying by the DFT matrix, and we are keeping
    # only the first half ("positive frequencies").
    # So discard the second half of rows, but transpose the array for
    # right-multiplication.
    # The DFT matrix is symmetric, so we could have done it more
    # directly, but this reflects our intention better.
    complex_dft_matrix_kept_values = _dft_matrix(fft_length)[
        : (fft_length // 2 + 1), :
    ].transpose()
    real_dft_tensor = tf.constant(
        np.real(complex_dft_matrix_kept_values).astype(np.float32),
        name="real_dft_matrix",
    )
    imag_dft_tensor = tf.constant(
        np.imag(complex_dft_matrix_kept_values).astype(np.float32),
        name="imaginary_dft_matrix",
    )
    signal_frame_length = signal_tensor.shape[-1]  # .value
    half_pad = (fft_length - signal_frame_length) // 2

    if padding == "center":
        # Center-padding.
        pad_values = tf.concat(
            [
                tf.zeros([tf.rank(signal_tensor) - 1, 2], tf.int32),
                [[half_pad, fft_length - signal_frame_length - half_pad]],
            ],
            axis=0,
        )
    elif padding == "right":
        # Right-padding.
        pad_values = tf.concat(
            [
                tf.zeros([tf.rank(signal_tensor) - 1, 2], tf.int32),
                [[0, fft_length - signal_frame_length]],
            ],
            axis=0,
        )

    padded_signal = tf.pad(signal_tensor, pad_values)

    result_real_part = tf.matmul(padded_signal, real_dft_tensor)
    result_imag_part = tf.matmul(padded_signal, imag_dft_tensor)

    return result_real_part, result_imag_part


# ==================================================================================================


def stft_tflite(signal, frame_length, frame_step, fft_length):
    """tflite-compatible implementation of tf.signal.stft.
    Compute the short-time Fourier transform of a 1D input while avoiding tf ops
    that are not currently supported in tflite (Rfft, Range, SplitV).
    fft_length must be fixed. A Hann window is of frame_length is always
    applied.
    Since fixed (precomputed) framing must be used, signal.shape[-1] must be a
    specific value (so "?"/None is not supported).
    Args:
        signal: 1D tensor containing the time-domain waveform to be transformed.
        frame_length: int, the number of points in each Fourier frame.
        frame_step: int, the number of samples to advance between successive frames.
        fft_length: int, the size of the Fourier transform to apply.
    Returns:
        Two (num_frames, fft_length) tensors containing the real and imaginary parts
        of the short-time Fourier transform of the input signal.
    """
    # Make the window be shape (1, frame_length) instead of just frame_length
    # in an effort to help the tflite broadcast logic.
    window = tf.reshape(
        tf.constant(
            (
                0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
            ).astype(np.float32),
            name="window",
        ),
        [1, tf.constant(frame_length, dtype=tf.int32)],
    )

    framed_signal = tf.signal.frame(signal, frame_length, frame_step)
    framed_signal *= window

    real_spectrogram, imag_spectrogram = _naive_rdft(
        framed_signal, fft_length, padding="right"
    )

    return real_spectrogram, imag_spectrogram


# ==================================================================================================


def stft_magnitude_tflite(signals, frame_length, frame_step, fft_length):
    """Calculate spectrogram avoiding tflite incompatible ops."""
    real_stft, imag_stft = stft_tflite(signals, frame_length, frame_step, fft_length)
    stft_magnitude = tf.sqrt(
        tf.add(real_stft * real_stft, imag_stft * imag_stft),
        name="magnitude_spectrogram",
    )

    return stft_magnitude
