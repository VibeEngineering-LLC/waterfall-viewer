from __future__ import annotations

import pytest

from awf.usb.shproto import (
    CRC_INIT,
    SHPROTO_ESC,
    SHPROTO_FINISH,
    SHPROTO_START,
    ShprotoDecoder,
    ShprotoEncoder,
)


def test_constants_from_c_header():
    """Проверить константы дословно."""
    assert SHPROTO_START == 0xFE
    assert SHPROTO_ESC == 0xFD
    assert SHPROTO_FINISH == 0xA5
    assert CRC_INIT == 0xFFFF


def test_encode_empty_payload_roundtrip():
    """Проверить roundtrip для пустого payload."""
    encoded = ShprotoEncoder.encode(cmd=0x10, payload=b"")
    dec = ShprotoDecoder()
    dec.feed(encoded)
    frame = dec.take_frame()
    assert frame == (0x10, b"")


def test_encode_short_payload_roundtrip():
    """Проверить roundtrip для короткого payload."""
    encoded = ShprotoEncoder.encode(cmd=0x01, payload=b"hello")
    dec = ShprotoDecoder()
    dec.feed(encoded)
    frame = dec.take_frame()
    assert frame == (0x01, b"hello")


def test_encode_payload_contains_marker_bytes():
    """Проверить кодирование payload с маркерами."""
    payload = bytes([SHPROTO_START, SHPROTO_ESC, SHPROTO_FINISH, 0x00, 0xFF])
    encoded = ShprotoEncoder.encode(cmd=0x02, payload=payload)
    dec = ShprotoDecoder()
    dec.feed(encoded)
    frame = dec.take_frame()
    assert frame == (0x02, payload)


def test_encode_cmd_is_marker_byte():
    """Проверить кодирование cmd, совпадающего с маркером."""
    cmd = SHPROTO_START
    payload = b"\x01\x02"
    encoded = ShprotoEncoder.encode(cmd=cmd, payload=payload)
    dec = ShprotoDecoder()
    dec.feed(encoded)
    frame = dec.take_frame()
    assert frame == (cmd, payload)


def test_encode_frame_structure():
    """Проверить структуру закодированного кадра."""
    encoded = ShprotoEncoder.encode(0x50, b"")
    assert encoded[0] == 0xFF  # преамбула
    assert encoded[1] == SHPROTO_START
    assert encoded[-1] == SHPROTO_FINISH

    # проверить, что в теле нет raw маркеров
    body = encoded[2:-1]
    for b in body:
        assert b != SHPROTO_START
        assert b != SHPROTO_FINISH


def test_streaming_encoder_matches_static():
    """Сравнить статический и потоковый API."""
    static_out = ShprotoEncoder.encode(0x22, b"AB\xfeXY")
    enc = ShprotoEncoder()
    enc.packet_start(0x22)
    for b in b"AB\xfeXY":
        enc.packet_add_data(b)
    out = enc.packet_complete()
    assert out == static_out


def test_crc_failure_marks_dropped():
    """Проверить, что CRC-ошибка приводит к сбросу кадра."""
    enc = ShprotoEncoder.encode(0x33, b"data")
    broken = bytearray(enc)
    broken[3] ^= 0xAA
    dec = ShprotoDecoder()
    dec.feed(broken)
    frame = dec.take_frame()
    assert frame is None
    assert dec.dropped == 1


def test_short_frame_dropped():
    """Проверить, что короткий кадр отбрасывается."""
    dec = ShprotoDecoder()
    dec.feed(bytes([SHPROTO_START, SHPROTO_FINISH]))
    frame = dec.take_frame()
    assert frame is None
    assert dec.dropped == 1


def test_noise_before_start_ignored():
    """Проверить игнорирование шума перед START."""
    noise = bytes([0x11, 0x22, 0x33])
    cmd = 0x44
    payload = b"test"
    valid_frame = ShprotoEncoder.encode(cmd, payload)
    full_feed = noise + valid_frame
    dec = ShprotoDecoder()
    dec.feed(full_feed)
    frame = dec.take_frame()
    assert frame == (cmd, payload)
    assert dec.dropped == 0


def test_second_start_resets_buffer():
    """Проверить сброс буфера при повторном START."""
    first_partial = bytes([SHPROTO_START, 0x01, 0x02, 0x03, 0x04])
    second_valid = ShprotoEncoder.encode(0x44, b"ok")
    # обрезаем только преамбулу 0xFF; SHPROTO_START оставляем — он сбросит первый буфер
    second_valid_from_start = second_valid[1:]
    full_feed = first_partial + second_valid_from_start
    dec = ShprotoDecoder()
    dec.feed(full_feed)
    frame = dec.take_frame()
    assert frame == (0x44, b"ok")
    assert dec.dropped == 0


def test_feed_returns_count_of_new_frames():
    """Проверить возврат количества новых кадров из feed."""
    enc1 = ShprotoEncoder.encode(0x10, b"a")
    enc2 = ShprotoEncoder.encode(0x20, b"bc")
    enc3 = ShprotoEncoder.encode(0x30, b"")
    full_feed = enc1 + enc2 + enc3
    dec = ShprotoDecoder()
    n = dec.feed(full_feed)
    assert n == 3
    frames = dec.frames()
    assert len(frames) == 3
    assert frames[0] == (0x10, b"a")
    assert frames[1] == (0x20, b"bc")
    assert frames[2] == (0x30, b"")
