from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from awf.model.spectrogram import Calibration, Spectrogram
from awf.io.n42_loader import (
    load_n42, parse_iso_duration,
    decode_counted_zeroes_scalar, decode_counted_zeroes_vec,
)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "waterfall_sample.n42"
COEFFS = [3.52287433, 0.383773901, 2.67593756e-05, -6.49007438e-09, 5.67558676e-13]
requires_sample = pytest.mark.skipif(not SAMPLE.exists(),
    reason="нет образца sample_data/waterfall_sample.n42")

def _make_wellformed_stream(rng, n_items):
    # Возвращает (tokens_list, expected_decoded_list).
    # Каждый элемент — либо литерал (ненулевое 1..50), либо нулевой пробег (маркер 0 + count 1..20).
    toks = []; expected = []
    for _ in range(n_items):
        if rng.random() < 0.5:
            v = int(rng.integers(1, 51)); toks.append(v); expected.append(v)
        else:
            c = int(rng.integers(1, 21)); toks.extend([0, c]); expected.extend([0] * c)
    return toks, expected

def test_parse_iso_duration():
    assert parse_iso_duration("PT10S") == 10.0
    assert parse_iso_duration("PT1M30S") == 90.0
    assert parse_iso_duration("P1DT2H") == 93600.0
    assert parse_iso_duration("") is None
    assert parse_iso_duration(None) is None
    assert parse_iso_duration("мусор") is None

def test_decode_hand_cases():
    # (строка токенов -> ожидаемый декод)
    cases = [
        ("5 0 3 7", [5, 0, 0, 0, 7]),
        ("0 4", [0, 0, 0, 0]),
        ("1 2 3", [1, 2, 3]),
        ("0 1", [0]),
        ("", []),
    ]
    for text, expected in cases:
        toks = [int(x) for x in text.split()]
        assert decode_counted_zeroes_scalar(toks) == expected
        vec = decode_counted_zeroes_vec(np.asarray(toks, dtype=np.int64))
        assert vec.tolist() == expected

def test_decode_vec_matches_scalar_fuzz():
    # actor != verifier: векторный декодер сверяется с независимым скалярным эталоном
    # на множестве случайных well-formed потоков. Фиксированный seed -> воспроизводимо.
    rng = np.random.default_rng(12345)
    for _ in range(200):
        toks, expected = _make_wellformed_stream(rng, int(rng.integers(0, 60)))
        arr = np.asarray(toks, dtype=np.int64)
        sca = decode_counted_zeroes_scalar(toks)
        vec = decode_counted_zeroes_vec(arr).tolist()
        assert sca == expected
        assert vec == expected

def test_calibration_polynomial():
    cal = Calibration.from_coeff_string(" ".join(repr(c) for c in COEFFS))
    en = cal.energies(8192)
    assert en.shape == (8192,)
    assert en[0] == pytest.approx(COEFFS[0], abs=1e-6)          # E(0) == c0
    assert en[4096] == pytest.approx(1738.1676, abs=1e-2)
    assert en[8191] == pytest.approx(3930.5273, abs=1e-2)
    assert bool(np.all(np.diff(en) > 0))                        # строго возрастает
    # скаляр vs массив согласованы
    assert cal.energy_of_channel(0) == pytest.approx(en[0], abs=1e-9)
    assert cal.energy_of_channel(8191) == pytest.approx(en[8191], abs=1e-6)

def test_channel_of_energy_roundtrip():
    cal = Calibration.from_coeff_string(" ".join(repr(c) for c in COEFFS))
    en = cal.energies(8192)
    for ch in (0, 100, 1000, 4096, 8000, 8191):
        e = float(en[ch])
        got = int(cal.channel_of_energy(e, 8192))
        assert abs(got - ch) <= 1                               # обратное преобразование с точностью +-1 канал

@requires_sample
def test_load_real_sample():
    sg = load_n42(SAMPLE)
    assert sg.n_slices == 256
    assert sg.n_channels == 8192
    assert sg.counts.dtype == np.uint16
    assert int(sg.counts.max()) == 22
    row_sums = sg.counts.sum(axis=1, dtype=np.int64)
    assert int(row_sums.min()) == 5962
    assert int(row_sums.max()) == 7043
    np.testing.assert_allclose(sg.calibration.coeffs, COEFFS, rtol=0, atol=0)
    # временная ось: шаг 10 с, всего 256 срезов -> последний оффсет 2550 с
    assert sg.time_offsets_s[0] == pytest.approx(0.0)
    assert sg.time_offsets_s[1] == pytest.approx(10.0)
    assert sg.time_offsets_s[-1] == pytest.approx(2550.0)
    assert np.all(sg.real_time_s == 10.0)
    assert np.all(sg.live_time_s == 10.0)
    assert sg.t0_iso == "2026-06-24T19:04:08Z"

@requires_sample
def test_load_max_slices():
    sg = load_n42(SAMPLE, max_slices=10)
    assert sg.n_slices == 10
    assert sg.n_channels == 8192

@requires_sample
def test_analysis_primitives():
    sg = load_n42(SAMPLE)
    total = int(sg.counts.sum(dtype=np.int64))
    # интегральный спектр
    assert int(sg.total_spectrum().sum()) == total
    # срез по времени == строка матрицы
    assert np.array_equal(sg.energy_spectrum(0), sg.counts[0])
    # сечение по каналу
    assert sg.channel_time_series(100).shape == (256,)
    # прямоугольная выборка по всей матрице == total
    assert sg.roi_sum(0, sg.n_slices, 0, sg.n_channels) == total
    # band по диапазону каналов
    band = sg.band_time_series(0, sg.n_channels)
    assert int(band.sum()) == total
    # LOD: max-pool сохраняет глобальный максимум; sum-pool сохраняет сумму
    ds_max, tc, cc = sg.downsample(50, 500, method="max")
    assert ds_max.shape[0] <= 50 and ds_max.shape[1] <= 500
    assert float(ds_max.max()) == 22.0
    assert tc.shape[0] == ds_max.shape[0] and cc.shape[0] == ds_max.shape[1]
    ds_sum, _, _ = sg.downsample(50, 500, method="sum")
    assert float(ds_sum.sum()) == pytest.approx(float(total))
    with pytest.raises(ValueError):
        sg.downsample(10, 10, method="bad")
