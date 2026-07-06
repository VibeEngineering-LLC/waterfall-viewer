from __future__ import annotations

from pathlib import Path

import pytest

from awf.model.series import RecordingSeries, SeriesPhase, SERIES_DEFAULT_GAP_SEC


def test_init_defaults(tmp_path):
    """Test default initialization and basic properties."""
    s = RecordingSeries(durations_sec=[10.0, 20.0], out_dir=tmp_path)
    assert s.n_tracks == 2
    assert s.phase == SeriesPhase.IDLE
    assert s.current_index == 0
    assert s.started_at is None
    assert not s.is_running
    assert not s.is_done
    assert not s.errored
    assert s.current_file is None
    assert s.current_duration_sec is None
    assert s.remaining_sec(0.0) is None
    assert s.progress() == (0, 2)
    assert s.gap_sec == SERIES_DEFAULT_GAP_SEC
    assert s.file_prefix == "series"
    assert isinstance(s.out_dir, Path)


def test_init_invalid_empty_durations(tmp_path):
    """Test initialization with empty durations list raises ValueError."""
    with pytest.raises(ValueError):
        RecordingSeries(durations_sec=[], out_dir=tmp_path)


def test_init_invalid_nonpositive_duration(tmp_path):
    """Test initialization with zero or negative duration raises ValueError."""
    with pytest.raises(ValueError):
        RecordingSeries(durations_sec=[10.0, 0.0], out_dir=tmp_path)
    with pytest.raises(ValueError):
        RecordingSeries(durations_sec=[-1.0], out_dir=tmp_path)


def test_init_invalid_gap_negative(tmp_path):
    """Test initialization with negative gap_sec raises ValueError."""
    with pytest.raises(ValueError):
        RecordingSeries(durations_sec=[10.0], out_dir=tmp_path, gap_sec=-1.0)


def test_init_invalid_prefix_empty(tmp_path):
    """Test initialization with empty file_prefix raises ValueError."""
    with pytest.raises(ValueError):
        RecordingSeries(durations_sec=[10.0], out_dir=tmp_path, file_prefix="")


def test_start_transitions_to_recording(tmp_path):
    """Test starting series transitions to recording phase."""
    s = RecordingSeries(durations_sec=[5.0, 7.0], out_dir=tmp_path)
    s.start(now=1000.0)
    assert s.phase == SeriesPhase.RECORDING
    assert s.current_index == 0
    assert s.started_at == 1000.0
    assert s.is_running
    assert not s.is_done
    assert s.current_file == tmp_path / "series_000"
    assert s.current_duration_sec == 5.0
    assert s.remaining_sec(1000.0) == 5.0
    assert s.remaining_sec(1002.5) == 2.5
    assert s.remaining_sec(1010.0) == 0.0


def test_start_twice_raises(tmp_path):
    """Test starting series twice raises RuntimeError."""
    s = RecordingSeries(durations_sec=[5.0], out_dir=tmp_path)
    s.start(now=100.0)
    with pytest.raises(RuntimeError):
        s.start(now=200.0)


def test_tick_no_gap_progresses_through_tracks(tmp_path):
    """Test ticking without gap progresses through tracks."""
    s = RecordingSeries(durations_sec=[5.0, 7.0], out_dir=tmp_path, gap_sec=0.0)
    s.start(now=1000.0)
    assert s.tick(1004.9) == SeriesPhase.RECORDING
    assert s.current_index == 0
    assert s.tick(1005.0) == SeriesPhase.RECORDING
    assert s.current_index == 1
    assert s.current_file == tmp_path / "series_001"
    assert s.current_duration_sec == 7.0
    assert s.remaining_sec(1005.0) == 7.0
    assert s.tick(1011.9) == SeriesPhase.RECORDING
    assert s.tick(1012.0) == SeriesPhase.DONE
    assert s.current_index == 2
    assert s.is_done
    assert s.current_file is None
    assert s.current_duration_sec is None
    assert s.remaining_sec(9999.0) is None


def test_tick_with_gap_alternates_recording_pause(tmp_path):
    """Test ticking with gap alternates recording and pause phases."""
    s = RecordingSeries(durations_sec=[5.0, 5.0, 5.0], out_dir=tmp_path, gap_sec=2.0)
    s.start(now=0.0)
    assert s.tick(5.0) == SeriesPhase.PAUSE
    assert s.current_index == 0
    assert s.current_file is None
    assert s.current_duration_sec == 2.0
    assert s.remaining_sec(5.0) == 2.0
    assert s.tick(6.0) == SeriesPhase.PAUSE
    assert s.tick(7.0) == SeriesPhase.RECORDING
    assert s.current_index == 1
    assert s.current_file == tmp_path / "series_001"
    assert s.tick(19.0) == SeriesPhase.DONE
    assert s.current_index == 3


def test_tick_gap_skipped_after_last_track(tmp_path):
    """Test that gap is skipped after last track."""
    s = RecordingSeries(durations_sec=[3.0, 3.0], out_dir=tmp_path, gap_sec=10.0)
    s.start(now=0.0)
    assert s.tick(3.0) == SeriesPhase.PAUSE
    assert s.tick(13.0) == SeriesPhase.RECORDING
    assert s.current_index == 1
    assert s.tick(16.0) == SeriesPhase.DONE


def test_tick_large_jump_skips_multiple_phases(tmp_path):
    """Test that large tick jumps skip multiple phases."""
    s = RecordingSeries(durations_sec=[1.0, 1.0, 1.0, 1.0], out_dir=tmp_path, gap_sec=0.0)
    s.start(now=0.0)
    assert s.tick(4.0) == SeriesPhase.DONE
    assert s.current_index == 4


def test_mark_error_stop_on_error_default(tmp_path):
    """Test that error stops series by default."""
    s = RecordingSeries(durations_sec=[5.0, 5.0, 5.0], out_dir=tmp_path)
    s.start(now=0.0)
    s.tick(5.0)
    assert s.current_index == 1
    s.mark_error(now=6.0)
    assert s.phase == SeriesPhase.DONE
    assert s.is_done
    assert s.errored


def test_mark_error_no_stop_advances(tmp_path):
    """Test that error does not stop series when stop_on_error=False."""
    s = RecordingSeries(durations_sec=[5.0, 5.0], out_dir=tmp_path, gap_sec=0.0, stop_on_error=False)
    s.start(now=0.0)
    s.mark_error(now=1.0)
    assert not s.errored
    assert s.tick(5.0) == SeriesPhase.RECORDING
    assert s.current_index == 1


def test_reset_returns_to_idle(tmp_path):
    """Test that reset returns series to idle state."""
    s = RecordingSeries(durations_sec=[5.0], out_dir=tmp_path)
    s.start(now=100.0)
    s.tick(6.0 + 100.0)
    assert s.is_done
    s.reset()
    assert s.phase == SeriesPhase.IDLE
    assert s.current_index == 0
    assert s.started_at is None
    assert not s.errored
    s.start(now=200.0)
    assert s.phase == SeriesPhase.RECORDING
    assert s.started_at == 200.0
