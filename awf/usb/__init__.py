"""Live-USB collector: shproto codec, serial reader, cumulative→delta collector.

Фаза 6 (замечание #PC-1, регистрационный номер #207). Модули пакета — Qt-free
логика (`shproto`, `collector`) + тонкая I/O-обёртка (`device`); UI живёт в
`awf/ui/tab_live.py`. Писатель `.aswf` — `awf/io/aswf_writer.py`.
"""

__all__ = ["shproto", "device", "collector"]
