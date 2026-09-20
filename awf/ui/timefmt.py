"""Задача #UI-243: единое форматирование меток времени для всех шкал (2D-карта, нижний
график, 3D). Относительный режим — смещение от начала записи; абсолютный — фактические
дата/время, восстановленные из `Spectrogram.t0_iso` + смещение среза."""
from datetime import datetime, timedelta


def parse_t0(t0_iso):
    """ISO-8601 из `Spectrogram.t0_iso` → datetime. None, если метки нет или она не парсится (файл без времени старта: абсолютный режим тогда недоступен, шкалы остаются относительными)."""
    if not t0_iso:
        return None
    try:
        return datetime.fromisoformat(str(t0_iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def clock_label(t0, offset_s, sub_second=False):
    """Момент t0 + offset_s как «ЧЧ:ММ:СС» (дата выносится в подпись оси — в тик не влезает)."""
    if t0 is None:
        return ""
    ts = t0 + timedelta(seconds=float(offset_s))
    if sub_second:
        return ts.strftime("%H:%M:%S.%f")[:-3]
    else:
        return ts.strftime("%H:%M:%S")


def date_label(t0):
    """Дата начала записи для подписи оси: «ДД.ММ.ГГГГ». Пусто, если метки времени нет."""
    if t0 is None:
        return ""
    return t0.strftime("%d.%m.%Y")
