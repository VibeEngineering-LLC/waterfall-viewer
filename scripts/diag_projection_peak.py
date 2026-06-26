"""Диагностика #50: почему центральный пик «не считается» в проекции секущей плоскости.
A) текущая  = sum(z_surface, axis=time)        — сумма ДИСПЛЕЙНЫХ (log) высот
B) альтерн. = zscale(sum(z_counts, axis=time)) — дисплей ИНТЕГРАЛА сырых counts
Запуск: QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 python scripts/diag_projection_peak.py
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PySide6 import QtWidgets
from awf.model.spectrogram import Calibration, Spectrogram
from awf.ui.view3d import Waterfall3DView
from awf.ui.zscale import apply_z_scale

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

ns, nc = 60, 80
rng = np.random.RandomState(7)
counts = rng.poisson(30, size=(ns, nc)).astype(np.float64)   # ровный фон
counts[:, 10:30] += 40                       # широкая слабая полка (во всех срезах)
peak_ch = nc // 2                            # высокий узкий ТРАНЗИЕНТНЫЙ пик (вспышка)
burst = slice(ns // 2, ns // 2 + 3)          # всего 3 среза из 60
counts[burst, peak_ch] += 1500
counts[burst, peak_ch - 1] += 400
counts[burst, peak_ch + 1] += 400

cal = Calibration(coeffs=[0.0, 1.0])
t = np.arange(ns, dtype=np.float64) * 1.0
sg = Spectrogram(counts=counts.astype(np.int64), calibration=cal, time_offsets_s=t,
                 real_time_s=np.full(ns, 1.0), live_time_s=np.full(ns, 1.0))

v = Waterfall3DView()
v.set_spectrogram(sg, max_time=400, max_chan=512)

zc = v._z_counts        # (nt, nc) сырые counts бинов (LOD)
zs = v._z_surface       # (nt, nc) дисплейные высоты
nt2, nc2 = zc.shape
pc = int(np.argmax(zc.sum(axis=0)))   # канал-пик в LOD-индексах

A = zs.sum(axis=0)                     # A) текущая проекция
A = A / A.max() * v._height_scale
integ = zc.sum(axis=0)                 # B) альтернатива: zscale(integral counts)
B = apply_z_scale(integ, v._z_mode, gain=v._gain, gamma=v._gamma, clip=v._clip).astype(np.float64)
B = B / B.max() * v._height_scale

shelf = int(round(20 * nc2 / nc)); bg = 2
def rel(arr, c):
    return arr[c] / arr.max()
print(f"nt={nt2} nc={nc2} height_scale={v._height_scale:.2f} z_mode={v._z_mode}")
print(f"канал-пик LOD={pc}  полка≈{shelf}  фон={bg}")
print("ОТНОСИТЕЛЬНАЯ ВЫСОТА кривой (доля от макс.):")
print(f"  фон   : A={rel(A,bg):.3f}   B={rel(B,bg):.3f}")
print(f"  полка : A={rel(A,shelf):.3f}   B={rel(B,shelf):.3f}")
print(f"  ПИК   : A={rel(A,pc):.3f}   B={rel(B,pc):.3f}")
print(f"интегралы counts: фон={integ[bg]:.0f} полка={integ[shelf]:.0f} ПИК={integ[pc]:.0f}")
print(f"argmax(A)={int(np.argmax(A))}  argmax(B)={int(np.argmax(B))}  (пик={pc})")
