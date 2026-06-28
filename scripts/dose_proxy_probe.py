"""Диагностика динамического диапазона дозового прокси P_k = Σ E_i·cps_i (кэВ/с) по срезам.
Цель — оценить, линеен ли масштаб между эталонами 2 и 75+ мЗв/ч прибора RC-103 (Задача #104).
Использование: python dose_proxy_probe.py <path.rcspg>"""
import sys
import numpy as np
from awf.io.rcspg_loader import load_rcspg

def main(path):
    sg = load_rcspg(path)
    en = sg.energies()                      # кэВ, длина n_channels
    cnt = sg.counts.astype(np.float64)      # [n_slices, n_channels]
    lt = np.asarray(sg.live_time_s, dtype=np.float64)
    lt = np.where(lt > 0, lt, np.nan)
    cps = cnt / lt[:, None]                 # отсчёты/с по каналам

    # отбрасываем последний канал (overflow, IV-R5)
    en2 = en[:-1]; cps2 = cps[:, :-1]
    total_cps = cps2.sum(axis=1)            # суммарная скорость счёта по срезу
    P = (cps2 * en2[None, :]).sum(axis=1)   # энерговклад, кэВ/с

    def stats(a, name):
        a = a[np.isfinite(a)]
        q = np.percentile(a, [0, 50, 90, 99, 100])
        print(f"{name}: min={q[0]:.4g} p50={q[1]:.4g} p90={q[2]:.4g} p99={q[3]:.4g} max={q[4]:.4g}")
    stats(total_cps, "total_cps (1/с)  ")
    stats(P,         "P=ΣE·cps (кэВ/с) ")

    kmax = int(np.nanargmax(P))
    print(f"\nпик P на срезе {kmax}, t={sg.time_offsets_s[kmax]:.1f} c, "
          f"total_cps={total_cps[kmax]:.4g}, P={P[kmax]:.4g} кэВ/с")
    pmed = np.nanmedian(P)
    print(f"медиана P = {pmed:.4g} кэВ/с; отношение пик/медиана = {P[kmax]/pmed:.1f}x")

    # Если пик == 75 мЗв/ч и масштаб линеен, уровню 2 мЗв/ч соответствует P ≈ пик*2/75:
    target_lo = P[kmax] * 2.0 / 75.0
    n_at_lo = int(np.sum(P >= target_lo))
    print(f"\nдля линейной гипотезы (пик=75 мЗв/ч): уровню 2 мЗв/ч отвечает P≈{target_lo:.4g} кэВ/с; "
          f"срезов с P≥этого: {n_at_lo} из {sg.n_slices}")
    print(f"диапазон P: пик/мин(>0) = {P[kmax]/np.nanmin(P[P>0]):.1f}x")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")