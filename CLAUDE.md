# CLAUDE.md — указатель (проект `atomspectra-waterfall-viewer`)

Полные рабочие заметки для разработки вынесены в **[`docs/dev/`](docs/dev/)**, чтобы корень
публичного репозитория оставался чистым для пользователей. Этот файл — тонкая заглушка: она
сохраняет автозагрузку контекста Claude Code (файл обязан лежать в корне и называться `CLAUDE.md`)
и ведёт на детальные документы.

## Куда смотреть

| Документ | Путь |
|---|---|
| Полные проектные инструкции + журнал задач `#1..#134`, `#REL-*` | [`docs/dev/CLAUDE_NOTES.md`](docs/dev/CLAUDE_NOTES.md) |
| Подробный журнал всех задач (200 КБ) | [`docs/dev/TASKS.md`](docs/dev/TASKS.md) |
| План реализации | [`docs/dev/IMPLEMENTATION_PLAN.md`](docs/dev/IMPLEMENTATION_PLAN.md) |
| Известные проблемы и совместимость | [`docs/dev/KNOWN_ISSUES.md`](docs/dev/KNOWN_ISSUES.md) |
| Ранние спецификации модулей | [`docs/dev/_specs/`](docs/dev/_specs/) |

## Критичные инварианты (перед любой правкой прочитать `docs/dev/CLAUDE_NOTES.md`)

- **Публичный репозиторий** `VibeEngineering-LLC/atomspectra-waterfall-viewer` (§24). Push — **только**
  по явной команде оператора. Никогда не коммитить серийники приборов, абсолютные пути оператора,
  IP/MAC/UUID/GPS/PII, e-mail, токены. `git add` — конкретные пути, **никогда** `-A`/`.`. Перед
  коммитом — скан на секреты/PII. Реальные данные спектрометра (`*.n42`/`*.rcspg`/`*.aswf`) и
  `awf/data/iaea_cache/` — gitignored, в репо не попадают.
- **Зоны агентов (§12):** это моя зона (скилл `atomspectra-waterfall-viewer-dev`). Чужие зоны
  (ESP32, SpectraVibe `gamma-spectrum-analysis` и т.п.) — read-only / писать промпт владельцу.
- **Тесты:** `pytest` под headless-Qt — `QT_QPA_PLATFORM=offscreen`, `PYTHONIOENCODING=utf-8`.
  Полный прогон на момент `#REL-*` — 535 passed (локально, со scipy + sample-файлом).
- **Зависимости:** `scipy` — жёсткая production-зависимость (`awf/analysis/peaks.py` `curve_fit`,
  `awf/analysis/deconvolve.py` `lsq_linear`), прописана в `requirements.txt` и `pyproject.toml`.
- **Общение с оператором — только на русском** (глобальный §5).
