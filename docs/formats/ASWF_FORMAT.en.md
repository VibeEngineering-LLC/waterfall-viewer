# ASWF File Format — Integration Specification

[🇷🇺 Русская версия](ASWF_FORMAT.md)

**ASWF** (AtomSpectra Waterfall Format) is a binary format for storing a spectrogram
(waterfall): a time series of spectra captured at fixed intervals.
Each `.aswf` file is self-contained — it carries its full header and data, requiring
no external schema to parse.

---

## File Structure

```
┌────────────────────────────────────────────────────────┐
│  Offset 0 │ 4 bytes │ Magic number "ASWF" (ASCII)      │
│  Offset 4 │ 4 bytes │ uint32 LE: JSON area length       │
│  Offset 8 │ N bytes │ JSON header (UTF-8 + spaces pad)  │
│  Offset 8+N         │ Data rows (payload)                │
└────────────────────────────────────────────────────────┘
```

| Field         | Offset   | Size     | Type      | Value                           |
|---------------|----------|----------|-----------|---------------------------------|
| Magic         | 0        | 4        | ASCII     | `41 53 57 46` = `"ASWF"`       |
| `hlen`        | 4        | 4        | uint32 LE | Reserved JSON area size         |
| JSON header   | 8        | `hlen`   | UTF-8     | Metadata (space-padded to hlen) |
| Payload       | 8+`hlen` | N×stride | binary    | Spectrum rows, oldest first     |

**Important:** `hlen` is the **reserved** (padded) JSON area size, not the actual
JSON text length. The current value is always `4096`. Parsers must use `hlen` to
compute the payload offset — do not scan for `}`.

---

## JSON Header

The header is fully self-describing: every field needed for decoding is present
inside the file — no external schema required.

### Header Fields

| Key            | Type             | Required | Description |
|----------------|------------------|:---:|------|
| `format`       | string           | yes | Always `"atomspectra-waterfall"` |
| `version`      | int              | yes | Format version: `1` or `2` |
| `channels`     | int              | yes | Channels per row; current value `8192` |
| `dtype`        | string           | yes | Element type: `"uint16"` |
| `byte_order`   | string           | yes | Byte order: `"little"` |
| `row_stride`   | int              | v2  | Row size in bytes `16386` (v2); absent in v1 |
| `row_time`     | object           | v2  | Duration field descriptor (see below) |
| `interval_sec` | int              | yes | Nominal recording interval, seconds |
| `started_at`   | int (unix ts)    | yes | Timestamp of the first row (UTC, epoch seconds) |
| `saved_rows`   | int              | yes | Rows in the file; `0` for an open (unfinalised) segment |
| `saved_at`     | int (unix ts)    | yes | Finalisation timestamp; `0` for an open segment |
| `serial`       | string           | no  | Device serial number |
| `calibration`  | array of floats  | no  | Energy calibration polynomial coefficients |

#### `row_time` object (v2 only)

```json
{
  "dtype":  "uint16",
  "unit":   "sec",
  "offset": 16384
}
```

`offset` is the byte offset of the duration field **within each row**.

### Example v2 Header

```json
{
  "saved_rows": 660,
  "saved_at": 1783198621,
  "format": "atomspectra-waterfall",
  "version": 2,
  "channels": 8192,
  "dtype": "uint16",
  "byte_order": "little",
  "row_stride": 16386,
  "row_time": {"dtype": "uint16", "unit": "sec", "offset": 16384},
  "interval_sec": 60,
  "started_at": 1783157403,
  "serial": "AS-001",
  "calibration": [0.0, 0.298, 0.0]
}
```

---

## Row Formats

### v1 — 16384 bytes per row

```
Bytes [0 .. 16383]:  8192 × uint16 LE — counts per interval, channel by channel
```

Effective row duration = `interval_sec` (nominal).

### v2 — 16386 bytes per row (current)

```
Bytes [0     .. 16383]:  8192 × uint16 LE — counts per interval
Bytes [16384 .. 16385]:  uint16 LE — actual live time, seconds
```

The duration field holds the **actual live time of the detector** for this interval.
If the value is `0`, substitute `interval_sec`.

### Version Detection

```python
is_v2  = "row_stride" in header
stride = header.get("row_stride", header["channels"] * 2)
```

### Row Count

```python
n_rows = (file_size - (8 + hlen)) // stride
```

If `saved_rows` in the header is `0` (unfinalised segment), compute `n_rows`
from the formula above.

---

## Spectrum Data

Each channel value in a row is a **delta** of the cumulative spectrum: the number of
pulses registered during this interval. Type `uint16` — values `0` to `65535`.

### Absolute Spectrum (sum over all rows)

```python
absolute = [0] * channels
for row in rows:
    for ch in range(channels):
        absolute[ch] += row[ch]
```

### Count Rate (counts/s) for Row i

```python
dur_i = duration[i] if duration[i] > 0 else interval_sec
rate  = [row[i][ch] / dur_i for ch in range(channels)]
```

---

## Row Timestamps

```
t[0] = started_at
t[i] = started_at + sum(duration[0..i-1])
```

For any row where `duration[j] == 0`, substitute `interval_sec` in the sum.

In v1, `duration[j] == interval_sec` for all rows.

---

## Energy Calibration: Channel → keV

If `calibration` is present, energy is computed as a polynomial:

```
E(ch) = a[0] + a[1]·ch + a[2]·ch² + …
```

where `a = calibration` (coefficient array, index = polynomial degree).

Example from the header above: `E(ch) = 0.0 + 0.298·ch` — linear scale.

If absent, the energy scale is unknown; use channel number as the X axis.

---

## Python: Minimal Parser

```python
import json
import struct
from pathlib import Path

def read_aswf(path):
    """Read an .aswf file, return (header, rows, durations).

    rows       -- list of N tuples, each with `channels` uint16 values.
    durations  -- list of N uint16 values (0 = use interval_sec).
                  For v1 all values equal header["interval_sec"].
    """
    buf   = Path(path).read_bytes()
    magic = buf[:4]
    if magic != b"ASWF":
        raise ValueError(f"Not an ASWF file: magic={magic!r}")

    hlen   = struct.unpack_from("<I", buf, 4)[0]
    header = json.loads(buf[8:8 + hlen].decode("utf-8"))

    ch     = header["channels"]
    stride = header.get("row_stride", ch * 2)
    is_v2  = "row_stride" in header

    payload = buf[8 + hlen:]
    n_rows  = len(payload) // stride

    rows      = []
    durations = []
    for i in range(n_rows):
        off  = i * stride
        row  = struct.unpack_from(f"<{ch}H", payload, off)
        rows.append(row)
        if is_v2:
            dur = struct.unpack_from("<H", payload, off + ch * 2)[0]
        else:
            dur = header["interval_sec"]
        durations.append(dur)

    return header, rows, durations


def row_timestamp(header, durations, index):
    """Unix timestamp of the start of row `index`."""
    ts = header["started_at"]
    iv = header["interval_sec"]
    for j in range(index):
        ts += durations[j] if durations[j] > 0 else iv
    return ts


def channel_to_kev(header, ch):
    """Energy of channel ch in keV, or None if no calibration."""
    cal = header.get("calibration")
    if not cal:
        return None
    return sum(a * ch**i for i, a in enumerate(cal))
```

---

## Device HTTP API

Segments are stored on the device's Flash and accessible over HTTP.

### List Segments

```
GET /api/waterfall/segments
```

Response `200 application/json`:

```json
{
  "segments": [
    { "name": "seg_00000.aswf", "size": 1056870, "finalized": true  },
    { "name": "seg_00001.aswf", "size": 524904,  "finalized": false }
  ],
  "ring_capacity": 64,
  "seg_count": 2
}
```

| Field       | Description |
|-------------|-------------|
| `name`      | Filename, used in the download request |
| `size`      | File size in bytes |
| `finalized` | `false` — segment still open (being written); header has `saved_rows=0` |

An unfinalised segment (`finalized: false`) can still be read; compute row count
from file size, not `saved_rows`.

### Download a Segment

```
GET /api/waterfall/segment?name=seg_00000.aswf
Authorization: Basic <base64(login:password)>
```

Response `200 application/octet-stream` — binary `.aswf` file.

### Delete a Segment (confirm receipt)

```
POST /api/waterfall/segment/delete?name=seg_00000.aswf
Authorization: Basic <base64(login:password)>
```

Response `200` — file deleted from Flash.

---

## Segment Ring

The device keeps a bounded number of finalised segments. When the ring is full,
the oldest segment is deleted automatically. The currently open segment
(`finalized: false`) does not count toward the limit.

Each segment holds up to `64` rows (~1 MB payload) and covers at most
10 minutes of recording regardless of interval.

---

## Merging Segments

To reconstruct a continuous file for a long recording period:

1. Fetch all segments (`GET /api/waterfall/segment?name=…`).
2. Sort by `started_at` from each header.
3. Concatenate payloads in chronological order.
4. Take metadata (`calibration`, `serial`, `interval_sec`) from the first segment.
5. Recompute `saved_rows` = sum of row counts from all segments.

Example:

```python
import json, struct
from pathlib import Path

def merge_aswf(paths_sorted, out_path):
    HDR_RESERVE = 4096
    files  = [Path(p).read_bytes() for p in paths_sorted]
    first  = files[0]
    hlen   = struct.unpack_from("<I", first, 4)[0]
    hdr    = json.loads(first[8:8 + hlen].decode("utf-8"))
    stride = hdr.get("row_stride", hdr["channels"] * 2)

    payload = b""
    for buf in files:
        h2 = struct.unpack_from("<I", buf, 4)[0]
        payload += buf[8 + h2:]

    total_rows    = len(payload) // stride
    hdr["saved_rows"] = total_rows
    hdr["saved_at"]   = 0  # unknown for manual merge

    hdr_bytes = json.dumps(hdr, ensure_ascii=False).encode("utf-8")
    hdr_bytes = hdr_bytes.ljust(HDR_RESERVE)

    with open(out_path, "wb") as f:
        f.write(b"ASWF")
        f.write(struct.pack("<I", HDR_RESERVE))
        f.write(hdr_bytes)
        f.write(payload)
```

---

## Edge Cases

| Situation | Behaviour |
|-----------|-----------|
| `saved_rows == 0` | Segment unfinalised. Compute row count from file size. |
| `saved_at == 0`   | Finalisation time unknown (open segment or manual merge). |
| `duration == 0`   | Actual live time unknown. Substitute `interval_sec`. |
| Unknown JSON keys | Ignore (future versions may add fields). |
| `hlen` ≠ 4096     | Reserved for future use. Always use the actual `hlen` from the file. |
| Truncated last row | `len(payload) % stride != 0` → discard the incomplete tail. |

---

## v3 — Self-Describing Rows (Extended Format)

v3 replaces the fixed row layout with a fully self-describing `row_fields` array in the header.
Each row may carry arbitrary typed fields: spectrum, duration, per-row timestamp, GPS coordinates,
dose rate, and any future extension — without breaking older parsers that ignore unknown fields.

### File Layout (v3)

```
ASWF(4) + hlen(4) + JSON(hlen) + baseline_section(B) + payload(N rows)
```

`B = baseline.count × 4` (zero when no `baseline` key in header).  
`payload_offset = 8 + hlen + B`.

### v3 Header Fields

| Key            | Type             | v3   | Description |
|----------------|------------------|:----:|-------------|
| `version`      | int              | yes  | `3` |
| `row_fields`   | array of objects | yes  | Field descriptors — see below |
| `row_stride`   | int              | no*  | Row size in bytes; **absent** when `compressed=true` |
| `compressed`   | bool             | no   | `true` → RLE-encoded rows (default `false`) |
| `baseline`     | object           | no   | `{"count": N}` — N uint32 LE values follow JSON header |

(*) Required for uncompressed v3; absent for compressed.

#### `row_fields` descriptor

```json
{
  "name":   "spectrum",
  "dtype":  "uint16",
  "offset": 0,
  "count":  8192
}
```

| Field    | Description |
|----------|-------------|
| `name`   | `"spectrum"`, `"duration"`, `"timestamp"`, `"latitude"`, `"longitude"`, `"dose_rate_usv_h"`, or any future name |
| `dtype`  | `"uint16"`, `"uint32"`, `"float32"` |
| `offset` | Byte offset of the field **within each row** (uncompressed); in compressed mode fields follow the RLE spectrum sequentially in offset-sort order |
| `count`  | Number of elements; present for `spectrum` (= `channels`); scalar fields omit or set `1` |

### Baseline Section

If the header contains `"baseline": {"count": N}`, a binary block of `N × 4` uint32 LE values
is inserted between the JSON area and the payload. This represents the cumulative spectrum
accumulated before the current session started (used for background subtraction).

```
payload_offset = 8 + hlen + (baseline.count × 4)
```

### Uncompressed v3

`row_stride` is required. Each row is `row_stride` bytes; fields are at their `offset` positions.
The spectrum occupies bytes `[0 .. channels×2 - 1]`.

### RLE-Compressed v3 (`compressed: true`)

`row_stride` is absent. Each row in the payload is encoded as:
1. **RLE spectrum** (variable length) — see encoding below.
2. **Non-spectrum fields** in ascending `offset` order, each at its natural dtype width.

#### RLE Encoding

Each element is a uint16 LE word:

| Word value         | Meaning |
|--------------------|---------|
| `v < 0x8000`       | Literal channel value `v` |
| `0x8000 ≤ v < 0xFFFF` | `v & 0x7FFF` consecutive zero channels |
| `0xFFFF`           | Reserved — parser must raise an error |

### Supported `name` Values (v3)

| Name               | dtype     | Description |
|--------------------|-----------|-------------|
| `spectrum`         | uint16    | Per-interval counts, `channels` elements |
| `duration`         | uint16    | Actual live time, seconds (0 → use `interval_sec`) |
| `timestamp`        | uint32    | Unix timestamp of this row (UTC, epoch seconds); overrides cumsum when > 0 |
| `latitude`         | float32   | GPS latitude (degrees); NaN if no fix |
| `longitude`        | float32   | GPS longitude (degrees); NaN if no fix |
| `dose_rate_usv_h`  | float32   | Dose rate, µSv/h; NaN if unavailable |

### Example v3 Header (uncompressed)

```json
{
  "format":       "atomspectra-waterfall",
  "version":      3,
  "channels":     8192,
  "dtype":        "uint16",
  "byte_order":   "little",
  "interval_sec": 60,
  "started_at":   1783157403,
  "saved_rows":   128,
  "saved_at":     1783165103,
  "serial":       "AS-003",
  "calibration":  [0.0, 0.298, 0.0],
  "row_stride":   16406,
  "baseline":     {"count": 8192},
  "row_fields": [
    {"name": "spectrum",       "dtype": "uint16",  "offset": 0,     "count": 8192},
    {"name": "duration",       "dtype": "uint16",  "offset": 16384},
    {"name": "dose_rate_usv_h","dtype": "float32", "offset": 16386},
    {"name": "latitude",       "dtype": "float32", "offset": 16390},
    {"name": "longitude",      "dtype": "float32", "offset": 16394},
    {"name": "timestamp",      "dtype": "uint32",  "offset": 16398},
    {"name": "dose_rate_msvh", "dtype": "float32", "offset": 16402}
  ]
}
```

---

## Format Version History

| Version | Changes |
|---------|---------|
| v1      | 16384 bytes per row. No duration field. `row_stride` absent from header. |
| v2      | +2-byte uint16 LE duration appended to each row. `row_stride=16386`, `row_time` added to header. |
| v3      | Self-describing `row_fields`; optional `baseline` section; optional RLE compression; per-row `timestamp`, GPS, and `dose_rate_usv_h`. |
