from __future__ import annotations

__all__ = [
    "SHPROTO_START",
    "SHPROTO_ESC",
    "SHPROTO_FINISH",
    "CRC_INIT",
    "ShprotoDecoder",
    "ShprotoEncoder",
]

SHPROTO_START = 0xFE
SHPROTO_ESC = 0xFD
SHPROTO_FINISH = 0xA5
CRC_INIT = 0xFFFF

_CRC16_TABLE = (
    0x0000, 0xC0C1, 0xC181, 0x0140, 0xC301, 0x03C0, 0x0280, 0xC241,
    0xC601, 0x06C0, 0x0780, 0xC741, 0x0500, 0xC5C1, 0xC481, 0x0440,
    0xCC01, 0x0CC0, 0x0D80, 0xCD41, 0x0F00, 0xCFC1, 0xCE81, 0x0E40,
    0x0A00, 0xCAC1, 0xCB81, 0x0B40, 0xC901, 0x09C0, 0x0880, 0xC841,
    0xD801, 0x18C0, 0x1980, 0xD941, 0x1B00, 0xDBC1, 0xDA81, 0x1A40,
    0x1E00, 0xDEC1, 0xDF81, 0x1F40, 0xDD01, 0x1DC0, 0x1C80, 0xDC41,
    0x1400, 0xD4C1, 0xD581, 0x1540, 0xD701, 0x17C0, 0x1680, 0xD641,
    0xD201, 0x12C0, 0x1380, 0xD341, 0x1100, 0xD1C1, 0xD081, 0x1040,
    0xF001, 0x30C0, 0x3180, 0xF141, 0x3300, 0xF3C1, 0xF281, 0x3240,
    0x3600, 0xF6C1, 0xF781, 0x3740, 0xF501, 0x35C0, 0x3480, 0xF441,
    0x3C00, 0xFCC1, 0xFD81, 0x3D40, 0xFF01, 0x3FC0, 0x3E80, 0xFE41,
    0xFA01, 0x3AC0, 0x3B80, 0xFB41, 0x3900, 0xF9C1, 0xF881, 0x3840,
    0x2800, 0xE8C1, 0xE981, 0x2940, 0xEB01, 0x2BC0, 0x2A80, 0xEA41,
    0xEE01, 0x2EC0, 0x2F80, 0xEF41, 0x2D00, 0xEDC1, 0xEC81, 0x2C40,
    0xE401, 0x24C0, 0x2580, 0xE541, 0x2700, 0xE7C1, 0xE681, 0x2640,
    0x2200, 0xE2C1, 0xE381, 0x2340, 0xE101, 0x21C0, 0x2080, 0xE041,
    0xA001, 0x60C0, 0x6180, 0xA141, 0x6300, 0xA3C1, 0xA281, 0x6240,
    0x6600, 0xA6C1, 0xA781, 0x6740, 0xA501, 0x65C0, 0x6480, 0xA441,
    0x6C00, 0xACC1, 0xAD81, 0x6D40, 0xAF01, 0x6FC0, 0x6E80, 0xAE41,
    0xAA01, 0x6AC0, 0x6B80, 0xAB41, 0x6900, 0xA9C1, 0xA881, 0x6840,
    0x7800, 0xB8C1, 0xB981, 0x7940, 0xBB01, 0x7BC0, 0x7A80, 0xBA41,
    0xBE01, 0x7EC0, 0x7F80, 0xBF41, 0x7D00, 0xBDC1, 0xBC81, 0x7C40,
    0xB401, 0x74C0, 0x7580, 0xB541, 0x7700, 0xB7C1, 0xB681, 0x7640,
    0x7200, 0xB2C1, 0xB381, 0x7340, 0xB101, 0x71C0, 0x7080, 0xB041,
    0x5000, 0x90C1, 0x9181, 0x5140, 0x9301, 0x53C0, 0x5280, 0x9241,
    0x9601, 0x56C0, 0x5780, 0x9741, 0x5500, 0x95C1, 0x9481, 0x5440,
    0x9C01, 0x5CC0, 0x5D80, 0x9D41, 0x5F00, 0x9FC1, 0x9E81, 0x5E40,
    0x5A00, 0x9AC1, 0x9B81, 0x5B40, 0x9901, 0x59C0, 0x5880, 0x9841,
    0x8801, 0x48C0, 0x4980, 0x8941, 0x4B00, 0x8BC1, 0x8A81, 0x4A40,
    0x4E00, 0x8EC1, 0x8F81, 0x4F40, 0x8D01, 0x4DC0, 0x4C80, 0x8C41,
    0x4400, 0x84C1, 0x8581, 0x4540, 0x8701, 0x47C0, 0x4680, 0x8641,
    0x8201, 0x42C0, 0x4380, 0x8341, 0x4100, 0x81C1, 0x8081, 0x4040,
)

def _crc16_byte(crc: int, byte: int) -> int:
    """CRC-16 update одним байтом (табличный вариант)."""
    return (crc >> 8) ^ _CRC16_TABLE[(crc ^ byte) & 0xFF]


def _crc16_buf(buf: bytes) -> int:
    """CRC-16 буфера с init=CRC_INIT."""
    crc = CRC_INIT
    for b in buf:
        crc = _crc16_byte(crc, b)
    return crc


class ShprotoDecoder:
    """Декодер shproto-кадров из потока байт."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._started = False
        self._esc = False
        self._frames: list[tuple[int, bytes]] = []
        self.dropped = 0

    def reset(self) -> None:
        """Сброс состояния и очереди кадров."""
        self._buf.clear()
        self._started = False
        self._esc = False
        self._frames.clear()
        self.dropped = 0

    def byte_received(self, byte: int) -> bool:
        """Принять один байт. True — кадр готов (в очереди), False — нет."""
        if byte == SHPROTO_START:
            self._buf.clear()
            self._started = True
            self._esc = False
            return False

        if not self._started:
            return False

        if byte == SHPROTO_ESC:
            self._esc = True
            return False

        if byte == SHPROTO_FINISH:
            if len(self._buf) >= 3 and _crc16_buf(bytes(self._buf)) == 0:
                cmd = self._buf[0]
                payload = bytes(self._buf[1:-2])
                self._frames.append((cmd, payload))
            else:
                self.dropped += 1
            self._started = False
            return len(self._frames) > 0

        # Обычный байт
        if self._esc:
            byte = (~byte) & 0xFF
            self._esc = False
        self._buf.append(byte)
        return False

    def feed(self, data: bytes) -> int:
        """Принять пачку байтов. Возвращает число новых готовых кадров."""
        old_len = len(self._frames)
        for b in data:
            self.byte_received(b)
        return len(self._frames) - old_len

    def take_frame(self) -> tuple[int, bytes] | None:
        """Извлечь один готовый кадр из очереди (FIFO). None если пусто."""
        if not self._frames:
            return None
        return self._frames.pop(0)

    def frames(self) -> list[tuple[int, bytes]]:
        """Извлечь ВСЕ готовые кадры и очистить очередь."""
        result = self._frames[:]
        self._frames.clear()
        return result


class ShprotoEncoder:
    """Энкодер shproto-кадров."""

    @staticmethod
    def encode(cmd: int, payload: bytes = b"") -> bytes:
        """Собрать полный кадр: 0xFF + START + escaped(cmd|payload|crc_le) + FINISH."""
        raw = bytes([cmd]) + payload
        crc = _crc16_buf(raw)
        body_bytes = raw + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
        escaped = _escape_stream(body_bytes)
        return bytes([0xFF, SHPROTO_START]) + escaped + bytes([SHPROTO_FINISH])

    def __init__(self) -> None:
        self._out = bytearray()
        self._crc = CRC_INIT

    def packet_start(self, cmd: int) -> None:
        """Начать новый пакет."""
        self._out.clear()
        self._crc = CRC_INIT
        self._out.append(0xFF)
        self._out.append(SHPROTO_START)
        self._crc = _crc16_byte(self._crc, cmd)
        self._append_escaped(cmd)

    def packet_add_data(self, byte: int) -> None:
        """Добавить байт данных в пакет."""
        self._crc = _crc16_byte(self._crc, byte)
        self._append_escaped(byte)

    def packet_complete(self) -> bytes:
        """Завершить пакет и вернуть полный байтовый буфер."""
        # Добавляем CRC
        crc_lo = self._crc & 0xFF
        crc_hi = (self._crc >> 8) & 0xFF
        self._append_escaped(crc_lo)
        self._append_escaped(crc_hi)
        # Добавляем FINISH
        self._out.append(SHPROTO_FINISH)
        return bytes(self._out)

    def _append_escaped(self, byte: int) -> None:
        if byte in (SHPROTO_START, SHPROTO_FINISH, SHPROTO_ESC):
            self._out.append(SHPROTO_ESC)
            byte = (~byte) & 0xFF
        self._out.append(byte)


def _escape_stream(data: bytes) -> bytes:
    """Экранировать байты в потоке."""
    result = bytearray()
    for b in data:
        if b in (SHPROTO_START, SHPROTO_FINISH, SHPROTO_ESC):
            result.append(SHPROTO_ESC)
            b = (~b) & 0xFF
        result.append(b)
    return bytes(result)
