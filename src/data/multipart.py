"""Seekable read-only view over a file split into parts.

iSign ships its pose archive as `iSign-poses_v1.1_part_a{a..d}` -- a plain
`split` of one 159 GiB zip. Zip needs to seek to the central directory at the
end of the file, so it cannot be streamed; the obvious fix is
`cat part_* > full.zip`, which costs another 159 GiB of scratch.

This presents the parts as one seekable file object instead, so
`zipfile.ZipFile(MultiPartFile(parts))` works with no duplication.
"""

from __future__ import annotations

import bisect
import io
from pathlib import Path


class MultiPartFile(io.RawIOBase):
    def __init__(self, parts: list[Path]):
        if not parts:
            raise ValueError("no parts given")
        self.parts = [Path(p) for p in parts]
        self.sizes = [p.stat().st_size for p in self.parts]
        # starts[i] = byte offset at which part i begins
        self.starts, off = [], 0
        for s in self.sizes:
            self.starts.append(off)
            off += s
        self.size = off
        self._pos = 0
        self._fh: dict[int, io.BufferedReader] = {}

    # --- io plumbing ----------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, min(self.size, base + pos))
        return self._pos

    def _handle(self, i: int) -> io.BufferedReader:
        if i not in self._fh:
            self._fh[i] = open(self.parts[i], "rb")
        return self._fh[i]

    def readinto(self, b) -> int:
        n = len(b)
        if n == 0 or self._pos >= self.size:
            return 0
        got = 0
        while got < n and self._pos < self.size:
            i = bisect.bisect_right(self.starts, self._pos) - 1
            fh = self._handle(i)
            fh.seek(self._pos - self.starts[i])
            # never read past the end of this part -- the next loop picks it up
            want = min(n - got, self.starts[i] + self.sizes[i] - self._pos)
            chunk = fh.read(want)
            if not chunk:
                break
            b[got:got + len(chunk)] = chunk
            got += len(chunk)
            self._pos += len(chunk)
        return got

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()
        super().close()


def open_split_zip(pattern_dir: Path, stem: str):
    """ZipFile over `{stem}_part_*` in `pattern_dir`, sorted by suffix."""
    import zipfile

    parts = sorted(Path(pattern_dir).glob(f"{stem}_part_*"))
    if not parts:
        raise FileNotFoundError(f"no parts matching {stem}_part_* in {pattern_dir}")
    return zipfile.ZipFile(MultiPartFile(parts))
