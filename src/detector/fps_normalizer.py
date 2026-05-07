"""
Resampling strumienia klatek do TARGET_FPS niezależnie od FPS źródła.

Drop-in między source.get_frame() a Detector.process_batch(). Trzyma
last_emit_t i decyduje per klatka czy emit/drop. Mapowanie nearest-neighbor
identyczne z research-notebooks/utils/fps_resampler.iterate_at_target_fps,
tak żeby trening i runtime widziały tę samą sekwencję klatek dla danej
szybkości źródła.
"""

from __future__ import annotations


class FrameRateNormalizer:
    """
    Decyduje czy klatka źródła ma być wyemitowana do downstream pipeline.

    Użycie:
        norm = FrameRateNormalizer(source_fps=30.0, target_fps=25.0)
        for src_frame in source_frames:
            if norm.should_emit():
                yield src_frame
            norm.advance()

    Mapowanie: target_idx → src_idx_required = round(target_idx * source_fps / target_fps).
    Klatka źródła jest emitowana jeśli (lub kilka razy jeśli upsample).
    """

    def __init__(self, source_fps: float, target_fps: float):
        if source_fps <= 0 or target_fps <= 0:
            raise ValueError(f"FPS musi być dodatni (source={source_fps}, target={target_fps})")
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.ratio = source_fps / target_fps
        self._src_idx = 0
        self._target_idx = 0

    def emit_count(self) -> int:
        """Ile razy bieżąca klatka źródła powinna być wyemitowana (0+ razy)."""
        count = 0
        while int(round(self._target_idx * self.ratio)) <= self._src_idx:
            required = int(round(self._target_idx * self.ratio))
            if required == self._src_idx:
                count += 1
                self._target_idx += 1
            elif required < self._src_idx:
                self._target_idx += 1
            else:
                break
        return count

    def advance(self) -> None:
        self._src_idx += 1

    @property
    def target_idx(self) -> int:
        return self._target_idx
