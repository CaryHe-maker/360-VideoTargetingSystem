"""ARTrackV2 tracker backend and model adapters."""

from track360.backends.artrack_backend import TrackerBackend, TrackerBackendImpl
from track360.backends.artrack_model import (
    ARTrackBackend,
    ARTrackPrediction,
    ARTrackSession,
    ARTrackTemplate,
    PyTorchARTrackV2Session,
)
from track360.backends.observation import buildRgbObservation, clipLocalBox
from track360.backends.template_cache import TemplateCache, TemplateSample, TemplateSnapshot

__all__ = [
    "ARTrackBackend",
    "ARTrackPrediction",
    "ARTrackSession",
    "ARTrackTemplate",
    "PyTorchARTrackV2Session",
    "TemplateCache",
    "TemplateSample",
    "TemplateSnapshot",
    "TrackerBackend",
    "TrackerBackendImpl",
    "buildRgbObservation",
    "clipLocalBox",
]
