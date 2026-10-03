import unittest

import numpy as np

from track360.backends import ARTrackBackend, ARTrackPrediction, TrackerBackendImpl
from track360.backends.artrack_model import ARTrackTemplate
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    LocalView,
    SphericalPoint,
    TemplateCommand,
    TemplateCommandKind,
    ViewSpec,
)


class _FakeARTrackSession:
    supportsOnlineTemplates = True

    def __init__(self):
        self.inferTemplateCounts = []
        self.deviceTemplateCalls = []

    def encodeTemplate(self, rgb, bbox):
        return ARTrackTemplate(rgb, bbox)

    def encodeTemplateDevice(self, deviceRgb, bbox, imageSize):
        self.deviceTemplateCalls.append((deviceRgb, bbox, imageSize))
        return ARTrackTemplate(deviceRgb, bbox)

    def infer(self, rgb, templateFeatures):
        self.inferTemplateCounts.append(len(templateFeatures))
        return ARTrackPrediction(BBoxXYWH(8.0, 9.0, 20.0, 18.0), 0.8, 0.8, 0.8)

    def close(self):
        return None


class ARTrackBackendTest(unittest.TestCase):
    def test_gpu_view_template_uses_device_pixels_instead_of_rgb_placeholder(self):
        class _DeviceRgb:
            shape = (3, 64, 64)
            ndim = 3

        point = SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0)
        spec = ViewSpec(0, BFoV(point, 1.0, 1.0), 64, 64)
        deviceRgb = _DeviceRgb()
        view = LocalView(
            spec,
            np.zeros((64, 64, 3), dtype=np.uint8),
            deviceRgb=deviceRgb,
        )
        box = BBoxXYWH(20.0, 20.0, 16.0, 16.0)
        session = _FakeARTrackSession()

        encoded = ARTrackBackend(session).encodeTemplateView(view, box)

        self.assertIs(encoded.tensor, deviceRgb)
        self.assertEqual(session.deviceTemplateCalls, [(deviceRgb, box, (64, 64))])

    def test_local_prediction_preserves_tracker_contract(self):
        point = SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0)
        spec = ViewSpec(0, BFoV(point, 1.0, 1.0), 64, 64)
        view = LocalView(spec, np.zeros((64, 64, 3), dtype=np.uint8))
        backend = TrackerBackendImpl(ARTrackBackend(_FakeARTrackSession()))
        backend.initialize(view, BBoxXYWH(20.0, 20.0, 16.0, 16.0))
        observations = backend.infer(
            (view,), TemplateCommand(TemplateCommandKind.KEEP, 1, None, None, 1)
        )
        self.assertEqual(observations[0].bbox.widthPx, 20.0)
        self.assertEqual(observations[0].predictedIoU, 0.8)
        backend.close()

    def test_online_template_updates_reach_artrack_session(self):
        point = SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0)
        spec = ViewSpec(0, BFoV(point, 1.0, 1.0), 64, 64)
        view = LocalView(spec, np.zeros((64, 64, 3), dtype=np.uint8))
        session = _FakeARTrackSession()
        backend = TrackerBackendImpl(ARTrackBackend(session))
        backend.initialize(view, BBoxXYWH(20.0, 20.0, 16.0, 16.0))
        backend.infer((view,), TemplateCommand(TemplateCommandKind.KEEP, 1, None, None, 1))
        backend.infer(
            (view,),
            TemplateCommand(
                TemplateCommandKind.UPDATE_RECENT,
                2,
                0,
                BBoxXYWH(21.0, 21.0, 15.0, 15.0),
                2,
            ),
        )
        self.assertEqual(session.inferTemplateCounts, [1, 2])
        backend.close()


if __name__ == "__main__":
    unittest.main()
