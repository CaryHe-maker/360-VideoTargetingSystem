import unittest

import numpy as np

from track360.backends import ARTrackBackend, ARTrackPrediction, TrackerBackendImpl
from track360.backends.artrack_model import ARTrackTemplate
from track360.core.types import BBoxXYWH, BFoV, LocalView, SphericalPoint, ViewSpec


class _FakeARTrackSession:
    def __init__(self):
        self.inferTemplateCounts = []

    def encodeTemplate(self, rgb, bbox):
        return ARTrackTemplate(rgb, bbox)

    def infer(self, rgb, templateFeatures):
        self.inferTemplateCounts.append(len(templateFeatures))
        return ARTrackPrediction(BBoxXYWH(8.0, 9.0, 20.0, 18.0), 0.8, 0.8, 0.8)

    def close(self):
        return None


class ARTrackBackendTest(unittest.TestCase):
    def test_template_view_keeps_the_field_of_view_it_was_cut_from(self):
        point = SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0)
        spec = ViewSpec(0, BFoV(point, 1.0, 0.8), 64, 64)
        view = LocalView(spec, np.zeros((64, 64, 3), dtype=np.uint8))
        box = BBoxXYWH(20.0, 20.0, 16.0, 16.0)

        encoded = ARTrackBackend(_FakeARTrackSession()).encodeTemplateView(view, box)

        self.assertIs(encoded.tensor, view.rgb)
        self.assertEqual(encoded.sourceHorizontalFovRad, 1.0)
        self.assertEqual(encoded.sourceVerticalFovRad, 0.8)

    def test_local_prediction_preserves_tracker_contract(self):
        point = SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0)
        spec = ViewSpec(0, BFoV(point, 1.0, 1.0), 64, 64)
        view = LocalView(spec, np.zeros((64, 64, 3), dtype=np.uint8))
        session = _FakeARTrackSession()
        backend = TrackerBackendImpl(ARTrackBackend(session))
        backend.initialize(view, BBoxXYWH(20.0, 20.0, 16.0, 16.0))
        observations = backend.infer((view,))
        self.assertEqual(observations[0].bbox.widthPx, 20.0)
        self.assertEqual(observations[0].predictedIoU, 0.8)
        # Every pass sees the frame-0 template and nothing else.
        backend.infer((view,))
        self.assertEqual(session.inferTemplateCounts, [1, 1])
        backend.close()


if __name__ == "__main__":
    unittest.main()
