import unittest

from track360.core.types import BBoxXYWH
from track360.evaluation.otb_metrics import OtbMetrics, bboxIoU, trackingLossRate


class PlanarMetricsTest(unittest.TestCase):
    def testOtbMetricsSummarizeSuccessRate(self) -> None:
        metrics = OtbMetrics()
        metrics.update(BBoxXYWH(0.0, 0.0, 10.0, 10.0), BBoxXYWH(0.0, 0.0, 10.0, 10.0))
        metrics.update(BBoxXYWH(0.0, 0.0, 5.0, 5.0), BBoxXYWH(0.0, 0.0, 10.0, 10.0))

        summary = metrics.summarize()

        self.assertAlmostEqual(summary["successRate@0.5"], 0.5)
        self.assertEqual(summary["lostFrameCount"], 0)
        self.assertEqual(summary["trackingLossRate"], 0.0)
        self.assertAlmostEqual(
            bboxIoU(
                BBoxXYWH(0.0, 0.0, 10.0, 10.0),
                BBoxXYWH(0.0, 0.0, 10.0, 10.0),
            ),
            1.0,
        )

    def testTrackingLossRateCountsOnlyZeroOverlapVisibleSamples(self) -> None:
        self.assertAlmostEqual(trackingLossRate([0.0, 0.2, 0.0, 1.0]), 0.5)
        self.assertEqual(trackingLossRate([]), 0.0)

        metrics = OtbMetrics(ious=[0.0, 0.2, 0.0, 1.0])
        summary = metrics.summarize()
        self.assertEqual(summary["lostFrameCount"], 2)
        self.assertAlmostEqual(summary["trackingLossRate"], 0.5)



if __name__ == "__main__":
    unittest.main()
