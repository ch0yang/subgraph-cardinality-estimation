from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from candidate_tree_estimator.training_utils import outer_fold_indices


class OuterFoldProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "fold": np.repeat(np.arange(5), 4),
            }
        )

    def test_each_run_uses_four_training_folds(self) -> None:
        for outer_fold in range(5):
            train_index, test_index = outer_fold_indices(
                self.frame,
                outer_fold,
            )
            self.assertEqual(len(train_index), 16)
            self.assertEqual(len(test_index), 4)
            self.assertTrue(
                np.all(
                    self.frame.iloc[test_index]["fold"].to_numpy()
                    == outer_fold
                )
            )
            self.assertTrue(
                np.all(
                    self.frame.iloc[train_index]["fold"].to_numpy()
                    != outer_fold
                )
            )
            self.assertEqual(
                np.intersect1d(train_index, test_index).size,
                0,
            )

    def test_test_folds_cover_every_row_once(self) -> None:
        test_indices = [
            outer_fold_indices(self.frame, outer_fold)[1]
            for outer_fold in range(5)
        ]
        combined = np.concatenate(test_indices)
        np.testing.assert_array_equal(
            np.sort(combined),
            np.arange(len(self.frame)),
        )


if __name__ == "__main__":
    unittest.main()
