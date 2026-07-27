import unittest

from src.core.experts.knn_expert import KNNExpert


class KNNLabelConsistencyTests(unittest.TestCase):
    def test_explicit_ng_label_overrides_incorrect_ok_folder(self):
        data = {"label": "NG", "analysis": {}}
        self.assertEqual(KNNExpert._resolve_record_label(data, "OK"), "NG")

    def test_explicit_ok_label_overrides_incorrect_ng_folder(self):
        data = {"label": "OK", "analysis": {}}
        self.assertEqual(KNNExpert._resolve_record_label(data, "NG"), "OK")

    def test_folder_is_used_only_when_json_has_no_label(self):
        self.assertEqual(KNNExpert._resolve_record_label({}, "NG"), "NG")
        self.assertEqual(KNNExpert._resolve_record_label({}, "OK"), "OK")

    def test_single_ng_neighbor_votes_one_hundred_percent_ng(self):
        neighbors = [(0.32, "NG", "sample_ng.json")]
        self.assertEqual(KNNExpert._weighted_vote(neighbors), 1.0)

    def test_single_ok_neighbor_votes_zero_percent_ng(self):
        neighbors = [(0.32, "OK", "sample_ok.json")]
        self.assertEqual(KNNExpert._weighted_vote(neighbors), 0.0)

    def test_similarity_does_not_change_the_neighbors_class(self):
        low_similarity_ng = [(0.80, "NG", "sample_ng.json")]
        high_similarity_ng = [(0.05, "NG", "sample_ng.json")]
        self.assertEqual(KNNExpert._weighted_vote(low_similarity_ng), 1.0)
        self.assertEqual(KNNExpert._weighted_vote(high_similarity_ng), 1.0)


if __name__ == "__main__":
    unittest.main()
