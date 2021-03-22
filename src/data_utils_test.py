import data_utils
import pandas as pd
import torch
import unittest

from transformers import RobertaTokenizerFast


class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4

    def test_sample(self):
        ## An example of a basic unit test, using class variables initialized in
        ## setUpClass().
        self.assertEqual(self.max_seq_len, 4)

    def test_encode_data(self):
        ## TODO: Write a unit test that asserts that the dimensions and dtype of the
        ## output of encode_data() are correct.
        ## input_ids should have shape [len(self.dataset), self.max_seq_len] and type torch.long.
        ## attention_mask should have the same shape and type.
        valid_shape = (len(self.dataset), self.max_seq_len)
        input_ids, attention_mask = data_utils.encode_data(self.dataset, self.tokenizer, max_seq_length=self.max_seq_len)
        self.assertEqual(input_ids.shape, valid_shape)
        self.assertIsInstance(input_ids.dtype, type(torch.long))
        self.assertEqual(attention_mask.shape, valid_shape)
        self.assertIsInstance(attention_mask.dtype, type(torch.long))

    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
        self.assertEqual(data_utils.extract_labels(self.dataset), [1,0])

if __name__ == "__main__":
    unittest.main()
