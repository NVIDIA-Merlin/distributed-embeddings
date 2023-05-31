# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test of embedding layers"""

import tensorflow as tf
import numpy as np

from absl.testing import parameterized

from keras.layers.preprocessing import integer_lookup
from keras.testing_infra import test_combinations
from distributed_embeddings.python.layers.embedding import IntegerLookup


# pylint:disable=missing-docstring, no-self-use
@test_combinations.run_all_keras_modes(always_skip_v1=True)
class IntegerLookupLayerTest(test_combinations.TestCase):

  # TODO: this test pass not but in theory it depends on atomic
  @parameterized.named_parameters(
      ("gpu", True),
      ("cpu", False),
  )
  def test_layer_with_list_input(self, use_gpu):
    vocab = [12, 36, 1138, 42]
    data = [[12, 1138, 42], [42, 1000, 36]]  # Note OOV tokens
    layer = integer_lookup.IntegerLookup(vocabulary=vocab)
    output = layer(data)
    expected_output = np.array([[1, 3, 4], [4, 0, 2]])

    test_layer = IntegerLookup(max_tokens=4, use_gpu=use_gpu)
    test_layer(tf.convert_to_tensor(vocab, tf.int64))  # Init with vocab
    test_output = test_layer(tf.convert_to_tensor(data, tf.int64))

    self.assertEqual(output.numpy().tolist(), expected_output.tolist())
    self.assertEqual(test_output.numpy().tolist(), expected_output.tolist())

  @parameterized.named_parameters(
      ("gpu", True),
      ("cpu", False),
  )
  def test_layer_against_native(self, use_gpu):
    for key_max in [100, 200, 500, 1000]:
      for vocab_size in [100, 200, 500, 1000]:

        vocab = tf.random.uniform(shape=(vocab_size,), maxval=key_max, dtype=tf.int64)
        unique_vocab = tf.size(tf.unique(vocab)[0])
        # make sure test table is full so we can compare against reference without inserting new
        # TODO: test get_vocabulary()
        test_layer = IntegerLookup(max_tokens=unique_vocab, use_gpu=use_gpu)
        test_layer(vocab)  # Init with vocab
        ref_layer = integer_lookup.IntegerLookup(vocabulary=test_layer.get_vocabulary()[1:])

        data = tf.range(0, 1024, dtype=tf.int64)
        ref_output = ref_layer(data)
        test_output = test_layer(data)

        self.assertAllEqual(ref_output, test_output)


if __name__ == "__main__":
  tf.test.main()
