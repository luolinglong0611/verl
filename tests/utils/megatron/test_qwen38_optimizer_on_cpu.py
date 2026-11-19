# Copyright 2026 the contributors to this fork
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from verl.experimental.qwen38_rl.optimizer.bf16_math import key
from verl.experimental.qwen38_rl.optimizer.coordinates import SRCoordinates, stochastic_round_coordinates


def test_strided_tensor_parallel_rounding_matches_the_logical_tensor():
    values = torch.linspace(-1.0, 1.0, 48).reshape(2, 12, 2)
    full = SRCoordinates("expert.weight", values.shape, values.shape)
    seed = key(full.canonical_name, 2, "weight", 126)
    expected = stochastic_round_coordinates(values, seed, full, 0).flatten()
    gathered = torch.empty_like(expected)
    for rank in range(3):
        local = SRCoordinates(
            "expert.weight", (2, 12, 2), (2, 4, 2), axis=1, partition_count=3, partition_rank=rank, partition_stride=2
        )
        indices = local.indices(0, 16)
        source = values.flatten()[indices]
        # Tile boundaries differ from the logical parameter's shape.
        gathered[indices[:5]] = stochastic_round_coordinates(source[:5], seed, local, 0)
        gathered[indices[5:]] = stochastic_round_coordinates(source[5:], seed, local, 5)

    assert torch.equal(gathered.view(torch.int16), expected.view(torch.int16))
