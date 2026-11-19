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
"""Exercise the finite relay with real concurrent HTTP range requests."""

import json
import socket
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

from verl.experimental.qwen38_rl.checkpoint_relay import REVISION, Range, fetch
from verl.experimental.qwen38_rl.training_relay import serve_training_plan


class RelayTests(unittest.TestCase):
    @patch("verl.experimental.qwen38_rl.training_relay.validate_source")
    @patch("verl.experimental.qwen38_rl.training_relay._private_ip", side_effect=str)
    @patch("verl.experimental.qwen38_rl.checkpoint_relay._private_ip", side_effect=str)
    def test_exact_concurrent_requests(self, _client_ip, _server_ip, _mount):
        loopback = "127.0.0.1"
        name = "model-00001-of-00213.safetensors"
        payload = bytes(range(128))
        ranges = [Range(name, len(payload), offset, 16) for offset in (8, 32, 96)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / name).write_bytes(payload)
            with socket.socket() as reserved:
                reserved.bind((loopback, 0))
                port = reserved.getsockname()[1]

            def consume(selected):
                for _ in range(20):
                    try:
                        data, _ = fetch(loopback, port, selected, local_ip=loopback, timeout=2)
                        return data
                    except ConnectionRefusedError:
                        time.sleep(0.05)
                raise TimeoutError("local relay never bound its port")

            with ThreadPoolExecutor(max_workers=5) as pool:
                ready = root / "READY.json"
                server = pool.submit(
                    serve_training_plan,
                    root,
                    loopback,
                    port,
                    {loopback: [asdict(row) for row in ranges]},
                    None,
                    workers=2,
                    seconds=5,
                    ready_file=ready,
                    plan_sha256="probe",
                )
                unplanned = Range(name, len(payload), 64, 16)
                with self.assertRaisesRegex(ValueError, "HTTP 404"):
                    consume(unplanned)
                connection = HTTPConnection(loopback, port, timeout=2)
                try:
                    for _ in range(2):
                        connection.request("GET", f"/{REVISION}/ready")
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.loads(response.read()), json.loads(ready.read_bytes()))
                finally:
                    connection.close()
                futures = [pool.submit(consume, selected) for selected in ranges]
                for selected, future in zip(ranges, futures, strict=False):
                    self.assertEqual(future.result(), payload[selected.offset : selected.offset + selected.length])
                receipt = server.result()
            self.assertEqual(receipt["ranges"], 3)
            self.assertEqual(receipt["bytes"], 48)
            self.assertEqual(receipt["source_plan_sha256"], json.loads(ready.read_bytes())["source_plan_sha256"])
