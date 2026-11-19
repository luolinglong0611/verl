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
"""NIXL full-tensor transport for RTP consumers with all-rank CUDA IPC acknowledgments."""

from __future__ import annotations

import asyncio
import hashlib

import torch

from verl.checkpoint_engine import nixl_checkpoint_engine as upstream
from verl.checkpoint_engine.base import CheckpointEngineRegistry

WIRE_PREFIX = "rtp-sha256-v1:"


def tensor_digest(tensor, *, chunk_bytes=8 * 1024 * 1024):
    """Hash GPU tensor bytes with a bounded host staging allocation.

    Each blocking device-to-host copy is complete before the host view is read.
    The caller must first wait for the CUDA stream that produced the tensor.
    """
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("a contiguous CUDA source is required")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("invalid digest staging limit")
    digest = hashlib.sha256()
    data = tensor.view(torch.uint8).view(-1)
    for offset in range(0, data.numel(), chunk_bytes):
        host = data.narrow(0, offset, min(chunk_bytes, data.numel() - offset)).cpu()
        digest.update(memoryview(host.numpy()).cast("B"))
    return digest.hexdigest()


def _wire_name(name, digest):
    if not isinstance(name, str) or not name or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("invalid source name or SHA256")
    return WIRE_PREFIX + digest + ":" + name


def _unwire_name(name):
    if not isinstance(name, str) or not name.startswith(WIRE_PREFIX):
        raise RuntimeError("NIXL source lacks trainer digest")
    rest = name[len(WIRE_PREFIX) :]
    digest, sep, original = rest.partition(":")
    if not sep or not original or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError("invalid NIXL source digest descriptor")
    return original, digest


class _NativeLeaseIterator:
    def __init__(self, engine, global_steps):
        self.engine = engine
        self.iterator = engine._guarded_receive(global_steps)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.iterator.__anext__()

    def confirm_native_ack(self, name, proof):
        self.engine.confirm_native_ack(name, proof)


@CheckpointEngineRegistry.register("rtp_nixl_guarded")
class RtpGuardedNixl(upstream.NIXLCheckpointEngine):
    """NIXL READ ring plus an explicit all-rank native ACK buffer lease."""

    def __init__(
        self,
        bucket_size,
        is_master=False,
        *,
        max_tensor_bytes=16 * 1024**3,
        timeout_seconds=600,
        transaction_timeout_seconds=43200,
        device="cuda",
    ):
        if (
            type(bucket_size) is not int
            or bucket_size <= 0
            or type(max_tensor_bytes) is not int
            or max_tensor_bytes <= 0
            or type(timeout_seconds) is not int
            or timeout_seconds <= 0
            or type(transaction_timeout_seconds) is not int
            or transaction_timeout_seconds <= 0
        ):
            raise ValueError("invalid NIXL buffer/time bounds")
        if device != "cuda":
            raise ValueError("RTP guarded NIXL requires CUDA buffers")
        super().__init__(bucket_size=bucket_size, is_master=is_master, device=device)
        self.max_tensor_bytes = max_tensor_bytes
        self.timeout_seconds = timeout_seconds
        self.transaction_timeout_seconds = transaction_timeout_seconds
        self.poisoned = False
        self.active = False
        self.transfer_started = False
        self.finished = False
        self.pending = None
        # A timeout may unwind the async generator while its shielded READ
        # remains in flight. Keep an incomplete oversized source reachable.
        self.partial_merge_value = None
        self.partial_chunks = []
        self.protocol_operations = []
        self.source_values = []
        self.pending_name = None
        self.pending_step = None
        self.native_acknowledged = False
        self.native_binding = None
        self.inflight_task = None

    def prepare(self):
        if (
            self.poisoned
            or self.active
            or self.pending is not None
            or self.partial_merge_value is not None
            or self.partial_chunks
            or self.protocol_operations
            or self.source_values
            or self.inflight_task is not None
        ):
            raise RuntimeError("previous NIXL generation not safely finalized")
        self.transfer_started = False
        self.finished = False
        self.native_binding = None
        try:
            result = super().prepare()
            self.active = True
            return result
        except BaseException:
            self.poisoned = True
            # Any partially registered upstream buffers remain process-owned;
            # external supervisor must stop peers before releasing this actor.
            raise

    def _begin_transfer(self, global_steps):
        if (
            not self.active
            or self.poisoned
            or self.transfer_started
            or type(global_steps) is not int
            or global_steps < 0
        ):
            raise RuntimeError("invalid or already started NIXL generation")
        self.transfer_started = True

    async def send_weights(self, weights, global_steps=None):
        self._begin_transfer(global_steps)
        try:
            self.inflight_task = asyncio.create_task(
                self._guarded_send(self._verified_sources(weights), global_steps=global_steps)
            )
            # Do not cancel an in-progress transfer on timeout: the upstream
            # coroutine owns registered buffers/descriptors and transfer handles.
            await asyncio.wait_for(asyncio.shield(self.inflight_task), timeout=self.transaction_timeout_seconds)
            self.inflight_task = None
            torch.cuda.synchronize()
            self.source_values.clear()
            self.finished = True
        except SystemExit as error:
            self.poisoned = True
            raise RuntimeError("upstream NIXL worker called exit on transfer error") from error
        except BaseException:
            self.poisoned = True
            raise

    def _verified_sources(self, weights):
        iterator = iter(weights)
        while True:
            if self.source_values:
                torch.cuda.synchronize()
                self.source_values.clear()
            try:
                name, value = next(iterator)
            except StopIteration:
                return
            self.source_values.append(value)
            if self.rank < 0:
                # Megatron collectives may still require all trainer ranks to
                # exhaust the same exporter, but only rank 0 sends data.
                yield name, value
                continue
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_cuda
                or not value.is_contiguous()
                or value.dtype not in (torch.bfloat16, torch.float32)
            ):
                raise RuntimeError("trainer must export contiguous BF16/FP32 CUDA tensors")
            if value.numel() * value.element_size() > self.max_tensor_bytes:
                raise RuntimeError("trainer source tensor exceeds transfer bound")
            # Encoding the hash in TensorMeta.name preserves the upstream
            # NIXL wire protocol without an uncoordinated side-channel. The
            # adapter sees only the original decoded HF name after verification.
            torch.cuda.synchronize(value.device)
            yield _wire_name(name, tensor_digest(value)), value

    def _new_operation(self, cls, *args):
        # Retain before __init__: a constructor can emit metadata then fail.
        operation = cls.__new__(cls)
        self.protocol_operations.append(operation)
        cls.__init__(operation, *args)
        return operation

    def _operation_done(self, operation):
        self.protocol_operations.remove(operation)

    @torch.no_grad()
    async def _guarded_send(self, weights, global_steps=None):
        # Same bounded two-bucket algorithm as pinned upstream, with explicit
        # lifetime ownership for each pending readable operation.
        if self.rank > 0:
            raise RuntimeError("trainer sender rank is invalid")
        if self.rank < 0:
            for _name, _weight in weights:
                pass
            return
        if self.next_agent is None:
            raise RuntimeError("sender has no next peer")
        send_buf, recv_buf = self.send_buf, self.recv_buf
        send_descs, recv_descs = self.send_descs, self.recv_descs
        readable = None
        bucket_meta = {}
        offset = 0
        async for meta, chunk in upstream.split_weight_chunks(weights, self.bucket_size):
            if offset + meta.chunk_size > self.bucket_size:
                torch.cuda.synchronize()
                if readable is not None:
                    await readable.wait_for_complete()
                    self._operation_done(readable)
                readable = self._new_operation(
                    upstream.ReadableOperation,
                    self.agent,
                    self.next_agent,
                    send_descs,
                    {"bucket_meta": bucket_meta, "is_last": False},
                )
                send_buf, recv_buf = recv_buf, send_buf
                send_descs, recv_descs = recv_descs, send_descs
                bucket_meta = {}
                offset = 0
            if offset + meta.chunk_size > self.bucket_size or meta.name in bucket_meta:
                raise RuntimeError("invalid outgoing bucket metadata")
            meta.offset = offset
            bucket_meta[meta.name] = meta
            send_buf[offset : offset + meta.chunk_size].copy_(chunk, non_blocking=True)
            offset += meta.chunk_size
        torch.cuda.synchronize()
        if readable is not None:
            await readable.wait_for_complete()
            self._operation_done(readable)
        readable = self._new_operation(
            upstream.ReadableOperation,
            self.agent,
            self.next_agent,
            send_descs,
            {"bucket_meta": bucket_meta, "is_last": True},
        )
        await readable.wait_for_complete()
        self._operation_done(readable)

    async def _receive_weight_chunks(self):
        """Forward each current bucket before yielding; do not await a later one.

        The pinned upstream waits for next metadata before yielding current
        metadata. A native per-source TP gather then deadlocks a multi-receiver
        ring. The existing two registered buffers remain leased until the
        consumer resumes and downstream READ has completed.
        """
        if self.prev_agent is None:
            raise RuntimeError("NIXL receiver has no previous agent")
        send_buf, recv_buf = self.send_buf, self.recv_buf
        send_descs, recv_descs = self.send_descs, self.recv_descs
        first = self._new_operation(upstream.ReadOperation, self.agent, self.prev_agent, recv_descs, self.bucket_size)
        metadata = await first.read_metadata()
        first.begin_read()
        await first.wait_for_complete()
        self._operation_done(first)
        send_buf, recv_buf = recv_buf, send_buf
        send_descs, recv_descs = recv_descs, send_descs
        while True:
            readable = None
            if self.next_agent is not None:
                # Constructor synchronously sends metadata; the downstream
                # receiver can reach its matching TP gather immediately.
                readable = self._new_operation(
                    upstream.ReadableOperation, self.agent, self.next_agent, send_descs, metadata
                )
            for tensor_meta in metadata["bucket_meta"].values():
                tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                yield tensor_meta, tensor
            if metadata["is_last"]:
                if readable is not None:
                    await readable.wait_for_complete()
                    self._operation_done(readable)
                return
            next_read = self._new_operation(
                upstream.ReadOperation, self.agent, self.prev_agent, recv_descs, self.bucket_size
            )
            next_metadata = await next_read.read_metadata()
            next_read.begin_read()
            if readable is not None:
                await readable.wait_for_complete()
                self._operation_done(readable)
            await next_read.wait_for_complete()
            self._operation_done(next_read)
            torch.cuda.synchronize()
            metadata = next_metadata
            send_buf, recv_buf = recv_buf, send_buf
            send_descs, recv_descs = recv_descs, send_descs

    def receive_weights(self, global_steps=None):
        return _NativeLeaseIterator(self, global_steps)

    def confirm_native_ack(self, name, proof):
        if (
            self.pending is None
            or self.native_acknowledged
            or name != self.pending_name
            or not isinstance(proof, dict)
            or proof.get("name") != name
            or proof.get("global_step") != self.pending_step
            or not proof.get("native_update_id")
            or not proof.get("native_incarnation")
            or proof.get("all_rank_acked") is not True
        ):
            self.poisoned = True
            raise RuntimeError("invalid native source acknowledgement")
        count = proof.get("replica_world_size")
        if type(count) is not int or count <= 0 or proof.get("acknowledged_ranks") != list(range(count)):
            self.poisoned = True
            raise RuntimeError("native acknowledgement does not cover the replica")
        binding = (proof["native_update_id"], proof["native_incarnation"], count)
        if self.native_binding is not None and self.native_binding != binding:
            self.poisoned = True
            raise RuntimeError("native transaction changed during source stream")
        self.native_binding = binding
        self.native_acknowledged = True

    async def _guarded_receive(self, global_steps):
        self._begin_transfer(global_steps)
        chunks = self._receive_weight_chunks().__aiter__()
        merge_name = merge_shape = merge_dtype = merge_value = merge_bytes = None
        merge_offset = 0
        try:
            while True:
                try:
                    self.inflight_task = asyncio.create_task(chunks.__anext__())
                    meta, chunk = await asyncio.wait_for(
                        asyncio.shield(self.inflight_task), timeout=self.timeout_seconds
                    )
                    self.inflight_task = None
                    self.partial_chunks.append(chunk)
                    if (
                        len(self.partial_chunks)
                        > (self.max_tensor_bytes + self.bucket_size - 1) // self.bucket_size + 2
                    ):
                        raise RuntimeError("source chunk count exceeds retention bound")
                except StopAsyncIteration:
                    self.inflight_task = None
                    break
                except SystemExit as error:
                    raise RuntimeError("upstream NIXL worker called exit on transfer error") from error
                if chunk.dtype != torch.uint8 or not chunk.is_cuda or not chunk.is_contiguous():
                    raise RuntimeError("NIXL returned an invalid GPU byte chunk")
                if meta.dtype not in (torch.bfloat16, torch.float32) or not meta.name:
                    raise RuntimeError("unsupported NIXL source dtype or name")
                original_name, expected_digest = _unwire_name(meta.name)
                shape = tuple(meta.shape)
                if not shape or any(type(n) is not int or n <= 0 for n in shape):
                    raise RuntimeError("invalid source shape")
                size = int(meta.dtype.itemsize)
                for n in shape:
                    size *= n
                if size > self.max_tensor_bytes:
                    raise RuntimeError("complete HF tensor exceeds reassembly bound")
                if (
                    type(meta.chunk_offset) is not int
                    or meta.chunk_offset < 0
                    or type(meta.chunk_size) is not int
                    or meta.chunk_size <= 0
                    or meta.chunk_size != chunk.numel()
                    or meta.chunk_offset + meta.chunk_size > size
                ):
                    raise RuntimeError("invalid NIXL source chunk bounds")
                if size <= self.bucket_size:
                    if merge_value is not None or meta.chunk_offset != 0 or meta.chunk_size != size:
                        raise RuntimeError("partial or overlapping short source")
                    value = chunk.view(meta.dtype).view(shape)
                else:
                    if merge_value is None:
                        if meta.chunk_offset != 0:
                            raise RuntimeError("large source did not begin at offset zero")
                        merge_name, merge_shape, merge_dtype = meta.name, shape, meta.dtype
                        merge_value = torch.empty(shape, dtype=meta.dtype, device=chunk.device)
                        self.partial_merge_value = merge_value
                        merge_bytes = merge_value.view(-1).view(torch.uint8)
                        merge_offset = 0
                    if (meta.name, shape, meta.dtype, meta.chunk_offset) != (
                        merge_name,
                        merge_shape,
                        merge_dtype,
                        merge_offset,
                    ):
                        raise RuntimeError("large source chunk order/shape mismatch")
                    merge_bytes[merge_offset : merge_offset + meta.chunk_size].copy_(chunk)
                    merge_offset += meta.chunk_size
                    if merge_offset < size:
                        continue
                    value = merge_value
                    merge_name = merge_shape = merge_dtype = merge_value = merge_bytes = None
                    merge_offset = 0
                if tensor_digest(value) != expected_digest:
                    raise RuntimeError("NIXL source differs from trainer SHA256")
                self.pending, self.pending_name, self.pending_step = value, original_name, global_steps
                self.partial_merge_value = None
                self.native_acknowledged = False
                yield original_name, value
                if not self.native_acknowledged:
                    raise RuntimeError("adapter advanced without a native all-rank ACK")
                self.pending = self.pending_name = self.pending_step = None
                self.native_acknowledged = False
                self.partial_chunks.clear()
                # Adapter must discard its own tensor reference before __anext__.
                del value, chunk
            if merge_value is not None:
                raise RuntimeError("NIXL stream ended inside a source tensor")
            self.finished = True
        except BaseException:
            self.poisoned = True
            # Retain current pending allocation; do not call upstream finalize
            # while any native reader or peer NIXL operation may still hold it.
            raise

    def finalize(self):
        if (
            self.poisoned
            or not self.active
            or not self.finished
            or self.pending is not None
            or self.partial_merge_value is not None
            or self.partial_chunks
            or self.protocol_operations
            or self.source_values
            or self.inflight_task is not None
        ):
            raise RuntimeError("NIXL generation failed or still owns an IPC lease")
        try:
            super().finalize()
        except BaseException:
            self.poisoned = True
            raise
        self.active = False
        self.native_binding = None
