# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""PyTorch ExportedProgram of relay."""
from collections import ChainMap, OrderedDict
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
import tvm
from tvm import relay

from base_fx_graph_translator import BaseFXGraphImporter


class _RelayEmitAdapter:
    """Relay ScopeBuilder에 맞춘 간단한 emit 어댑터"""

    def __init__(self, sb: relay.ScopeBuilder):
        self.sb = sb
        self._idx = 0

    def emit(self, expr: relay.Expr) -> relay.Var:
        name = f"v{self._idx}"
        self._idx += 1
        return self.sb.let(name, expr)

    # dataflow/emit_output/emit_func_output 대체용 no-op/헬퍼
    def dataflow(self):
        # Relay에는 별도의 dataflow 스코프가 없으니 contextmanager 더미 제공
        class _NoOp:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _NoOp()

    def emit_output(self, expr: relay.Expr) -> relay.Expr:
        # Relax의 emit_output은 SSA 바인딩을 만들지만 Relay에선 최종적으로 반환할 expr만 넘기면 됩니다.
        return expr

    def emit_func_output(self, expr: relay.Expr):
        # Relax에선 함수 블록 종료시 사용했지만, Relay에선 sb.ret(...)로 대체하므로 여기선 사용하지 않음
        return expr


class ExportedProgramImporter(BaseFXGraphImporter):
    """An importer from ExportedProgram to relay."""

    from torch import fx

    ########## Unary Ops ##########

    def _hardtanh(self, node: fx.Node) -> relay.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.args[1] if len(args) > 1 else node.kwargs.get("min_val", -1.0)
        max_val = node.args[2] if len(args) > 2 else node.kwargs.get("max_val", 1.0)
        return self.block_builder.emit(relay.op.clip(x, min_val, max_val))

    def _log2(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(
            relay.op.divide(
                relay.op.log(x), relay.const(0.6931471805599453, x.struct_info.dtype)
            )
        )

    def _log10(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(
            relay.op.divide(
                relay.op.log(x), relay.const(2.302585092994046, x.struct_info.dtype)
            )
        )

    def _log1p(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        one = relay.const(1, x.struct_info.dtype)
        return self.block_builder.emit(relay.op.log(relay.op.add(x, one)))

    def _reciprocal(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(
            relay.op.divide(relay.const(1.0, x.struct_info.dtype), x)
        )

    def _sub(self, node: fx.Node) -> relay.Var:
        x, y = self.retrieve_args(node)
        return self.block_builder.emit(relay.op.subtract(x, y))

    def _div(self, node: fx.Node) -> relay.Var:
        x, y = self.retrieve_args(node)
        return self.block_builder.emit(relay.op.divide(x, y))

    ########## Prim ##########

    # prim::TupleConstruct: tuple 생성
    def prim_TupleConstruct(self, node):
        elems = [self.get_arg_value(arg) for arg in node.args]
        return self.block_builder.emit(relay.expr.Tuple(elems))

    # prim::TupleUnpack: tuple 해체
    def prim_TupleUnpack(self, node):
        tup = self.retrieve_args(node)[0]
        fields = [relay.TupleGetItem(tup, i) for i in range(len(tup.fields))]
        return fields

    ########## Neural Network ##########

    def _convolution_default(self, node: fx.Node) -> relay.Var:
        """
        Handle torch.ops.aten.convolution.default

        Args signature: (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
        """
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        weight_shape = self.get_shape_weight(weight)
        out_channels = weight_shape[0]
        kernel_size = (weight_shape[2], weight_shape[3])
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else [1, 1]
        padding = args[4] if len(args) > 4 else [0, 0]
        dilation = args[5] if len(args) > 5 else [1, 1]
        transposed = args[6] if len(args) > 6 else False
        output_padding = args[7] if len(args) > 7 else [0, 0]
        groups = args[8] if len(args) > 8 else 1

        # For now, only support non-transposed convolutions
        if transposed:
            raise NotImplementedError(
                "Transposed convolutions are not yet supported in convolution.default"
            )

        # Delegate to the existing _conv2d_impl from the base class
        return self._conv2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def _batch_norm(self, node: fx.Node, training: bool) -> relay.Var:
        import numpy as np
        from torch import fx as fx

        x, x_shape, dtype = self.get_shape_relay_Var(node)
        channel = int(x_shape[1])
        weight = self.env.get(node.args[1], relay.const(np.ones(channel), dtype=dtype))
        bias = self.env.get(node.args[2], relay.const(np.zeros(channel), dtype=dtype))
        running_mean = self.env.get(
            node.args[3], relay.const(np.zeros(channel), dtype=dtype)
        )
        running_var = self.env.get(
            node.args[4], relay.const(np.ones(channel), dtype=dtype)
        )
        ignore_running_stats = (
            node.args[5]
            if len(node.args) > 5
            else node.kwargs.get("track_running_stats", True)
        )
        track_running_stats = not ignore_running_stats
        momentum = (
            node.args[6] if len(node.args) > 6 else node.kwargs.get("momentum", 0.1)
        )
        eps = node.args[7] if len(node.args) > 7 else node.kwargs.get("eps", 1e-05)

        if track_running_stats:
            training = True

        return self.block_builder.emit(
            relay.nn.batch_norm(
                data=x,
                gamma=weight,
                beta=bias,
                moving_mean=running_mean,
                moving_var=running_var,
                axis=1,  # Always over channel
                epsilon=eps,
            )[0]
        )

    def _batch_norm_legit_functional(self, node: fx.Node) -> relay.Var:
        # This method is called for batch_norm in training mode
        # TODO does not have correctness!
        # TODO we need to store the running mean and variance returned by the
        # previous call to batch_norm and pass it again
        training = True
        return self._batch_norm(node, training)

    def _batch_norm_legit_no_training(self, node: fx.Node) -> relay.Var:
        # This method is called for batch_norm in eval mode
        training = False
        return self._batch_norm(node, training)

    def _cross_entropy_default(self, node: fx.Node) -> relay.Expr:
        preds = self.env[node.args[0]]
        targets = self.env[node.args[1]]
        weight = self.env.get(node.args[2], None) if len(node.args) > 2 else None
        reduction = node.kwargs.get("reduction", "mean")
        ignore_index = node.kwargs.get("ignore_index", -100)
        return self._cross_entropy_loss(preds, targets, weight, reduction, ignore_index)

    def _group_norm(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        num_groups = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05

        dim = len(self.shape_of(x))
        return self.block_builder.emit(
            relay.op.nn.group_norm(
                x,
                gamma,
                beta,
                num_groups=num_groups,
                channel_axis=1,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _upsample_impl(
        self,
        x: relay.Expr,
        node: fx.Node,
        size,
        scale_factor,
        method: str,
        align_corners: bool,
    ) -> relay.Var:
        coord_trans = "align_corners" if align_corners else "half_pixel"
        _, x_shape, _ = self.get_shape_relay_Var(node)

        # 🔹 size 계산 (조건문 제거, 직접 계산)
        if size is None and scale_factor is not None:
            # scale_factor를 미리 평가하여 조건문 제거
            try:
                # scale_factor가 스칼라인지 확인
                sf_h = (
                    scale_factor[0]
                    if hasattr(scale_factor, "__getitem__")
                    else scale_factor
                )
                sf_w = (
                    scale_factor[1]
                    if hasattr(scale_factor, "__getitem__") and len(scale_factor) > 1
                    else sf_h
                )
            except (TypeError, IndexError):
                # scale_factor가 스칼라인 경우
                sf_h = sf_w = scale_factor

            out_h = int(x_shape[2] * sf_h)
            out_w = int(x_shape[3] * sf_w)
            size = (out_h, out_w)

        return self.block_builder.emit(
            relay.op.image.resize2d(
                x,
                size=size,
                layout="NCHW",
                method=method,
                coordinate_transformation_mode=coord_trans,
            )
        )

    def _upsample_bilinear2d(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2]
            if len(node.args) > 2
            else node.kwargs.get("align_corners", True)
        )
        scale_factor = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor", 1)
        )
        return self._upsample_impl(
            x,
            node,
            size=size,
            scale_factor=scale_factor,
            method="linear",
            align_corners=align_corners,
        )

    def _upsample_nearest2d(self, node: fx.node) -> relay.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)

        if size:
            scale_factor = None  # Can only define size or scale_factor, not both
            align_corners = (
                node.args[2]
                if len(node.args) > 2
                else node.kwargs.get("align_corners", None)
            )

        else:
            # TODO figure out why pytorch export passes a list such as
            # [scale_factor,scale_factor] instead of just an int for
            # scale_factor. Using first element for now
            scale_factor = (
                node.args[2][0]
                if len(node.args) > 2
                else node.kwargs.get("scale_factor", 1)
            )
            align_corners = (
                node.args[3]
                if len(node.args) > 3
                else node.kwargs.get("align_corners", None)
            )

        return self._upsample_impl(
            x,
            node,
            size=size,
            scale_factor=scale_factor,
            method="nearest_neighbor",
            align_corners=align_corners,
        )

    def _upsample_bicubic2d(self, node: fx.node) -> relay.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2]
            if len(node.args) > 2
            else node.kwargs.get("align_corners", None)
        )
        if size is not None:
            scale_factor = None
        else:
            scale_arg = (
                node.args[3]
                if len(node.args) > 3
                else node.kwargs.get("scale_factor", 1)
            )
            if isinstance(scale_arg, (list, tuple)):
                scale_factor = scale_arg[0]
            else:
                scale_factor = scale_arg

        return self._upsample_impl(
            x,
            node,
            size=size,
            scale_factor=scale_factor,
            method="cubic",
            align_corners=align_corners,
        )

    def _lstm(self, node: fx.Node) -> relay.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        hx = args[1] if len(args) > 1 else None
        params = args[2] if len(args) > 2 else None
        has_biases = args[3] if len(args) > 3 else True
        num_layers = args[4] if len(args) > 4 else 1
        _dropout = args[5] if len(args) > 5 else 0.0  # Not used in inference
        _train = args[6] if len(args) > 6 else False  # Not used in inference
        bidirectional = args[7] if len(args) > 7 else False
        batch_first = args[8] if len(args) > 8 else False
        if bidirectional:
            raise NotImplementedError("Bidirectional LSTM is not yet supported")
        if num_layers > 1:
            raise NotImplementedError("Multi-layer LSTM is not yet supported")
        input_shape = self.shape_of(input_tensor)
        if batch_first:
            # Input shape: (batch, seq_len, input_size)
            batch_size, seq_len, input_size = input_shape
        else:
            # Input shape: (seq_len, batch, input_size)
            seq_len, batch_size, input_size = input_shape

        if isinstance(seq_len, tvm.tir.IntImm):
            seq_len = seq_len.value
        if isinstance(batch_size, tvm.tir.IntImm):
            batch_size = batch_size.value
        if isinstance(input_size, tvm.tir.IntImm):
            input_size = input_size.value
        # Extract hidden size from the LSTM parameters
        # The parameters are: [weight_ih, weight_hh, bias_ih, bias_hh]
        # weight_ih shape: (4 * hidden_size, input_size)
        # weight_hh shape: (4 * hidden_size, hidden_size)
        if params and len(params) >= 2:
            weight_ih = params[0]
            weight_hh = params[1]
            # Extract hidden size from weight dimensions
            # weight_ih has shape (4 * hidden_size, input_size)
            weight_ih_shape = self.shape_of(weight_ih)
            hidden_size = (
                weight_ih_shape[0] // 4
            )  # 4 gates: input, forget, cell, output
        else:
            # Fallback to a default hidden size
            hidden_size = 16
        # Implement actual LSTM computation using  relay operations
        # LSTM equations:
        # i_t = sigmoid(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
        # f_t = sigmoid(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
        # g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
        # o_t = sigmoid(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)
        # c_t = f_t * c_{t-1} + i_t * g_t
        # h_t = o_t * tanh(c_t)
        dtype = input_tensor.struct_info.dtype
        if params and len(params) >= 4:
            weight_ih = params[0]  # (4 * hidden_size, input_size)
            weight_hh = params[1]  # (4 * hidden_size, hidden_size)
            bias_ih = params[2] if has_biases else None  # (4 * hidden_size,)
            bias_hh = params[3] if has_biases else None  # (4 * hidden_size,)
        else:
            # Fallback: create zero weights
            weight_ih = self.block_builder.emit(
                relay.op.zeros(relay.ShapeExpr((4 * hidden_size, input_size)), dtype)
            )
            weight_hh = self.block_builder.emit(
                relay.op.zeros(relay.ShapeExpr((4 * hidden_size, hidden_size)), dtype)
            )
            bias_ih = None
            bias_hh = None
        # Initialize hidden and cell states
        if hx is not None and len(hx) >= 2:
            h_0 = hx[0]  # (num_layers, batch_size, hidden_size)
            c_0 = hx[1]  # (num_layers, batch_size, hidden_size)
            # Extract the first layer's hidden state
            h_prev = self.block_builder.emit(
                relay.op.take(h_0, relay.const(0, "int64"), axis=0, mode="clip")
            )
            c_prev = self.block_builder.emit(
                relay.op.take(c_0, relay.const(0, "int64"), axis=0, mode="clip")
            )
        else:
            h_prev = self.block_builder.emit(
                relay.op.zeros(relay.ShapeExpr((batch_size, hidden_size)), dtype)
            )
            c_prev = self.block_builder.emit(
                relay.op.zeros(relay.ShapeExpr((batch_size, hidden_size)), dtype)
            )
        # Reshape input for processing
        if batch_first:
            # Input: (batch, seq_len, input_size) -> (seq_len, batch, input_size)
            input_reshaped = self.block_builder.emit(
                relay.op.permute_dims(input_tensor, axes=[1, 0, 2])
            )
        else:
            input_reshaped = input_tensor
        weight_ih_t = self.block_builder.emit(
            relay.op.permute_dims(weight_ih, axes=[1, 0])
        )
        weight_hh_t = self.block_builder.emit(
            relay.op.permute_dims(weight_hh, axes=[1, 0])
        )
        outputs = []
        for t in range(seq_len):
            # Get input at time t: (batch_size, input_size)
            x_t = self.block_builder.emit(
                relay.op.take(
                    input_reshaped, relay.const(t, "int64"), axis=0, mode="clip"
                )
            )
            # Compute gates: W_ih * x_t + W_hh * h_{t-1} + bias
            # Input-to-hidden: (batch_size, input_size) @ (4*hidden_size, input_size).T
            ih_gates = self.block_builder.emit(
                relay.op.linear_algebra.matmul(x_t, weight_ih_t)
            )

            # Hidden-to-hidden: (batch_size, hidden_size) @ (4*hidden_size, hidden_size).T
            hh_gates = self.block_builder.emit(
                relay.op.linear_algebra.matmul(h_prev, weight_hh_t)
            )
            # Add biases if present
            if bias_ih is not None and bias_hh is not None:
                gates = self.block_builder.emit(
                    relay.op.add(
                        relay.op.add(relay.op.add(ih_gates, bias_ih), hh_gates), bias_hh
                    )
                )
            elif bias_ih is not None:
                gates = self.block_builder.emit(
                    relay.op.add(relay.op.add(ih_gates, bias_ih), hh_gates)
                )
            elif bias_hh is not None:
                gates = self.block_builder.emit(
                    relay.op.add(relay.op.add(ih_gates, hh_gates), bias_hh)
                )
            else:
                gates = self.block_builder.emit(relay.op.add(ih_gates, hh_gates))
            # Split gates: (batch_size, 4 * hidden_size) -> 4 x (batch_size, hidden_size)
            gate_size = hidden_size
            i_gate = self.block_builder.emit(
                relay.op.strided_slice(gates, axes=[1], begin=[0], end=[gate_size])
            )
            f_gate = self.block_builder.emit(
                relay.op.strided_slice(
                    gates, axes=[1], begin=[gate_size], end=[2 * gate_size]
                )
            )
            g_gate = self.block_builder.emit(
                relay.op.strided_slice(
                    gates, axes=[1], begin=[2 * gate_size], end=[3 * gate_size]
                )
            )
            o_gate = self.block_builder.emit(
                relay.op.strided_slice(
                    gates, axes=[1], begin=[3 * gate_size], end=[4 * gate_size]
                )
            )
            # Apply activations
            i_t = self.block_builder.emit(relay.op.sigmoid(i_gate))
            f_t = self.block_builder.emit(relay.op.sigmoid(f_gate))
            g_t = self.block_builder.emit(relay.op.tanh(g_gate))
            o_t = self.block_builder.emit(relay.op.sigmoid(o_gate))
            # Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
            c_t = self.block_builder.emit(
                relay.op.add(
                    relay.op.multiply(f_t, c_prev), relay.op.multiply(i_t, g_t)
                )
            )
            # Update hidden state: h_t = o_t * tanh(c_t)
            h_t = self.block_builder.emit(relay.op.multiply(o_t, relay.op.tanh(c_t)))
            # Store output
            outputs.append(h_t)
            # Update for next iteration
            h_prev = h_t
            c_prev = c_t
        # Stack outputs: (seq_len, batch_size, hidden_size)
        output = self.block_builder.emit(relay.op.stack(outputs, axis=0))
        # Reshape back to batch_first if needed
        if batch_first:
            # (seq_len, batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
            output = self.block_builder.emit(
                relay.op.permute_dims(output, axes=[1, 0, 2])
            )
        return output

    def _gru(self, node: fx.Node) -> relay.Var:
        args = self.retrieve_args(node)
        input_tensor = args[0]
        hx = args[1] if len(args) > 1 else None
        params = args[2] if len(args) > 2 else None
        has_biases = args[3] if len(args) > 3 else True
        num_layers = args[4] if len(args) > 4 else 1
        _dropout = args[5] if len(args) > 5 else 0.0  # Not used in inference
        _train = args[6] if len(args) > 6 else False  # Not used in inference
        bidirectional = args[7] if len(args) > 7 else False
        batch_first = args[8] if len(args) > 8 else False

        if bidirectional:
            raise NotImplementedError("Bidirectional GRU is not yet supported")

        input_shape = self.shape_of(input_tensor)
        if batch_first:
            batch_size, seq_len, input_size = input_shape
        else:
            seq_len, batch_size, input_size = input_shape

        if isinstance(seq_len, tvm.tir.IntImm):
            seq_len = seq_len.value
        if isinstance(batch_size, tvm.tir.IntImm):
            batch_size = batch_size.value
        if isinstance(input_size, tvm.tir.IntImm):
            input_size = input_size.value

        if params and len(params) >= 2:
            # For multi-layer, we need to extract the first layer's weights
            # to determine hidden size
            if num_layers > 1:
                # Multi-layer: params[0] is first layer's weight_ih
                weight_ih = params[0]
            else:
                # Single layer: params[0] is weight_ih
                weight_ih = params[0]
            # Extract hidden size from weight dimensions
            # weight_ih has shape (3 * hidden_size, input_size)
            weight_ih_shape = self.shape_of(weight_ih)
            hidden_size = weight_ih_shape[0] // 3  # 3 gates: reset, update, new
        else:
            # Fallback to a default hidden size
            hidden_size = 16

        # Implement actual GRU computation using relay operations
        # GRU equations:
        # r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
        # z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
        # n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn))
        # h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        dtype = input_tensor.struct_info.dtype

        # Reshape input for processing
        if batch_first:
            # Input: (batch, seq_len, input_size) -> (seq_len, batch, input_size)
            input_reshaped = self.block_builder.emit(
                relay.op.permute_dims(input_tensor, axes=[1, 0, 2])
            )
        else:
            input_reshaped = input_tensor

        # Initialize hidden states for all layers
        if hx is not None:
            # hx shape: (num_layers, batch_size, hidden_size)
            h_states = []
            for layer in range(num_layers):
                h_layer = self.block_builder.emit(
                    relay.op.take(hx, relay.const(layer, "int64"), axis=0, mode="clip")
                )
                h_states.append(h_layer)
        else:
            h_states = []
            for layer in range(num_layers):
                h_layer = self.block_builder.emit(
                    relay.op.zeros(relay.ShapeExpr((batch_size, hidden_size)), dtype)
                )
                h_states.append(h_layer)

        outputs = []

        for t in range(seq_len):
            # Get input at time t: (batch_size, input_size)
            x_t = self.block_builder.emit(
                relay.op.take(
                    input_reshaped, relay.const(t, "int64"), axis=0, mode="clip"
                )
            )

            # Process through each layer
            current_input = x_t
            new_h_states = []

            for layer in range(num_layers):
                # Get layer parameters
                if params and len(params) >= 4 * num_layers:
                    # Multi-layer case: params are organized as
                    # [layer0_ih, layer0_hh, layer0_bias_ih, layer0_bias_hh, layer1_ih, ...]
                    param_offset = layer * 4
                    weight_ih = params[param_offset]
                    weight_hh = params[param_offset + 1]
                    bias_ih = params[param_offset + 2] if has_biases else None
                    bias_hh = params[param_offset + 3] if has_biases else None
                elif params and len(params) >= 4:
                    # Single layer case
                    weight_ih = params[0]
                    weight_hh = params[1]
                    bias_ih = params[2] if has_biases else None
                    bias_hh = params[3] if has_biases else None
                else:
                    # Fallback: create zero weights
                    weight_ih = self.block_builder.emit(
                        relay.op.zeros(
                            relay.ShapeExpr(
                                (
                                    3 * hidden_size,
                                    input_size if layer == 0 else hidden_size,
                                )
                            ),
                            dtype,
                        )
                    )
                    weight_hh = self.block_builder.emit(
                        relay.op.zeros(
                            relay.ShapeExpr((3 * hidden_size, hidden_size)), dtype
                        )
                    )
                    bias_ih = None
                    bias_hh = None

                # Get previous hidden state for this layer
                h_prev = h_states[layer]

                # Split weights by gates: PyTorch GRU gate order: reset, update, new (r, z, n)
                gate_size = hidden_size

                # Reset gate weights
                weight_ih_r = self.block_builder.emit(
                    relay.op.strided_slice(
                        weight_ih, axes=[0], begin=[0], end=[gate_size]
                    )
                )
                weight_hh_r = self.block_builder.emit(
                    relay.op.strided_slice(
                        weight_hh, axes=[0], begin=[0], end=[gate_size]
                    )
                )

                # Update gate weights
                weight_ih_z = self.block_builder.emit(
                    relay.op.strided_slice(
                        weight_ih, axes=[0], begin=[gate_size], end=[2 * gate_size]
                    )
                )
                weight_hh_z = self.block_builder.emit(
                    relay.op.strided_slice(
                        weight_hh, axes=[0], begin=[gate_size], end=[2 * gate_size]
                    )
                )

                # New gate weights
                weight_ih_n = self.block_builder.emit(
                    relay.op.strided_slice(
                        weight_ih, axes=[0], begin=[2 * gate_size], end=[3 * gate_size]
                    )
                )
                weight_hh_n = self.block_builder.emit(
                    relay.op.strided_slice(
                        weight_hh, axes=[0], begin=[2 * gate_size], end=[3 * gate_size]
                    )
                )

                # Transpose weights for matmul
                weight_ih_r_t = self.block_builder.emit(
                    relay.op.permute_dims(weight_ih_r, axes=[1, 0])
                )
                weight_hh_r_t = self.block_builder.emit(
                    relay.op.permute_dims(weight_hh_r, axes=[1, 0])
                )
                weight_ih_z_t = self.block_builder.emit(
                    relay.op.permute_dims(weight_ih_z, axes=[1, 0])
                )
                weight_hh_z_t = self.block_builder.emit(
                    relay.op.permute_dims(weight_hh_z, axes=[1, 0])
                )
                weight_ih_n_t = self.block_builder.emit(
                    relay.op.permute_dims(weight_ih_n, axes=[1, 0])
                )
                weight_hh_n_t = self.block_builder.emit(
                    relay.op.permute_dims(weight_hh_n, axes=[1, 0])
                )

                # Compute reset gate: r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
                r_ih = self.block_builder.emit(
                    relay.op.linear_algebra.matmul(current_input, weight_ih_r_t)
                )
                r_hh = self.block_builder.emit(
                    relay.op.linear_algebra.matmul(h_prev, weight_hh_r_t)
                )
                if bias_ih is not None and bias_hh is not None:
                    bias_ih_r = self.block_builder.emit(
                        relay.op.strided_slice(
                            bias_ih, axes=[0], begin=[0], end=[gate_size]
                        )
                    )
                    bias_hh_r = self.block_builder.emit(
                        relay.op.strided_slice(
                            bias_hh, axes=[0], begin=[0], end=[gate_size]
                        )
                    )
                    r_t = self.block_builder.emit(
                        relay.op.sigmoid(
                            relay.op.add(
                                relay.op.add(relay.op.add(r_ih, bias_ih_r), r_hh),
                                bias_hh_r,
                            )
                        )
                    )
                else:
                    r_t = self.block_builder.emit(
                        relay.op.sigmoid(relay.op.add(r_ih, r_hh))
                    )

                # Compute update gate: z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
                z_ih = self.block_builder.emit(
                    relay.op.linear_algebra.matmul(current_input, weight_ih_z_t)
                )
                z_hh = self.block_builder.emit(
                    relay.op.linear_algebra.matmul(h_prev, weight_hh_z_t)
                )
                if bias_ih is not None and bias_hh is not None:
                    bias_ih_z = self.block_builder.emit(
                        relay.op.strided_slice(
                            bias_ih, axes=[0], begin=[gate_size], end=[2 * gate_size]
                        )
                    )
                    bias_hh_z = self.block_builder.emit(
                        relay.op.strided_slice(
                            bias_hh, axes=[0], begin=[gate_size], end=[2 * gate_size]
                        )
                    )
                    z_t = self.block_builder.emit(
                        relay.op.sigmoid(
                            relay.op.add(
                                relay.op.add(relay.op.add(z_ih, bias_ih_z), z_hh),
                                bias_hh_z,
                            )
                        )
                    )
                else:
                    z_t = self.block_builder.emit(
                        relay.op.sigmoid(relay.op.add(z_ih, z_hh))
                    )

                # Compute new gate: n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn))
                n_ih = self.block_builder.emit(
                    relay.op.linear_algebra.matmul(current_input, weight_ih_n_t)
                )
                n_hh = self.block_builder.emit(
                    relay.op.linear_algebra.matmul(h_prev, weight_hh_n_t)
                )
                if bias_ih is not None and bias_hh is not None:
                    bias_ih_n = self.block_builder.emit(
                        relay.op.strided_slice(
                            bias_ih,
                            axes=[0],
                            begin=[2 * gate_size],
                            end=[3 * gate_size],
                        )
                    )
                    bias_hh_n = self.block_builder.emit(
                        relay.op.strided_slice(
                            bias_hh,
                            axes=[0],
                            begin=[2 * gate_size],
                            end=[3 * gate_size],
                        )
                    )
                    n_t = self.block_builder.emit(
                        relay.op.tanh(
                            relay.op.add(
                                relay.op.add(n_ih, bias_ih_n),
                                relay.op.multiply(r_t, relay.op.add(n_hh, bias_hh_n)),
                            )
                        )
                    )
                else:
                    n_t = self.block_builder.emit(
                        relay.op.tanh(relay.op.add(n_ih, relay.op.multiply(r_t, n_hh)))
                    )

                # Update hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
                one_minus_z = self.block_builder.emit(
                    relay.op.subtract(relay.const(1.0, dtype), z_t)
                )
                h_t = self.block_builder.emit(
                    relay.op.add(
                        relay.op.multiply(one_minus_z, n_t),
                        relay.op.multiply(z_t, h_prev),
                    )
                )

                new_h_states.append(h_t)

                current_input = h_t

            # Update hidden states for next time step
            h_states = new_h_states

            # Store output (from the last layer)
            outputs.append(h_states[-1])

        # Stack outputs: (seq_len, batch_size, hidden_size)
        output = self.block_builder.emit(relay.op.stack(outputs, axis=0))

        # Reshape back to batch_first if needed
        if batch_first:
            # (seq_len, batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
            output = self.block_builder.emit(
                relay.op.permute_dims(output, axes=[1, 0, 2])
            )

        return output

    ########## Manipulation ##########

    def _narrow(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        start = node.args[2]
        length = node.args[3]
        return self.block_builder.emit(
            relay.op.strided_slice(x, [dim], [start], [length])
        )

    def _select(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = relay.const(node.args[2], "int64")
        return self.block_builder.emit(relay.op.take(x, index, dim))

    def _slice(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        axes = [node.args[1]]
        begin = [node.args[2]]
        end = [node.args[3]]
        stride = [node.args[4] if len(node.args) > 4 else 1]
        return self.block_builder.emit(
            relay.op.strided_slice(x, axes, begin, end, stride)
        )

    def _unflatten(self, node: fx.Node) -> relay.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = node.args[1]
        sizes = node.args[2]

        x_shape = list(self.shape_of(x))
        if dim < 0:
            dim += len(x_shape)

        new_shape = x_shape[:dim] + sizes + x_shape[dim + 1 :]
        return self.block_builder.emit(relay.op.reshape(x, new_shape))

    ########## Creation ##########

    def _one_hot(self, node: fx.Node) -> relay.Var:
        x = self.env[node.args[0]]
        num_classes = (
            node.args[1] if len(node.args) > 1 else node.kwargs.get("num_classes")
        )
        if num_classes is None:
            raise ValueError("num_classes not found in node.args or node.kwargs")

        on_value = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("on_value", 1)
        )
        off_value = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("off_value", 0)
        )
        axis = node.args[4] if len(node.args) > 4 else node.kwargs.get("axis", -1)

        on_value = relay.PrimValue(on_value)
        off_value = relay.PrimValue(off_value)

        return self.block_builder.emit(
            relay.op.one_hot(x, on_value, off_value, num_classes, axis)
        )

    def _hamming_window(self, node: fx.Node) -> relay.Var:
        args = self.retrieve_args(node)

        window_size = args[0]
        periodic = args[1] if len(args) > 1 else True
        alpha = args[2] if len(args) > 2 else 0.54
        beta = args[3] if len(args) > 3 else 0.46
        dtype = node.kwargs.get("dtype", "float")
        dtype = self._convert_data_type(dtype)

        return self.block_builder.emit(
            relay.op.hamming_window(window_size, periodic, alpha, beta, dtype)
        )

    def _zeros(self, node: fx.Node) -> relay.Var:
        args = self.retrieve_args(node)
        size = relay.ShapeExpr(
            args[0] if isinstance(args[0], (list, tuple)) else (args[0],)
        )
        dtype = self._convert_data_type(
            node.kwargs.get("dtype", torch.get_default_dtype()), self.env
        )
        return self.block_builder.emit(relay.op.zeros(size, dtype))

    def _instance_norm(self, node: fx.Node):
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        gamma = self.env.get(node.args[1], relay.const(np.ones(channel), dtype=dtype))
        beta = self.env.get(node.args[2], relay.const(np.zeros(channel), dtype=dtype))
        eps = node.args[4] if node.args[4] else 1e-05
        channel_axis = 1
        dim = len(self.shape_of(x))

        return self.block_builder.emit(
            relay.op.nn.instance_norm(
                x,
                gamma,
                beta,
                channel_axis=channel_axis,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    ########## Others ##########

    def create_convert_map(
        self,
    ) -> Dict[str, Callable[[fx.Node], relay.Var]]:
        import operator

        return {
            # unary
            "abs.default": self._unary_op(relay.op.abs),
            "acos.default": self._unary_op(relay.op.acos),
            "acosh.default": self._unary_op(relay.op.acosh),
            "asin.default": self._unary_op(relay.op.asin),
            "asinh.default": self._unary_op(relay.op.asinh),
            "atan.default": self._unary_op(relay.op.atan),
            "atanh.default": self._unary_op(relay.op.atanh),
            "bitwise_not.default": self._unary_op(relay.op.bitwise_not),
            "ceil.default": self._unary_op(relay.op.ceil),
            "celu.default": self._celu,
            "clamp.default": self._clamp,
            "clamp_min.default": self._clamp_min,
            "clamp_max.default": self._clamp_max,
            "convolution.default": self._convolution_default,
            "cos.default": self._unary_op(relay.op.cos),
            "cosh.default": self._unary_op(relay.op.cosh),
            "dropout.default": lambda node: self.env[node.args[0]],
            "dropout_.default": lambda node: self.env[node.args[0]],
            "elu.default": self._elu,
            "elu_.default": self._elu,
            "erf.default": self._unary_op(relay.op.erf),
            "exp.default": self._unary_op(relay.op.exp),
            "floor.default": self._unary_op(relay.op.floor),
            "gelu.default": self._gelu,
            "hardsigmoid.default": self._hardsigmoid,
            "sigmoid.default": self._unary_op(relay.op.sigmoid),
            "hardswish.default": self._hardswish,
            "hardswish_.default": self._hardswish,
            "hardtanh.default": self._hardtanh,
            "hardtanh_.default": self._hardtanh,
            "isfinite.default": self._unary_op(relay.op.isfinite),
            "isinf.default": self._unary_op(relay.op.isinf),
            "isin.Tensor_Tensor": self._isin,
            "isnan.default": self._unary_op(relay.op.isnan),
            "leaky_relu.default": self._leakyrelu,
            "leaky_relu_.default": self._leakyrelu,
            "log.default": self._unary_op(relay.op.log),
            "log2.default": self._log2,
            "log10.default": self._log10,
            "log1p.default": self._log1p,
            "logical_not.default": self._unary_op(relay.op.logical_not),
            "log_softmax.int": self._log_softmax,
            "neg.default": self._unary_op(relay.op.negative),
            "pad.default": self._pad,
            "pixel_shuffle.default": self._pixel_shuffle,
            "prelu.default": self._prelu,
            "reciprocal.default": self._reciprocal,
            "relu.default": self._unary_op(relay.op.nn.relu),
            "relu_.default": self._unary_op(relay.op.nn.relu),
            "round.default": self._round,
            "rsqrt.default": self._unary_op(relay.op.rsqrt),
            "rsub.Tensor": self._rsub,
            "rsub.Scalar": self._rsub,
            "softmax.int": self._softmax,
            "softplus.default": self._softplus,
            "softshrink.default": self._softshrink,
            "softsign.default": self._softsign,
            "tan.default": self._unary_op(relay.op.tan),
            "tanh.default": self._unary_op(relay.op.tanh),
            "trunc.default": self._unary_op(relay.op.trunc),
            # binary
            "add.Tensor": self._binary_op(relay.op.add, operator.add),
            "add_.Tensor": self._binary_op(relay.op.add, operator.add),
            "bitwise_or_.Scalar": self._binary_op(relay.op.bitwise_or, operator.or_),
            "bitwise_or.Scalar": self._binary_op(relay.op.bitwise_or, operator.or_),
            "bitwise_or_.Tensor": self._binary_op(relay.op.bitwise_or, operator.or_),
            "bitwise_or.Tensor": self._binary_op(relay.op.bitwise_or, operator.or_),
            "div.Tensor": self._binary_op(relay.op.divide, operator.truediv),
            "div.Tensor_mode": self._div,
            "eq": self._binary_op(relay.op.equal, operator.eq),
            "eq.Scalar": self._binary_op(relay.op.equal, operator.eq),
            "eq.Tensor": self._binary_op(relay.op.equal, operator.eq),
            "floor_divide.default": self._binary_op(
                relay.op.floor_divide, operator.floordiv
            ),
            "fmod.Scalar": self._fmod,
            "fmod.Tensor": self._fmod,
            "ge.Scalar": self._binary_op(relay.op.greater_equal, operator.ge),
            "ge.Tensor": self._binary_op(relay.op.greater_equal, operator.ge),
            "gt.Scalar": self._binary_op(relay.op.greater, operator.gt),
            "gt.Tensor": self._binary_op(relay.op.greater, operator.gt),
            "le.Scalar": self._binary_op(relay.op.less_equal, operator.le),
            "le.Tensor": self._binary_op(relay.op.less_equal, operator.le),
            "lt.Scalar": self._binary_op(relay.op.less, operator.lt),
            "lt.Tensor": self._binary_op(relay.op.less, operator.lt),
            "max.other": self._binary_op(relay.op.maximum, max),
            "min.other": self._binary_op(relay.op.minimum, min),
            "max.default": self._unary_op(relay.op.max),
            "min.default": self._unary_op(relay.op.min),
            "remainder.Tensor": self._binary_op(relay.op.floor_mod, operator.mod),
            "remainder.Scalar": self._binary_op(relay.op.floor_mod, operator.mod),
            "mul": self._binary_op(relay.op.multiply, operator.mul),
            "mul.default": self._binary_op(relay.op.multiply, operator.mul),
            "mul.Scalar": self._binary_op(relay.op.multiply, operator.mul),
            "mul.Tensor": self._binary_op(relay.op.multiply, operator.mul),
            "mul_.Tensor": self._binary_op(relay.op.multiply, operator.mul),
            "ne.Tensor": self._binary_op(relay.op.not_equal, operator.ne),
            "ne.Scalar": self._binary_op(relay.op.not_equal, operator.ne),
            "outer.default": lambda node: self.block_builder.emit(
                relay.op.outer(self.env[node.args[0]], self.env[node.args[1]])
            ),
            "pow.Scalar": self._binary_op(relay.op.power, operator.pow),
            "pow.Tensor_Scalar": self._binary_op(relay.op.power, operator.pow),
            "pow.Tensor_Tensor": self._binary_op(relay.op.power, operator.pow),
            "sub.Tensor": self._binary_op(relay.op.subtract, operator.sub),
            "__and__.Tensor": self._binary_op(relay.op.bitwise_and, operator.and_),
            "__and__.Scalar": self._binary_op(relay.op.bitwise_and, operator.and_),
            "__or__.Tensor": self._binary_op(relay.op.bitwise_or, operator.or_),
            "__or__.Scalar": self._binary_op(relay.op.bitwise_or, operator.or_),
            "__xor__.Tensor": self._binary_op(relay.op.bitwise_xor, operator.xor),
            "__xor__.Scalar": self._binary_op(relay.op.bitwise_xor, operator.xor),
            # linear algebra
            "linalg_vector_norm.default": self._norm,
            # neural network
            "_native_batch_norm_legit_functional.default": self._batch_norm_legit_functional,
            "_native_batch_norm_legit_no_training.default": self._batch_norm_legit_no_training,
            "batch_norm.default": self._batch_norm_legit_no_training,
            "adaptive_avg_pool1d.default": self._adaptive_avg_pool1d,
            "adaptive_avg_pool2d.default": self._adaptive_avg_pool2d,
            "adaptive_avg_pool3d.default": self._adaptive_avg_pool3d,
            "addmm.default": self._addmm,
            "avg_pool1d.default": self._avg_pool1d,
            "avg_pool2d.default": self._avg_pool2d,
            "avg_pool3d.default": self._avg_pool3d,
            "baddbmm.default": self._baddbmm,
            "conv_transpose1d.default": self._conv_transpose1d,
            "conv_transpose2d.input": self._conv_transpose2d,
            "conv1d.default": self._conv1d,
            "conv2d.default": self._conv2d,
            "conv3d.default": self._conv3d,
            "cross_entropy_loss.default": self._cross_entropy_default,
            "einsum.default": self._einsum,
            "embedding.default": lambda node: self._embedding_impl(
                self.env[node.args[1]], self.env[node.args[0]]
            ),
            "group_norm.default": self._group_norm,
            "instance_norm.default": self._instance_norm,
            "layer_norm.default": self._layer_norm,
            "linear.default": self._linear,
            "lstm.input": self._lstm,
            "gru.input": self._gru,
            "max_pool1d.default": self._max_pool1d,
            "max_pool2d.default": self._max_pool2d,
            "max_pool3d.default": self._max_pool3d,
            "scaled_dot_product_attention.default": self._scaled_dot_product_attention,
            "unbind.int": self._unbind,
            "upsample_bilinear2d.vec": self._upsample_bilinear2d,
            "upsample_nearest2d.vec": self._upsample_nearest2d,
            "upsample_bicubic2d.vec": self._upsample_bicubic2d,
            # statistical
            "mean.dim": self._mean,
            "prod.default": self._prod,
            "std.correction": self._std,
            "sum.default": self._sum,
            "sum.dim_IntList": self._sum,
            "var.correction": self._var,
            # search
            "argmax.default": self._argmax_argmin(relay.op.argmax),
            "argmin.default": self._argmax_argmin(relay.op.argmin),
            "where.self": self._where,
            "bucketize.Tensor": self._bucketize,
            # tensor manipulation
            "argsort.default": self._argsort,
            "broadcast_to.default": self._broadcast_to,
            "cat.default": self._cat,
            "chunk.default": self._chunk,
            "clamp.Tensor": self._clamp,
            "concat.default": self._cat,
            "copy_.default": self._copy_,
            "cumsum.default": self._cumsum,
            "cumprod.default": self._cumprod,
            "expand.default": self._expand,
            "expand_as.default": self._expand_as,
            "flatten.using_ints": self._flatten,
            "flip.default": self._flip,
            "gather.default": self._gather,
            "index.Tensor": self._index_tensor,
            "index_put_.default": self._index_put,
            "meshgrid.indexing": self._meshgrid,
            "meshgrid.default": self._meshgrid,
            "narrow.default": self._narrow,
            "permute.default": self._permute,
            "repeat.default": self._repeat,
            "roll.default": self._roll,
            "select.int": self._select,
            "slice.Tensor": self._slice,
            "slice_scatter.default": self._slice_scatter,
            "sort.default": self._sort,
            "split.Tensor": self._split,
            "split_with_sizes.default": self._split,
            "squeeze.default": self._squeeze,
            "squeeze.dim": self._squeeze,
            "stack.default": self._stack,
            "take.default": self._take,
            "tile.default": self._tile,
            "topk.default": self._topk,
            "transpose.int": self._transpose,
            "unflatten.int": self._unflatten,
            "unsqueeze.default": lambda node: self.block_builder.emit(
                relay.op.expand_dims(self.env[node.args[0]], node.args[1])
            ),
            "view.default": self._reshape,
            "reshape.default": self._reshape,
            "reshape_as.default": self._reshape_as,
            # tensor creation
            "_to_copy.default": self._to_copy,
            "arange.default": self._arange,
            "arange.start": self._arange,
            "arange.start_step": self._arange,
            "detach.default": self._detach,
            "detach_.default": self._detach,
            "contiguous.default": lambda node: self.env[node.args[0]],  # no-op
            "clone.default": lambda node: self.env[node.args[0]],
            "empty.memory_format": self._empty,
            "empty_like.default": self._empty_like,
            "eye.default": self._eye,
            "eye.m": self._eye,
            "fill.Scalar": self._fill,
            "fill_.Scalar": self._inplace_fill,
            "full.default": self._full,
            "full_like.default": self._full_like,
            "hamming_window.periodic": self._hamming_window,
            "hamming_window.periodic_alpha": self._hamming_window,
            "hamming_window.periodic_alpha_beta": self._hamming_window,
            "hamming_window.default": self._hamming_window,
            "index_select.default": self._index_select,
            "lift_fresh_copy.default": self._to_copy,
            "linspace.default": self._linspace,
            "masked_fill.Scalar": self._masked_fill,
            "masked_fill_.Scalar": self._inplace_masked_fill,
            "new_ones.default": self._new_ones,
            "new_zeros.default": self._new_zeros,
            "one_hot.default": self._one_hot,
            "ones.default": self._ones,
            "ones_like.default": lambda node: self.block_builder.emit(
                relay.op.ones_like(self.env[node.args[0]])
            ),
            "zero_.default": self._zeros_inplace,
            "zeros.default": self._zeros,
            "zeros_like.default": self._zeros_like,
            # datatype
            "to.dtype": self._to,
            "to.dtype_layout": self._to,
            "type_as.default": self._type_as,
            # other
            "getitem": self._getitem,
            "item.default": self._item,
            "sym_size.int": lambda node: relay.const(1),
            "_assert_scalar.default": lambda node: self.env[node.args[0]],
        }

    def build_relay_function_from_fx(
        self,
        func_name: str,
        nodes,  # FX/ExportedProgram 노드 리스트
        inputs_vars: dict,  # {name_hint: relay.Var}
        exported_program,  # torch.export.ExportedProgram
        func_attrs=None,
        unwrap_unit_return_tuple: bool = True,
        no_bind_return_tuple: bool = False,
    ):
        # 1) ScopeBuilder 생성 + emit 어댑터 연결
        sb = relay.ScopeBuilder()
        self.block_builder = _RelayEmitAdapter(sb)  # 기존 핸들러(emit 호출) 재사용

        # 2) placeholder 매핑을 위해 env는 그대로 활용
        #    (self.env[node] = relay.Expr)
        #    필요 시 초기화
        self.env.clear()

        output_expr = None

        # 3) (Relax의 with function/dataflow 제거) — 그냥 순서대로 노드 변환
        for node in nodes:
            if node.op == "placeholder":
                # 심볼릭 입력 무시
                if (
                    "grapharg" in node.meta
                    and node.meta["grapharg"].fake_tensor is None
                ):
                    continue
                # FX placeholder 이름과 inputs_vars 바인딩
                self.env[node] = inputs_vars[node.name]

            elif node.op == "output":
                # 출력 처리
                args = self.retrieve_args(node)
                assert len(args) == 1, "output must take single tuple"
                ret = args[0]

                # ret: (tuple of relay.Expr) 또는 relay.Tuple
                if (
                    unwrap_unit_return_tuple
                    and isinstance(ret, (tuple, list))
                    and len(ret) == 1
                ):
                    output_expr = ret[0]
                elif no_bind_return_tuple and isinstance(ret, (tuple, list)):
                    # 여러 개를 별도 바인딩 없이 바로 return 하고 싶다면 Tuple로 묶지 않고 처리
                    # 하지만 Relay Function은 단일 Expr 반환이 필요하므로 Tuple로 반환하는 게 일반적
                    output_expr = relay.Tuple(list(ret))
                else:
                    # tuple/list라면 Tuple로, 이미 relay.Tuple이면 그대로
                    if isinstance(ret, (tuple, list)):
                        output_expr = relay.Tuple(list(ret))
                    else:
                        output_expr = ret
                break

            elif node.op == "get_attr":
                # 모듈 상의 속성(텐서 파라미터 등)을 env에 바인딩
                self.env[node] = getattr(exported_program.graph_module, node.target)

            elif node.op == "call_function":
                # 디스패치: convert_map에서 handler를 찾아 호출
                func_name_key = node.target.__name__
                if func_name_key not in self.convert_map:
                    raise AssertionError(
                        f"Unsupported function types ['{func_name_key}']"
                    )
                self.env[node] = self.convert_map[func_name_key](node)

            else:
                raise ValueError(f"Unsupported op {node.op}")

        assert output_expr is not None, "no output produced"

        # 4) ScopeBuilder로 최종 반환 연결
        sb.ret(output_expr)

        # 5) Relay Function 생성
        #    params는 입력 순서를 유지해야 하므로 dict values를 리스트로
        params = list(inputs_vars.values())
        body = sb.get()

        func = relay.Function(params, body)
        if func_attrs:
            func = func.with_attr(func_attrs)

        # 함수 이름 부여(옵션)
        func = func.with_attr("global_symbol", func_name)

        return func

    def torch_to_tvm_fp32(self, t: torch.Tensor) -> tvm.nd.NDArray:
        import numpy as np

        """
        Torch Tensor -> tvm.nd.NDArray (float64를 float32로 내림, DLPack 사용, 안전 fallback 포함)
        """
        t = t.detach().cpu().contiguous()
        # # 가장 흔한 원인: float64 -> float32 강제
        t = t.to(torch.float32)
        # 매 호출마다 새로운 캡슐을 만들어야 함 (PyCapsule 재사용 금지)
        return tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(t))

    def build_binding(self, exported_program, keep_params_as_input: bool):
        # buffers + constants
        try:
            named_buffers = OrderedDict(exported_program.named_buffers())
        except AttributeError:
            named_buffers = OrderedDict(exported_program.graph_module.named_buffers())
        constants = (
            exported_program.constants
        )  # dict[str, torch.Tensor] (비텐서 있을 수 있어 주의)

        to_bind_parameters = ChainMap(named_buffers, constants)

        # parameters: keep_params_as_input == False 일 때만 바인딩(= 입력에서 제거)
        if not keep_params_as_input:
            try:
                named_params = OrderedDict(exported_program.named_parameters())
            except AttributeError:
                named_params = OrderedDict(
                    exported_program.graph_module.named_parameters()
                )
            to_bind_parameters = to_bind_parameters.new_child(named_params)

        binding = {}
        for tensor_name, tensor_value in to_bind_parameters.items():
            # constants에 비텐서가 섞였을 수 있음
            if not isinstance(tensor_value, torch.Tensor):
                continue

            # graph_signature에서 relay var 이름 찾기
            bind_name = None
            for spec in exported_program.graph_signature.input_specs:
                # PARAMETER / BUFFER / CONSTANT_TENSOR에 대해 spec.target가 Torch 쪽 이름과 일치
                if tensor_name == spec.target:
                    bind_name = spec.arg.name
                    break
            if bind_name is None:
                # 그래프에 이미 상수로 내장된 항목일 수 있음 → 스킵
                continue

            binding[bind_name] = self.torch_to_tvm_fp32(tensor_value)

        return binding

    def _to_relay_dim(self, dim, axis):
        # 0: N, 1: C, 2: H, 3: W (NCHW 가정)
        # SymInt라도 값이 있으면 int()로 캐스팅 시도
        if isinstance(dim, torch.SymInt):
            try:
                return int(dim)
            except Exception:
                # N만 동적으로 허용하고, C/H/W는 반드시 정수로 필요
                return (
                    TAny()
                    if axis == 0
                    else ValueError("C/H/W must be static for conv2d")
                )
        return int(dim)

    def create_input_vars(
        self, exported_program: torch.export.ExportedProgram
    ) -> Tuple[Dict[str, relay.Var], Dict[str, relay.Var]]:
        """Create relay input vars."""
        parameters_buffers_constants = OrderedDict()
        user_inputs = OrderedDict()
        torch_symbol_to_relay_var: Dict[str, tvm.tir.Var] = {}

        for spec in exported_program.graph_signature.input_specs:
            name_hint = spec.arg.name
            if spec.kind is torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
                torch_shape = exported_program.tensor_constants[spec.target].shape
                torch_dtype = exported_program.tensor_constants[spec.target].dtype
            elif spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                for node in exported_program.graph.find_nodes(
                    op="placeholder", target=spec.target
                ):
                    if node.name == name_hint and "tensor_meta" in node.meta:
                        torch_shape = node.meta["tensor_meta"].shape
                        torch_dtype = node.meta["tensor_meta"].dtype
                        break
            else:
                # PARAMETER or BUFFER
                torch_shape = exported_program.state_dict[spec.target].shape
                torch_dtype = exported_program.state_dict[spec.target].dtype

            dtype = self._convert_data_type(torch_dtype)
            # handle symbolic dims
            relay_shape = [self._to_relay_dim(d, i) for i, d in enumerate(torch_shape)]
            relay_var = relay.Var(name_hint, relay.TensorType(relay_shape, dtype))

            # ✅ Relay에서는 TensorType 사용
            relay_var = relay.Var(name_hint, relay.TensorType(relay_shape, dtype))

            if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                user_inputs[name_hint] = relay_var
            else:
                parameters_buffers_constants[name_hint] = relay_var

        return parameters_buffers_constants, user_inputs

    def from_exported_program(
        self,
        exported_program: torch.export.ExportedProgram,
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
    ) -> tvm.IRModule:
        """Convert a PyTorch ExportedProgram to a relay program."""
        from torch import fx  # type: ignore
        from tvm.relay.build_module import bind_params_by_name

        # Create input variables.
        parameter_buffer_constant_vars, user_input_vars = self.create_input_vars(
            exported_program
        )
        inputs_vars = user_input_vars.copy()
        inputs_vars.update(parameter_buffer_constant_vars)

        # func_name과 attrs 설정
        func_name = "main"
        func_attrs = (
            {"num_input": len(user_input_vars)} if keep_params_as_input else None
        )

        # FX 노드 리스트
        nodes: List[fx.Node] = exported_program.graph.nodes

        # 지원되지 않는 함수 타입 체크
        self._check_unsupported_func_type(nodes)

        # ✅ Relay 함수로 변환
        func = self.build_relay_function_from_fx(
            func_name=func_name,
            nodes=nodes,
            inputs_vars=inputs_vars,
            exported_program=exported_program,
            func_attrs=func_attrs,
            unwrap_unit_return_tuple=unwrap_unit_return_tuple,
            no_bind_return_tuple=no_bind_return_tuple,
        )

        # Relay IRModule 생성
        mod = tvm.IRModule.from_expr(func)

        # 바인딩 생성 (float32 통일 + 안전 변환)
        binding = self.build_binding(exported_program, keep_params_as_input)

        # 0.11 계열은 BindParams 패스가 없으므로 함수 단위 바인딩
        mod["main"] = bind_params_by_name(mod["main"], binding)
        mod = relay.transform.InferType()(mod)

        if keep_params_as_input:
            parameters = dict(exported_program.named_parameters())
            params = [tvm.runtime.from_dlpack(p.detach()) for p in parameters.values()]
            mod["main"] = mod["main"].with_attr("params", params)

        return mod


def from_exported_program(
    exported_program: torch.export.ExportedProgram,
    *,
    keep_params_as_input: bool = False,
    unwrap_unit_return_tuple: bool = False,
    no_bind_return_tuple: bool = False,
) -> tvm.IRModule:
    """Convert a PyTorch ExportedProgram to a relay program

    Parameters
    ----------
    exported_program : torch.export.ExportedProgram
        The PyTorch ExportedProgram to convert.

    keep_params_as_input : bool
        Whether to keep model parameters as input variables.

    unwrap_unit_return_tuple : bool
        A boolean flag indicating if to the return value when it is an unit tuple.
        When the return value is not a unit tuple, no unwrap will take place.

    no_bind_return_tuple : bool
        A boolean flag indicating whether to bind the return tuple as a relay var.
        If the flag is true and the return value is a tuple, it will not bind it to a var.

    Returns
    -------
    output : tvm.IRModule
        The import result IRModule, with the function "main" containing the
        translated logic.

    Examples
    --------
    Users can use the torch.export.export() to extract a torch.export.ExportedProgram
    from a PyTorch model. The following codes show how to convert a PyTorch model to
    a relay program.

    .. code-block:: python

        # Import the importer.
        import tvm
        from tvm.relay.frontend.torch import from_exported_program
        import torch
        from torch.export import export

        # Define the module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

            def forward(self, input):
                return self.linear(input)

        # Instantiate the model and create the input info dict.
        torch_model = MyModule()

        # Use torch.export.export() to convert the PyTorch model into ExportedProgram.
        example_args = (torch.rand(128, 10, dtype=torch.float32),)
        exported_program = export(torch_model, args=example_args)

        # Use the importer to import the ExportedProgram to relay.
        mod: tvm.IRModule = from_exported_program(exported_program)
    """
    # decompose into Core ATen operators
    exported_program.run_decompositions()

    return ExportedProgramImporter().from_exported_program(
        exported_program,
        keep_params_as_input,
        unwrap_unit_return_tuple,
        no_bind_return_tuple,
    )
