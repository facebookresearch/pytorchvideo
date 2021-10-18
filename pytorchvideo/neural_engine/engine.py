# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
from collections import OrderedDict
from typing import List, Union

import networkx as nx
from pytorchvideo.neural_engine import HookBase


class NeuralEngine:
    """
    NeuralEngine takes a list of hooks and executes them in their topological order. The
    topological order of the hooks is determined by their required inputs and outputs.
    """

    def __init__(self, hooks: List[HookBase]) -> None:
        self.hooks = hooks
        self.execution_order_func = NeuralEngine.topological_sort

    def get_execution_order(self, status):
        self.execution_order_func(status, self.hooks)

    def set_execution_order_func(self, func):
        self.execution_order_func = func

    @staticmethod
    def topological_sort(status, hooks):
        # Get DAG
        graph = nx.DiGraph()
        edges = []
        pending_outputs = []
        output_to_hook = {}
        for hook in hooks:
            for pair in itertools.product(hook.get_inputs(), hook.get_outputs()):
                edges.append(pair)
            for output in hook.get_outputs():
                assert output not in pending_outputs
                output_to_hook[output] = hook
                pending_outputs.append(output)
        graph.add_edges_from(edges)
        for _current in nx.topological_sort(graph):
            if _current in pending_outputs:
                _hook = output_to_hook[_current]
                yield _hook
                for _hook_out in _hook.get_outputs():
                    pending_outputs.remove(_hook_out)
            else:
                assert _current in status
        assert len(pending_outputs) == 0

    def run(self, status: OrderedDict):
        for hook in self.get_execution_order(status):
            status.update(hook.run(status))
        return status

    def __enter__(
        self,
    ):
        return self

    def __exit__(
        self,
        type,
        value,
        traceback,
    ):
        pass

    def __call__(
        self,
        status: Union[OrderedDict, str],
    ):
        # If not specified, the default input should be the path to video.
        if type(status) == str:
            status = {"path": status}
        return self.run(status)
