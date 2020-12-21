# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry


"""
This file provides a metric query registry,
as well as a singleton interface for calling metric query in registry.
These metric queries are functions that take a model, model_input and optional kwargs argument,
and return the corresponding metric.
E.g. users might register a metric query that returns flops for the given model and input.
"""

METRIC_QUERY_REGISTRY = Registry("METRIC_QUERY")
METRIC_QUERY_REGISTRY.__doc__ = """
Registry for metric query.

The registered metric query function will be called with `func(model, input_data, **kwargs)`.
The call should return the metric defined by the metric query function.
"""


def metric_query(model, input_data, metric_query_name, **kwargs):
    """
    Based on metric_query_name,
    call corresponding query func in METRIC_QUERY_REGISTRY to finish metric query.
    For any metric query, user is supposed to provide model and its input as basic input,
    selecting query function using metric_query_name,
    and provide any other arguments required by the specific metric query function.
    """
    return METRIC_QUERY_REGISTRY.get(metric_query_name)(model, input_data, **kwargs)
