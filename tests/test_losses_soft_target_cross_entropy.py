# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import unittest

import torch
import torch.nn.functional as F
from pytorchvideo.losses.soft_target_cross_entropy import SoftTargetCrossEntropyLoss


class TestSoftTargetCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.set_rng_state(torch.manual_seed(42).get_state())

    def test_soft_target_cross_entropy_loss(self):
        """
        Test the soft target cross entropy loss.
        """
        for batch_size, num_class, use_1D_target in itertools.product(
            (1, 8), (2, 10), (True, False)
        ):
            loss = SoftTargetCrossEntropyLoss()

            # Test forwarding.
            for (
                input_tensor,
                target_tensor,
            ) in TestSoftTargetCrossEntropyLoss._get_inputs(
                batch_size=batch_size, num_class=num_class, use_1D_target=use_1D_target
            ):
                output_tensor = loss(input_tensor, target_tensor)
                output_shape = output_tensor.shape

                self.assertEqual(
                    output_shape,
                    torch.Size([]),
                    "Output shape {} is different from expected.".format(output_shape),
                )

                # If target is normalized, output_tensor must match direct eval
                if target_tensor.ndim == 1 or all(target_tensor.sum(dim=-1) == 1):

                    _target_tensor = target_tensor
                    if target_tensor.ndim == 1:
                        _target_tensor = torch.nn.functional.one_hot(
                            target_tensor, num_class
                        )

                    _output_tensor = torch.sum(
                        -_target_tensor * F.log_softmax(input_tensor, dim=-1), dim=-1
                    ).mean()

                    self.assertTrue(abs(_output_tensor - output_tensor) < 1e-6)

    @staticmethod
    def _get_inputs(
        batch_size: int = 16, num_class: int = 400, use_1D_target: bool = True
    ) -> torch.tensor:
        """
        Provide different tensors as test cases.

        Yield:
            (torch.tensor): tensor as test case input.
        """
        # Prepare random tensor as test cases.
        if use_1D_target:
            target_shape = (batch_size,)
        else:
            target_shape = (batch_size, num_class)
        input_shape = (batch_size, num_class)

        yield torch.rand(input_shape), torch.randint(num_class, target_shape)
