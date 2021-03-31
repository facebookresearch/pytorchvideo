## Accelerator Model Zoo
Accelerator model zoo provides a set of efficient models on target device with pretrain checkpoints. To learn more about how to build model, load checkpoint and deploy, please refer to [Use PytorchVideo/Accelerator Model Zoo](tutorial_accelerator_use_accelerator_model_zoo.md).

**Efficient Models for mobile CPU**
All top1/top5 accuracies are measured with 10-clip evaluation. Latency is benchmarked on Samsung S8 phone with 1s input clip length.

| model  | model buracyuilder                                                            | top1 | top5 | latency (ms) | parameters (M) | checkpoint          |
|--------|--------------------------------------------------------------------------|-------|-------|--------------|----------------|---------------------|
| X3D_XS | models.accelerator.mobile_cpu.efficient_x3d.EfficientX3d(expansion="XS") | 68.5  | 88.0  |          233 | 3.8            | [link](placeholder) |
| X3D_S  | models.accelerator.mobile_cpu.efficient_x3d.EfficientX3d(expansion="S")  | 73.0  | 90.6  |          764 | 3.8            | [link](placeholder) |