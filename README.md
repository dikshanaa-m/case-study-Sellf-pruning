pip install torch torchvision matplotlib numpy

T4 GPU on Google colab
Created a custom layer called PrunableLinear that adds a learnable gate to every weight. The gate is a value between 0 and 1 (using sigmoid).
During training, an L1 penalty pushes these gates toward zero, effectively removing unimportant weights from the network.

Total Loss = Classification Loss + λ × Sparsity Loss

L1 has a constant gradient so it keeps pushing gate values toward zero no matter how small they already are. L2's gradient shrinks as the value gets smaller so it never fully reaches zero. That's why L1 actually prunes weights while L2 just makes them small.

Training with λ (lambda) = 0.001     Training with λ (lambda) = 0.05     Training with λ (lambda) = 0.5
Final Test Accuracy : 54.49%         Final Test Accuracy : 54.91%        Final Test Accuracy : 54.80%
Sparsity Level      : 53.77%         Sparsity Level      : 60.12%        Sparsity Level      : 83.70%


Lambda        Test Accuracy  Sparsity Level
  ------------------------------------------------
  1e-03                54.49%          53.77%
  5e-02                54.91%          60.12%
  5e-01                54.80%          83.70%
====================================================


