Self-Pruning Neural Network on CIFAR-10

 How to Run:
pip install torch torchvision matplotlib numpy

Run on Google Colab with T4 GPU

 How It Works(SOLUTION)
I created a custom layer called PrunableLinear that adds a learnable 
gate to every weight. The gate is a value between 0 and 1 (using sigmoid). 
During training, an L1 penalty pushes these gates toward zero, effectively 
removing unimportant weights from the network.

Total Loss = Classification Loss + λ × Sparsity Loss

 Why L1 and Not L2
L1 has a constant gradient so it keeps pushing gate values toward zero 
no matter how small they already are. L2's gradient shrinks as the value 
gets smaller so it never fully reaches zero. That's why L1 actually prunes 
weights while L2 just makes them small.

Results

 Lambda   Test Accuracy  Sparsity Level 

 1e-03     54.49%         53.77%     
5e-02     54.91%         60.12%        
5e-01     54.80%         83.70%        

As lambda increases, sparsity increases from 53% to 83% while 
accuracy stays stable — showing the network successfully identifies 
and removes unimportant weights.

 Plots
- gate_distribution.png — gate value distribution
- tradeoff_plot.png — sparsity vs accuracy trade-off
