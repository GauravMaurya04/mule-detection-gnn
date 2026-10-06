"""Account-task GNNs in plain PyTorch: no pyg-lib or compiled extensions, so the same code runs
on the laptop GPU, on CPU and in CI. Message passing is index_add_ over edge lists."""
