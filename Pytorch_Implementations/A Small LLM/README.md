# Implementations of Mini LLM (GPT-2 Style)

- v0: Implementation from scratch without any transformer function using pytorch
- v1: Using pytorch's scaled dot production attention for efficient transformer blocks
- v1_cuda: A ~120M LLM model which can be trained using cuda only
  - Uses multi-gpu setup with `torchrun`