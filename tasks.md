# Feature and Task List

## Dynamic Batch Sizing for Training

### Description

Implement a dynamic batch sizing strategy to optimize GPU memory usage and training efficiency. Currently, the training script uses a fixed batch size (`--per_device_train_batch_size`) for all batches, regardless of the sequence length of the samples within them. When using `--group_by_length`, batches are created with samples of similar lengths, but the batch size remains constant. This forces the user to set a batch size small enough to accommodate the longest samples, leading to GPU underutilization for batches containing shorter samples.

### Proposed Solution

The goal is to create batches that are sized based on the total number of audio frames or tokens, rather than a fixed number of samples. This would allow for:
-   Fewer samples per batch for very long audio clips.
-   More samples per batch for shorter audio clips.

This approach ensures that each batch has a roughly equivalent computational load, maximizing GPU throughput and minimizing wasted memory from excessive padding.

### Implementation Steps

1.  **Create a Custom `BatchSampler`:**
    -   Develop a new class that inherits from `torch.utils.data.Sampler`.
    -   This sampler will need to access the lengths of all audio samples in the dataset.
    -   Instead of yielding batches of a fixed number of indices, it should dynamically create batches.

2.  **Batch Creation Logic:**
    -   Define a `max_tokens_per_batch` or `max_frames_per_batch` threshold.
    -   The sampler should iterate through the dataset (ideally sorted by length) and add samples to a batch until the cumulative length/token count reaches the defined threshold.
    -   Once the threshold is met, the sampler yields the batch of indices and starts a new one.

3.  **Integration with `DataLoader`:**
    -   Modify the training script (`run_parler_tts_training.py`) to use this new custom `BatchSampler`.
    -   The `DataLoader` should be initialized with `batch_sampler=YourCustomBatchSampler(...)` instead of `batch_size`, `shuffle`, `sampler`, and `drop_last`.
    -   The `--per_device_train_batch_size` argument would be replaced by a new argument like `--max_tokens_per_batch`.

### Benefits

-   **Improved GPU Utilization:** Keeps the GPU consistently busy with optimally sized batches.
-   **Faster Training:** Potentially speeds up the overall training time by processing more samples in the same amount of time.
-   **Reduced OOM Errors:** More robust against out-of-memory errors when dealing with datasets that have a wide variance in sample lengths.
