# Bloombee Optimized for Batch Inference

Optimized version of Bloombee with memory management improvements for batch inference.

## Changes Made

### Server Files Modified:
- `bloom_venv/lib/python3.12/site-packages/bloombee/server/backend.py`
  - Added 256MB memory limit for batch processing
  - Dynamic chunking based on batch size
  - Max length capped at 256 tokens
  - Debug logging added

- `bloom_venv/lib/python3.12/site-packages/bloombee/server/handler.py`
  - Request-level memory capping
  - Debug logging for max_length values

- `bloom_venv/lib/python3.12/site-packages/bloombee/server/memory_cache.py`
  - Allocation size monitoring

## Test Scripts

- `bloombee_batch_inference_test.py` - Tests batch sizes 1, 2, 4, 8, 16
- `test_single_prompts.py` - Tests individual prompts (200 tokens)

## Usage

1. Install dependencies: `pip install bloombee transformers torch`
2. Start server: `python -m bloombee.server.run_server --model huggyllama/llama-7b`
3. Run tests: `python bloombee_batch_inference_test.py`

## Performance

- Fixed 4GB OOM crashes
- Improved batch processing speed
- Fixed 1-token generation limit in batch mode
- Reduced memory allocations from 4GB+ to 256MB-1.5GB

## License

Same as original Bloombee. 