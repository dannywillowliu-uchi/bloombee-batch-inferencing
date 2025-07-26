#!/usr/bin/env python3
"""
True Batch Inference Test for BloomBee

This script tests true batch inference functionality with BloomBee.
It processes multiple prompts together in a single batch through one session.
"""

import asyncio
import time
import torch
from bloombee.utils import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

class BatchInferenceTest:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    async def setup_model(self, initial_peers):
        """Initialize the distributed model and tokenizer"""
        print("Setting up distributed model...")
        self.model = AutoDistributedModelForCausalLM.from_pretrained(
            "huggyllama/llama-7b",
            torch_dtype=torch.float32,
            device_map="auto",
            initial_peers=initial_peers,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model setup complete!")
        
    async def test_batch_inference(self, batch_size):
        """Test batch inference for a given batch size"""
        print(f"\nTesting batch size: {batch_size}")
        
        # Create identical prompts for batch testing
        prompts = ["Hello, how are you today?"] * batch_size
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=32
        )
        prompt_length = inputs['input_ids'].shape[1]
        print(f"[DEBUG] Client: prompt_length={prompt_length}, max_new_tokens=10")
        
        # Move to same device as model
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Processing batch of {batch_size} prompts...")
        
        start_time = time.time()
        
        try:
            # Use generate_batch for batch processing
            with torch.no_grad():
                # Ensure max_new_tokens is per sequence, not total for batch
                max_new_tokens_per_sequence = 5
                print(f"[DEBUG] Generating with max_new_tokens={max_new_tokens_per_sequence} per sequence, batch_size={batch_size}")
                print(f"[DEBUG] Total tokens to generate: {max_new_tokens_per_sequence * batch_size}")
                
                # Try using regular generate() for batch size 1, generate_batch() for others
                if batch_size == 1:
                    print("[DEBUG] Using regular generate() for batch size 1")
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,  # Only generate 1 token for batch size 1
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    print("[DEBUG] Using generate_batch() for batch size > 1")
                    outputs = self.model.generate_batch(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens_per_sequence * batch_size,  # Scale by batch size
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        batch_size=batch_size
                    )
                
                print(f"[DEBUG] Output shape: {outputs.shape}")
                print(f"[DEBUG] Input shape: {input_ids.shape}")
                print(f"[DEBUG] Expected tokens per sequence: {outputs.shape[1] - input_ids.shape[1]}")
                
                # Show actual token counts for first sequence
                if batch_size == 1:
                    input_tokens = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    output_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"[DEBUG] Input tokens: '{input_tokens}'")
                    print(f"[DEBUG] Output tokens: '{output_tokens}'")
                    print(f"[DEBUG] Generated text: '{output_tokens[len(input_tokens):]}'")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            total_tokens = outputs.shape[0] * outputs.shape[1]
            tokens_per_second = total_tokens / total_time
            time_per_sequence = total_time / batch_size
            
            print(f"✓ Batch {batch_size} completed successfully!")
            print(f"  Time: {total_time:.4f}s")
            print(f"  Tokens/s: {tokens_per_second:.2f}")
            print(f"  Time/seq: {time_per_sequence:.4f}s")
            
            return {
                'batch_size': batch_size,
                'time': total_time,
                'tokens_per_second': tokens_per_second,
                'time_per_sequence': time_per_sequence,
                'success': True,
                'outputs': outputs
            }
            
        except Exception as e:
            print(f"✗ Error in batch generation: {e}")
            return {
                'batch_size': batch_size,
                'time': None,
                'tokens_per_second': None,
                'time_per_sequence': None,
                'success': False,
                'error': str(e)
            }
    
    async def run_performance_test(self, initial_peers):
        """Run performance test across different batch sizes"""
        print("=" * 60)
        print("BATCH INFERENCE PERFORMANCE TEST")
        print("=" * 60)
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4,8,16]
        results = []
        
        for batch_size in batch_sizes:
            result = await self.test_batch_inference(batch_size)
            results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        # Print summary table
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'Batch':<6} {'Time(s)':<10} {'Tokens/s':<12} {'Time/Seq(s)':<12} {'Status':<10}")
        print("-" * 60)
        
        for result in results:
            if result['success']:
                print(f"{result['batch_size']:<6} {result['time']:<10.4f} {result['tokens_per_second']:<12.2f} {result['time_per_sequence']:<12.4f} {'✓':<10}")
            else:
                print(f"{result['batch_size']:<6} {'ERROR':<10} {'-':<12} {'-':<12} {'✗':<10}")
                print(f"  Error: {result['error']}")
        
        print("=" * 60)
        
        # Print generated outputs for each batch
        print("\n" + "=" * 60)
        print("GENERATED OUTPUTS")
        print("=" * 60)
        
        for result in results:
            if result['success'] and 'outputs' in result:
                print(f"\nBatch {result['batch_size']}:")
                print("-" * 40)
                for i, output in enumerate(result['outputs']):
                    # Decode the output tokens back to text
                    decoded_output = self.tokenizer.decode(output, skip_special_tokens=True)
                    print(f"  Output {i+1}: {decoded_output}")
        
        print("=" * 60)
        
        # Calculate scaling efficiency
        print("\nSCALING ANALYSIS:")
        successful_results = [r for r in results if r['success']]
        if len(successful_results) > 1:
            baseline = successful_results[0]
            print(f"\nBaseline (Batch 1): {baseline['tokens_per_second']:.2f} tokens/sec")
            print(f"{'Batch':<6} {'Speedup':<10} {'Efficiency':<12} {'Theoretical':<12} {'Status':<10}")
            print("-" * 60)
            
            for result in successful_results[1:]:
                if result['success'] and baseline['success']:
                    speedup = baseline['time_per_sequence'] / result['time_per_sequence']
                    efficiency = speedup / (result['batch_size'] / baseline['batch_size'])
                    theoretical_speedup = result['batch_size'] / baseline['batch_size']
                    
                    # Determine if scaling is good, acceptable, or poor
                    if efficiency >= 0.8:
                        status = "Excellent"
                    elif efficiency >= 0.6:
                        status = "Good"
                    elif efficiency >= 0.4:
                        status = "Acceptable"
                    else:
                        status = "Poor"
                    
                    print(f"{result['batch_size']:<6} {speedup:<10.2f} {efficiency:<12.2f} {theoretical_speedup:<12.2f} {status:<10}")
            
            # Overall scaling analysis
            print(f"\n{'='*60}")
            print("BATCHING PERFORMANCE ANALYSIS:")
            print(f"{'='*60}")
            
            # Find best performing batch
            best_batch = max(successful_results, key=lambda x: x['tokens_per_second'])
            print(f"Best Performance: Batch {best_batch['batch_size']} at {best_batch['tokens_per_second']:.2f} tokens/sec")
            
            # Calculate average efficiency
            efficiencies = []
            for result in successful_results[1:]:
                if result['success'] and baseline['success']:
                    speedup = baseline['time_per_sequence'] / result['time_per_sequence']
                    efficiency = speedup / (result['batch_size'] / baseline['batch_size'])
                    efficiencies.append(efficiency)
            
            if efficiencies:
                avg_efficiency = sum(efficiencies) / len(efficiencies)
                print(f"Average Efficiency: {avg_efficiency:.2f}x (theoretical max: 1.0x)")
                
                if avg_efficiency >= 0.8:
                    print("Excellent batching performance - near-optimal GPU utilization")
                elif avg_efficiency >= 0.6:
                    print("Good batching performance - efficient resource usage")
                elif avg_efficiency >= 0.4:
                    print("Acceptable batching performance - some overhead present")
                else:
                    print("Poor batching performance - significant overhead or bottlenecks")
            
            # Scaling pattern analysis
            print(f"\nSCALING PATTERN ANALYSIS:")
            if len(successful_results) >= 3:
                # Check if performance plateaus or continues improving
                tokens_per_sec = [r['tokens_per_second'] for r in successful_results]
                batch_sizes = [r['batch_size'] for r in successful_results]
                
                # Find if there's a clear peak
                max_tokens_idx = tokens_per_sec.index(max(tokens_per_sec))
                optimal_batch = batch_sizes[max_tokens_idx]
                
                if optimal_batch == batch_sizes[-1]:
                    print(f"Performance still improving at batch size {optimal_batch}")
                    print(f"Consider testing larger batch sizes if memory allows")
                else:
                    print(f"Performance peaks at batch size {optimal_batch}")
                    print(f"Optimal batch size for your setup: {optimal_batch}")
                
                # Check for diminishing returns
                if len(tokens_per_sec) >= 4:
                    recent_gains = []
                    for i in range(1, len(tokens_per_sec)):
                        gain = tokens_per_sec[i] - tokens_per_sec[i-1]
                        recent_gains.append(gain)
                    
                    if recent_gains and recent_gains[-1] < recent_gains[0] * 0.5:
                        print(f"Diminishing returns detected - gains are decreasing")
                    else:
                        print(f"Consistent performance gains across batch sizes")

async def main():
    # Initialize test
    test = BatchInferenceTest()
    
    # Your server peer addresses
    initial_peers = [
        "/ip4/127.0.0.1/tcp/31338/p2p/12D3KooWCUo23gL3PZHfTcYhAkBFy5KB76UE4pca1kq8P8Yq9AfA"
    ]
    
    try:
        # Setup model
        await test.setup_model(initial_peers)
        
        # Run performance test
        await test.run_performance_test(initial_peers)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 