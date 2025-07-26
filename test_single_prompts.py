#!/usr/bin/env python3

import torch
import time
import asyncio
from bloombee import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

class SinglePromptTester:
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    async def setup_model(self, initial_peers):
        """Setup the distributed model with the specified peers."""
        print("Setting up distributed model...")
        self.model = AutoDistributedModelForCausalLM.from_pretrained(
            "huggyllama/llama-7b",
            torch_dtype=torch.float32,
            device_map="auto",
            initial_peers=initial_peers,
        )
        # Load tokenizer separately
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ Distributed model initialized successfully!")
    
    def test_prompt(self, prompt: str, max_new_tokens: int = 10):
        """Test a single prompt and return the generated output."""
        print(f"\n{'='*60}")
        print(f"TESTING PROMPT: {repr(prompt)}")
        print(f"{'='*60}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Move to same device as model
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        
        print(f"Input tokens: {input_ids.shape[1]}")
        print(f"Input text: {repr(prompt)}")
        
        # Generate with torch.no_grad() for efficiency
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.time()
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate metrics
        total_time = end_time - start_time
        new_tokens = outputs.shape[1] - input_ids.shape[1]
        tokens_per_second = new_tokens / total_time if total_time > 0 else 0
        
        print(f"\nRESULTS:")
        print(f"  Generation time: {total_time:.4f}s")
        print(f"  New tokens generated: {new_tokens}")
        print(f"  Tokens per second: {tokens_per_second:.2f}")
        print(f"  Full output: {repr(generated_text)}")
        
        return {
            'prompt': prompt,
            'input_tokens': input_ids.shape[1],
            'new_tokens': new_tokens,
            'time': total_time,
            'tokens_per_second': tokens_per_second,
            'full_output': generated_text
        }
    
    async def run_tests(self):
        """Run tests for both prompts."""
        print("Starting single prompt tests...")
        
        # Test prompts
        prompts = [
            "Hello",
            "You've made significant progress in incorporating the KV cache into your LLaMA MHA generation function! The main issues in"
        ]
        
        results = []
        
        for prompt in prompts:
            result = self.test_prompt(prompt, max_new_tokens=200)
            results.append(result)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        for i, result in enumerate(results, 1):
            print(f"\nTest {i}:")
            print(f"  Prompt: {repr(result['prompt'])}")
            print(f"  Input tokens: {result['input_tokens']}")
            print(f"  Generated tokens: {result['new_tokens']}")
            print(f"  Speed: {result['tokens_per_second']:.2f} tokens/sec")
            print(f"  Output: {repr(result['full_output'])}")
        
        return results

async def main():
    # Server peer to connect to
    initial_peers = ["/ip4/127.0.0.1/tcp/31338/p2p/12D3KooWDRMZA7u7ZwswUxsCmkN3AY9LsnW4rRKnFzRux94vSqUM"]
    
    tester = SinglePromptTester()
    await tester.setup_model(initial_peers)
    results = await tester.run_tests()

if __name__ == "__main__":
    asyncio.run(main()) 