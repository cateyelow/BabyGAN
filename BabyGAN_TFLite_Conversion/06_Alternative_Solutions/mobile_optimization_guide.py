#!/usr/bin/env python3
"""
Mobile Optimization Guide for StyleGAN
"""

# Recommended optimizations for mobile deployment

optimization_config = {
    'resolution': 256,  # Reduce from 1024
    'latent_size': 512,  # Keep same
    'num_layers': 14,   # Reduce from 18
    'channel_multiplier': 1,  # Reduce from 2
    'quantization': 'INT8',
    'pruning': True,
    'knowledge_distillation': True
}

# Memory requirements
memory_estimates = {
    'original_1024': '500+ MB',
    'mobile_256': '15-25 MB',
    'quantized_256': '5-10 MB'
}

print("Mobile optimization settings:")
for key, value in optimization_config.items():
    print(f"  {key}: {value}")
