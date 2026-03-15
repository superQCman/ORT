#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loader for architecture simulation.
"""

import yaml
from pathlib import Path


class ArchConfig:
    """Load and access hardware architecture configuration from YAML."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config_data = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    print(f"[CONFIG] Loaded configuration from {self.config_path}")
                    return data or {}
            except Exception as e:
                print(f"[CONFIG] Warning: Failed to load {self.config_path}: {e}")
                print("[CONFIG] Using default configuration")
                return self._default_config()
        else:
            print("[CONFIG] No config file specified, using defaults")
            return self._default_config()
    
    def _default_config(self):
        """Return default configuration."""
        return {
            'cache_hierarchy': {
                'line_size': 64,
                'l1': {
                    'size_bytes': 32 * 1024,
                    'associativity': 8,
                    'hit_latency': 4
                },
                'l2': {
                    'size_bytes': 512 * 1024,
                    'associativity': 8,
                    'hit_latency': 12
                },
                'l3': {
                    'size_bytes': 8 * 1024 * 1024,
                    'associativity': 16,
                    'hit_latency': 35
                },
                'memory': {
                    'latency': 200
                }
            },
            'rob': {
                'entries': 192,
                'window_size': 400
            },
            'pipeline': {
                'fetch_width': 4,
                'decode_width': 4,
                'rename_width': 4,
                'commit_width': 8,
                'issue_widths': {
                    'alu': 3,
                    'fp': 2,
                    'ls': 2
                }
            },
            'load_store_pipes': {
                'load_store_pipes': 2,
                'load_only_pipes': 10
            },
            'icache': {
                'max_fills': 8,
                'fill_latency': 40,
                'size_bytes': 4 * 1024,
                'line_size': 64,
                'fetch_width': 8
            },
            'fetch_buffer': {
                'entries': 64
            },
            'branch_prediction': {
                'simple': {
                    'misprediction_rate': 0.05,
                    'seed': 1
                },
                'tage': {
                    'num_tables': 8,
                    'table_size': 2048,
                    'tag_bits': 10,
                    'ghr_bits': 200,
                    'base_size': 4096,
                    'counter_bits': 3,
                    'usefulness_bits': 2,
                    'seed': 1
                }
            },
            'analysis': {
                'top_n': 50,
                # CDF plotting
                'cdf': {
                    'quantile_step': 0.02,     # generate quantiles every 2%
                    'tail_quantile': 0.9,      # threshold for tail zoom plots
                    'separate_figs': True,     # generate separate figures per resource
                    
                    'output': {
                        'png_dpi': 200,
                        'dir': "./result"
                    }
                }
            }
        }
    
    def get(self, path: str, default=None):
        """
        Get config value by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., 'cache_hierarchy.l1.size_bytes')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        val = self.config_data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val


# Global config instance
config = None


def init_config(config_path: str = None):
    """Initialize global config instance."""
    global config
    config = ArchConfig(config_path)
    return config


def get_config():
    """Get global config instance."""
    global config
    if config is None:
        config = ArchConfig()
    return config
