#!/usr/bin/env python3
"""
Comprehensive fix for multiprocessing pickle issues with DNAInput
"""

import re

with open('trl/scripts/vllm_serve.py', 'r') as f:
    content = f.read()

# Fix 1: Move DNAInput definition to module level (outside try/except)
old_import_section = '''# Import DNA processing components with proper error handling
DNA_LLM_AVAILABLE = False
DLProcessor = None
DNAInput = None
CHAT_TEMPLATE = None

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoConfig,
    )
    
    from bioreason.utils.dna_utils import DNAInput
    from bioreason.models.dl.processing_dl import DLProcessor
    from bioreason.models.dl.chat_template_dl import CHAT_TEMPLATE
    
    # Try to import Evo2 components
    try:
        from bioreason.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer
        register_evo2_tokenizer()
        EVO2_AVAILABLE = True
    except ImportError:
        EVO2_AVAILABLE = False
        print("Warning: Evo2 tokenizer not available")
    
    DNA_LLM_AVAILABLE = True
    print("✅ DNA-LLM components loaded successfully")
    
except ImportError as e:
    print(f"Warning: DNA-LLM components not available: {e}")
    print("DNA functionality will be disabled.")
    
    # Create minimal fallback classes
    class DNAInput:
        def __init__(self, sequence: str):
            self.sequence = sequence
    
    class DLProcessor:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            raise RuntimeError("DNA functionality not available - bioreason package not installed")
    
    CHAT_TEMPLATE = ""
    EVO2_AVAILABLE = False'''

new_import_section = '''# Define fallback classes at module level for proper pickling
class FallbackDNAInput:
    """Fallback DNAInput class that can be pickled across processes."""
    def __init__(self, sequence: str):
        self.sequence = sequence

class FallbackDLProcessor:
    """Fallback DLProcessor class."""
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        raise RuntimeError("DNA functionality not available - bioreason package not installed")

# Import DNA processing components with proper error handling
DNA_LLM_AVAILABLE = False
DLProcessor = FallbackDLProcessor
DNAInput = FallbackDNAInput  # Use fallback by default
CHAT_TEMPLATE = ""
EVO2_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoConfig,
    )
    
    from bioreason.utils.dna_utils import DNAInput as RealDNAInput
    from bioreason.models.dl.processing_dl import DLProcessor as RealDLProcessor
    from bioreason.models.dl.chat_template_dl import CHAT_TEMPLATE as RealCHAT_TEMPLATE
    
    # Only override if real components are available
    DNAInput = RealDNAInput
    DLProcessor = RealDLProcessor
    CHAT_TEMPLATE = RealCHAT_TEMPLATE
    
    # Try to import Evo2 components
    try:
        from bioreason.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer
        register_evo2_tokenizer()
        EVO2_AVAILABLE = True
    except ImportError:
        EVO2_AVAILABLE = False
        print("Warning: Evo2 tokenizer not available")
    
    DNA_LLM_AVAILABLE = True
    print("✅ DNA-LLM components loaded successfully")
    
except ImportError as e:
    print(f"Warning: DNA-LLM components not available: {e}")
    print("DNA functionality will be disabled - using fallback classes")'''

# Apply the import section fix
content = content.replace(old_import_section, new_import_section)

# Fix 2: Simplify the DNAInput creation logic
old_creation_logic = '''                # Create DNAInput objects safely
                try:
                    dna_inputs = [DNAInput(sequence=seq) for seq in dna_for_prompt] if dna_for_prompt else []
                except (TypeError, AttributeError) as e:
                    print(f'⚠️ DNAInput creation failed: {e}')
                    # Fallback: use simple dict objects that mimic DNAInput
                    dna_inputs = [type('DNAInput', (), {'sequence': seq})() for seq in dna_for_prompt] if dna_for_prompt else []'''

new_creation_logic = '''                # Create DNAInput objects - now safe for multiprocessing
                dna_inputs = [DNAInput(sequence=seq) for seq in dna_for_prompt] if dna_for_prompt else []'''

content = content.replace(old_creation_logic, new_creation_logic)

with open('trl/scripts/vllm_serve.py', 'w') as f:
    f.write(content)

print('✅ Applied comprehensive multiprocessing fix')
print('   - Moved DNAInput to module level for proper pickling')
print('   - Simplified creation logic')
print('   - Fixed fallback class handling')
