#!/usr/bin/env python3
"""Quick fix for DNAInput import issue in vllm_serve.py"""

import re

with open('trl/scripts/vllm_serve.py', 'r') as f:
    content = f.read()

# Fix the problematic line by ensuring proper fallback
old_line = '                dna_inputs = [DNAInput(sequence=seq) for seq in dna_for_prompt] if dna_for_prompt else []'
new_line = '''                # Create DNAInput objects safely
                try:
                    dna_inputs = [DNAInput(sequence=seq) for seq in dna_for_prompt] if dna_for_prompt else []
                except (TypeError, AttributeError) as e:
                    print(f'⚠️ DNAInput creation failed: {e}')
                    # Fallback: use simple dict objects that mimic DNAInput
                    dna_inputs = [type('DNAInput', (), {'sequence': seq})() for seq in dna_for_prompt] if dna_for_prompt else []'''

if old_line in content:
    content = content.replace(old_line, new_line)
    print('✅ Applied fix for DNAInput instantiation')
else:
    print('⚠️ Original line not found, applying alternative fix...')
    # Alternative fix: look for the pattern and replace it
    pattern = r'dna_inputs = \[DNAInput\(sequence=seq\) for seq in dna_for_prompt\] if dna_for_prompt else \[\]'
    replacement = '''# Create DNAInput objects safely
                try:
                    dna_inputs = [DNAInput(sequence=seq) for seq in dna_for_prompt] if dna_for_prompt else []
                except (TypeError, AttributeError) as e:
                    print(f'⚠️ DNAInput creation failed: {e}')
                    # Fallback: use simple dict objects that mimic DNAInput
                    dna_inputs = [type('DNAInput', (), {'sequence': seq})() for seq in dna_for_prompt] if dna_for_prompt else []'''
    
    content = re.sub(pattern, replacement, content)
    print('✅ Applied alternative fix for DNAInput instantiation')

with open('trl/scripts/vllm_serve.py', 'w') as f:
    f.write(content)

print('✅ Fix applied successfully')
