#!/usr/bin/env python3

# Test assembly emulator to debug the infinite loop issue
# This reproduces the square numbers test that's failing

import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.program_in_new_assembly import AssemblyEmulator

def test_assembly_squares():
    """Test a simple loop that should generate squares"""
    # Test the user's problematic code first
    assembly_code = """
SET R1 1          ; R1 = i  (starts at 1)
SET R4 0          ; R4 = memory index (starts at 0)

loop:
    MUL R2 R1 R1  ; R2 = i * i
    STORE R2 R4   ; mem[R4] = R2   (store square)
    INC  R4       ; next memory address
    INC  R1       ; next i
    LT   R1 21    ; flag = (i < 21)   ; stop after i = 20
    JT   loop     ; jump back if true

HCF              ; halt – the program has finished
"""

    print("Testing assembly code:")
    print(assembly_code)
    print("\n" + "="*50 + "\n")

    try:
        emulator = AssemblyEmulator(assembly_code)
        emulator.run()
        
        print("Final registers:")
        for reg, val in emulator.registers.items():
            print(f"  {reg}: {val}")
        
        print(f"\nFlag: {emulator.flag}")
        print(f"Instruction pointer: {emulator.instruction_pointer}")
        
        print("\nFirst 20 memory locations:")
        for i in range(20):
            print(f"  memory[{i}]: {emulator.memory[i]}")
        
        # Check if we got the expected squares
        expected = [i*i for i in range(1, 21)]
        actual = emulator.memory[:20]
        
        print(f"\nExpected: {expected}")
        print(f"Actual:   {actual}")
        print(f"Match: {expected == actual}")
        
        return expected == actual
        
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_assembly_squares()