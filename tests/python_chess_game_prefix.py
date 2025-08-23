from evaluator import *

DESCRIPTION = "Test if the model can correctly call a python API for a moderately popular python library."

TAGS = ['code', 'python']

question = """
In python-chess I have a game = chess.pgn.read_game(pgn).

Write a function print_all_prefixes(game) that prints the PGN notation for each prefix of the game, where each prefix contains the moves from the start up to move N, for N from 1 to the total number of moves. Print each prefix on a separate line.

Call your function print_all_prefixes(game).

"""

test_case = """import io
import chess.pgn
print_all_prefixes(chess.pgn.read_game(io.StringIO('1. Nf3 Nf6 2. c4 g6 3. Nc3 Bg7 4. d4 O-O 5. Bf4 d5 6. Qb3 dxc4 7. Qxc4 c6 8. e4 Nbd7 9. Rd1 Nb6 10. Qc5 Bg4 11. Bg5 Na4 12. Qa3 Nxc3 13. bxc3 Nxe4 14. Bxe7 Qb6 15. Bc4 Nxc3')))"""

def check(txt):
    lines = txt.strip().split('\n')
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Should have multiple prefixes (at least 5 for reasonable output)
    if len(lines) < 5:
        return False, f"Expected at least 5 prefixes, got {len(lines)}"
    
    # Check that we have the expected starting move pattern
    if not lines[0].startswith('1. Nf3'):
        return False, "First prefix should start with '1. Nf3'"
        
    # Check that each line is a valid prefix (each should be contained in the next)
    # and that they're increasing in length
    for i in range(len(lines)-1):
        current = lines[i].strip()
        next_line = lines[i+1].strip()
        
        # Each prefix should be shorter than the next
        if len(current) >= len(next_line):
            return False, f"Prefix {i+1} should be longer than prefix {i}"
        
        # Next line should start with current line (prefix property)
        if not next_line.startswith(current.split(' *')[0]):  # Handle potential variations
            # More flexible check - verify the move sequence is extending
            current_moves = current.replace('.', ' ').split()
            next_moves = next_line.replace('.', ' ').split()
            
            # Should have more moves in next line
            if len(next_moves) <= len(current_moves):
                return False, f"Move count not increasing: {len(current_moves)} to {len(next_moves)}"
    
    # Check that we have reasonable number of prefixes (15 moves = 30 half-moves, expect ~15 prefixes)
    if len(lines) < 10:
        return False, f"Expected more prefixes for the test game, got {len(lines)}"
        
    return True, ""
    
    

TestPyChessPrefix = question >> LLMRun() >> ExtractCode() >> PythonRun(test_case) >> PyFunc(check)

if __name__ == "__main__":
    print(run_test(TestPyChessPrefix))
