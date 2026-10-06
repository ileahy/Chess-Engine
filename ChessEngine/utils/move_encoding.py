#mapping from nn output layer to moves and vice versa
import chess

index_to_move = []
for f in range(64):
    for t in range(64):
        index_to_move.append(chess.Move(f, t))

# pregenerated list with code once
promotions = ['a7a8q', 'a2a1q', 'a7a8r', 'a2a1r', 'a7a8b', 'a2a1b', 'a7a8n', 'a2a1n', 'b7b8q', 'b2b1q', 'b7b8r', 'b2b1r', 'b7b8b', 'b2b1b', 'b7b8n', 'b2b1n', 'c7c8q', 
              'c2c1q', 'c7c8r', 'c2c1r', 'c7c8b', 'c2c1b', 'c7c8n', 'c2c1n', 'd7d8q', 'd2d1q', 'd7d8r', 'd2d1r', 'd7d8b', 'd2d1b', 'd7d8n', 'd2d1n', 'e7e8q', 'e2e1q', 
              'e7e8r', 'e2e1r', 'e7e8b', 'e2e1b', 'e7e8n', 'e2e1n', 'f7f8q', 'f2f1q', 'f7f8r', 'f2f1r', 'f7f8b', 'f2f1b', 'f7f8n', 'f2f1n', 'g7g8q', 'g2g1q', 'g7g8r', 
              'g2g1r', 'g7g8b', 'g2g1b', 'g7g8n', 'g2g1n', 'h7h8q', 'h2h1q', 'h7h8r', 'h2h1r', 'h7h8b', 'h2h1b', 'h7h8n', 'h2h1n', 'a7b8q', 'a7b8r', 'a7b8b', 'a7b8n', 
              'b7a8q', 'b7a8r', 'b7a8b', 'b7a8n', 'b7c8q', 'b7c8r', 'b7c8b', 'b7c8n', 'c7b8q', 'c7b8r', 'c7b8b', 'c7b8n', 'c7d8q', 'c7d8r', 'c7d8b', 'c7d8n', 'd7c8q', 
              'd7c8r', 'd7c8b', 'd7c8n', 'd7e8q', 'd7e8r', 'd7e8b', 'd7e8n', 'e7d8q', 'e7d8r', 'e7d8b', 'e7d8n', 'e7f8q', 'e7f8r', 'e7f8b', 'e7f8n', 'f7e8q', 'f7e8r', 
              'f7e8b', 'f7e8n', 'f7g8q', 'f7g8r', 'f7g8b', 'f7g8n', 'g7f8q', 'g7f8r', 'g7f8b', 'g7f8n', 'g7h8q', 'g7h8r', 'g7h8b', 'g7h8n', 'h7g8q', 'h7g8r', 'h7g8b', 
              'h7g8n', 'a2b1q', 'a2b1r', 'a2b1b', 'a2b1n', 'b2a1q', 'b2a1r', 'b2a1b', 'b2a1n', 'b2c1q', 'b2c1r', 'b2c1b', 'b2c1n', 'c2b1q', 'c2b1r', 'c2b1b', 'c2b1n', 
              'c2d1q', 'c2d1r', 'c2d1b', 'c2d1n', 'd2c1q', 'd2c1r', 'd2c1b', 'd2c1n', 'd2e1q', 'd2e1r', 'd2e1b', 'd2e1n', 'e2d1q', 'e2d1r', 'e2d1b', 'e2d1n', 'e2f1q', 
              'e2f1r', 'e2f1b', 'e2f1n', 'f2e1q', 'f2e1r', 'f2e1b', 'f2e1n', 'f2g1q', 'f2g1r', 'f2g1b', 'f2g1n', 'g2f1q', 'g2f1r', 'g2f1b', 'g2f1n', 'g2h1q', 'g2h1r', 
              'g2h1b', 'g2h1n', 'h2g1q', 'h2g1r', 'h2g1b', 'h2g1n']
for p in promotions:
    index_to_move.append(chess.Move.from_uci(p))

move_to_index = {
    move: idx
    for idx, move in enumerate(index_to_move)
}
