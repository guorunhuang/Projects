// ─────────────────────────────────────────────
//  CLIENT-SIDE SUDOKU UTILITIES
// ─────────────────────────────────────────────

export function isValidPlacement(board, row, col, num, size) {
  const boxSize = size === 9 ? 3 : 2;
  for (let c = 0; c < size; c++) if (board[row][c] === num) return false;
  for (let r = 0; r < size; r++) if (board[r][col] === num) return false;
  const boxRow = Math.floor(row / boxSize) * boxSize;
  const boxCol = Math.floor(col / boxSize) * boxSize;
  for (let r = boxRow; r < boxRow + boxSize; r++)
    for (let c = boxCol; c < boxCol + boxSize; c++)
      if (board[r][c] === num) return false;
  return true;
}

export function isCellInvalid(board, row, col, size) {
  const val = board[row][col];
  if (val === 0) return false;
  const boxSize = size === 9 ? 3 : 2;

  for (let c = 0; c < size; c++)
    if (c !== col && board[row][c] === val) return true;
  for (let r = 0; r < size; r++)
    if (r !== row && board[r][col] === val) return true;

  const boxRow = Math.floor(row / boxSize) * boxSize;
  const boxCol = Math.floor(col / boxSize) * boxSize;
  for (let r = boxRow; r < boxRow + boxSize; r++)
    for (let c = boxCol; c < boxCol + boxSize; c++)
      if ((r !== row || c !== col) && board[r][c] === val) return true;

  return false;
}

export function isBoardComplete(board, size) {
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) {
      if (board[r][c] === 0) return false;
      if (isCellInvalid(board, r, c, size)) return false;
    }
  return true;
}

export function findHintCell(board, puzzle, size) {
  // Find the cell with fewest valid placements among unfilled non-fixed cells
  const candidates = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (board[r][c] === 0) {
        const valid = [];
        for (let num = 1; num <= size; num++) {
          if (isValidPlacement(board, r, c, num, size)) valid.push(num);
        }
        candidates.push({ row: r, col: c, valid });
      }
    }
  }
  if (candidates.length === 0) return null;
  // Sort by fewest options, prefer naked singles
  candidates.sort((a, b) => a.valid.length - b.valid.length);
  const best = candidates[0];
  if (best.valid.length === 0) return null;
  return { row: best.row, col: best.col, value: best.valid[0] };
}
