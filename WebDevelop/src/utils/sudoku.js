// Sudoku utility functions

// Check if placing num at (row, col) is valid
export function isValidPlacement(board, row, col, num, size) {
  const boxSize = size === 9 ? 3 : 2;

  // Check row
  for (let c = 0; c < size; c++) {
    if (board[row][c] === num) return false;
  }

  // Check col
  for (let r = 0; r < size; r++) {
    if (board[r][col] === num) return false;
  }

  // Check box
  const boxRow = Math.floor(row / boxSize) * boxSize;
  const boxCol = Math.floor(col / boxSize) * boxSize;
  for (let r = boxRow; r < boxRow + boxSize; r++) {
    for (let c = boxCol; c < boxCol + boxSize; c++) {
      if (board[r][c] === num) return false;
    }
  }

  return true;
}

// Solve board using backtracking, returns true if solved
function solveSudoku(board, size) {
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      if (board[row][col] === 0) {
        const nums = shuffle(Array.from({ length: size }, (_, i) => i + 1));
        for (const num of nums) {
          if (isValidPlacement(board, row, col, num, size)) {
            board[row][col] = num;
            if (solveSudoku(board, size)) return true;
            board[row][col] = 0;
          }
        }
        return false;
      }
    }
  }
  return true;
}

// Count solutions (up to 2) for backtracking uniqueness check
function countSolutions(board, size, limit = 2) {
  let count = 0;
  function solve(b) {
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        if (b[row][col] === 0) {
          for (let num = 1; num <= size; num++) {
            if (isValidPlacement(b, row, col, num, size)) {
              b[row][col] = num;
              solve(b);
              if (count >= limit) return;
              b[row][col] = 0;
            }
          }
          return;
        }
      }
    }
    count++;
  }
  solve(board.map(r => [...r]));
  return count;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Generate a full solved board
function generateSolvedBoard(size) {
  const board = Array.from({ length: size }, () => Array(size).fill(0));
  solveSudoku(board, size);
  return board;
}

// Generate puzzle by removing cells, ensuring unique solution (backtracking)
export function generatePuzzle(size, clues) {
  const solved = generateSolvedBoard(size);
  const puzzle = solved.map(r => [...r]);

  const positions = [];
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++)
      positions.push([r, c]);

  shuffle(positions);

  let removed = 0;
  const target = size * size - clues;

  for (const [r, c] of positions) {
    if (removed >= target) break;
    const backup = puzzle[r][c];
    puzzle[r][c] = 0;

    // Check uniqueness
    const solutions = countSolutions(puzzle, size, 2);
    if (solutions === 1) {
      removed++;
    } else {
      puzzle[r][c] = backup;
    }
  }

  return { puzzle, solution: solved };
}

// Check if a cell's value violates rules given the current board state
export function isCellInvalid(board, row, col, size) {
  const val = board[row][col];
  if (val === 0) return false;

  const boxSize = size === 9 ? 3 : 2;

  // Check row
  for (let c = 0; c < size; c++) {
    if (c !== col && board[row][c] === val) return true;
  }

  // Check col
  for (let r = 0; r < size; r++) {
    if (r !== row && board[r][col] === val) return true;
  }

  // Check box
  const boxRow = Math.floor(row / boxSize) * boxSize;
  const boxCol = Math.floor(col / boxSize) * boxSize;
  for (let r = boxRow; r < boxRow + boxSize; r++) {
    for (let c = boxCol; c < boxCol + boxSize; c++) {
      if ((r !== row || c !== col) && board[r][c] === val) return true;
    }
  }

  return false;
}

// Check if board is completely and validly filled
export function isBoardComplete(board, size) {
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (board[r][c] === 0) return false;
      if (isCellInvalid(board, r, c, size)) return false;
    }
  }
  return true;
}

// Find a cell with only one valid answer (for hint system)
export function findHintCell(board, size) {
  const hints = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (board[r][c] === 0) {
        const valid = [];
        for (let num = 1; num <= size; num++) {
          if (isValidPlacement(board, r, c, num, size)) {
            valid.push(num);
          }
        }
        if (valid.length === 1) {
          hints.push({ row: r, col: c, value: valid[0] });
        }
      }
    }
  }
  if (hints.length === 0) return null;
  return hints[Math.floor(Math.random() * hints.length)];
}
