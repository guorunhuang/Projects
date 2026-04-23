// ─────────────────────────────────────────────
//  SUDOKU SOLVER & GENERATOR UTILITIES
// ─────────────────────────────────────────────

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function isValidPlacement(board, row, col, num, size) {
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

/**
 * Validate a user-supplied custom puzzle using backtracking.
 * Returns { valid: true, solution } if exactly one solution exists,
 * or { valid: false, reason } otherwise.
 */
export function validateCustomPuzzle(puzzle) {
  const size = puzzle.length;

  // 1. Basic structure check
  if (size !== 9) return { valid: false, reason: 'Board must be 9×9.' };
  for (const row of puzzle) {
    if (!Array.isArray(row) || row.length !== size)
      return { valid: false, reason: 'Board must be 9×9.' };
    for (const v of row) {
      if (!Number.isInteger(v) || v < 0 || v > size)
        return { valid: false, reason: `Cell values must be integers 0–${size}.` };
    }
  }

  // 2. Check that the given clues don't already conflict
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const v = puzzle[r][c];
      if (v === 0) continue;
      // Temporarily blank the cell so isValidPlacement checks the rest
      puzzle[r][c] = 0;
      const ok = isValidPlacement(puzzle, r, c, v, size);
      puzzle[r][c] = v;
      if (!ok) return { valid: false, reason: 'The given clues contain a conflict.' };
    }
  }

  // 3. Count solutions via backtracking (stop early at 2)
  const count = countSolutions(puzzle, size, 2);
  if (count === 0) return { valid: false, reason: 'This puzzle has no valid solution.' };
  if (count > 1)   return { valid: false, reason: 'This puzzle has more than one solution.' };

  // 4. Derive the unique solution
  const solution = puzzle.map(r => [...r]);
  solveSudoku(solution, size);

  return { valid: true, solution };
}

export function generatePuzzle(size, clues) {
  const board = Array.from({ length: size }, () => Array(size).fill(0));
  solveSudoku(board, size);
  const solution = board.map(r => [...r]);
  const puzzle = board.map(r => [...r]);

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
    if (countSolutions(puzzle, size, 2) === 1) {
      removed++;
    } else {
      puzzle[r][c] = backup;
    }
  }

  return { puzzle, solution };
}

// ─────────────────────────────────────────────
//  RANDOM GAME NAME GENERATOR
// ─────────────────────────────────────────────
const ADJECTIVES = [
  'Cosmic', 'Silent', 'Ancient', 'Neon', 'Frozen', 'Crimson', 'Velvet',
  'Phantom', 'Shadow', 'Crystal', 'Iron', 'Golden', 'Silver', 'Dark',
  'Swift', 'Blazing', 'Infinite', 'Hidden', 'Mystic', 'Electric',
];

const NOUNS = [
  'Grid', 'Cipher', 'Matrix', 'Nexus', 'Vortex', 'Labyrinth', 'Paradox',
  'Enigma', 'Oracle', 'Puzzle', 'Sequence', 'Pattern', 'Codex', 'Rune',
  'Signal', 'Beacon', 'Prism', 'Vertex', 'Spiral', 'Circuit',
];

export function generateGameName() {
  const adj  = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const noun = NOUNS[Math.floor(Math.random() * NOUNS.length)];
  const num  = Math.floor(Math.random() * 999) + 1;
  return `${adj} ${noun} #${num}`;
}
