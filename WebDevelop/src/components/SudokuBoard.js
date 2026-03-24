import React, { useCallback } from 'react';
import { useSudoku } from '../context/SudokuContext';
import SudokuCell from './SudokuCell';
import './SudokuBoard.css';

export default function SudokuBoard() {
  const { state, selectCell, inputValue, getCellState } = useSudoku();
  const { mode, userBoard } = state;

  if (!mode || userBoard.length === 0) return null;

  const size = mode === 'easy' ? 6 : 9;

  const handleKeyDown = useCallback((e, row, col) => {
    const num = parseInt(e.key);
    if (!isNaN(num) && num >= 1 && num <= size) {
      inputValue(row, col, num); return;
    }
    if (e.key === 'Backspace' || e.key === 'Delete' || e.key === '0') {
      inputValue(row, col, 0); return;
    }
    const moves = { ArrowUp:[-1,0], ArrowDown:[1,0], ArrowLeft:[0,-1], ArrowRight:[0,1] };
    if (moves[e.key]) {
      e.preventDefault();
      const [dr, dc] = moves[e.key];
      selectCell(
        Math.max(0, Math.min(size - 1, row + dr)),
        Math.max(0, Math.min(size - 1, col + dc))
      );
    }
  }, [size, inputValue, selectCell]);

  return (
    <div
      className={`sudoku-board size-${size}`}
      style={{ '--grid-size': size }}
      role="grid"
      aria-label="Sudoku board"
    >
      {Array.from({ length: size }, (_, r) =>
        Array.from({ length: size }, (_, c) => (
          <SudokuCell
            key={`${r}-${c}`}
            row={r} col={c} size={size}
            cellState={getCellState(r, c)}
            onSelect={(row, col) => selectCell(row, col)}
            onKeyDown={handleKeyDown}
          />
        ))
      )}
    </div>
  );
}
