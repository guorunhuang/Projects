import React, { useCallback } from 'react';
import { useSudoku } from '../sudoku/SudokuContext.jsx';
import SudokuCell from './SudokuCell.jsx';

export default function SudokuBoard({ readOnly = false }) {
  const { state, selectCell, inputValue, getCellState } = useSudoku();
  const { mode, userBoard } = state;

  if (!mode || userBoard.length === 0) return null;

  const size = mode === 'easy' ? 6 : 9;

  const handleKeyDown = useCallback((e, row, col) => {
    if (readOnly) return;
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
  }, [size, inputValue, selectCell, readOnly]);

  return (
    <div
      className={`sudoku-board size-${size}${readOnly ? ' board-readonly' : ''}`}
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
            onSelect={readOnly ? () => {} : selectCell}
            onKeyDown={handleKeyDown}
            readOnly={readOnly}
          />
        ))
      )}
    </div>
  );
}
