import React from 'react';
import { useSudoku } from '../sudoku/SudokuContext.jsx';

export default function NumberPad() {
  const { state, inputValue } = useSudoku();
  const { mode, selectedCell, isComplete } = state;
  const size = mode === 'easy' ? 6 : 9;

  if (!selectedCell || isComplete) return null;

  const handlePress = (num) => {
    inputValue(selectedCell.row, selectedCell.col, num);
  };

  return (
    <div className="number-pad">
      {Array.from({ length: size }, (_, i) => i + 1).map(n => (
        <button key={n} className="pad-btn" onClick={() => handlePress(n)}>
          {n}
        </button>
      ))}
      <button className="pad-btn erase-btn" onClick={() => handlePress(0)}>
        ⌫
      </button>
    </div>
  );
}
