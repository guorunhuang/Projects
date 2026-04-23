import React from 'react';
import { useSudoku } from '../sudoku/SudokuContext.jsx';

export default function Timer() {
  const { state, formatTime } = useSudoku();
  const { elapsedSeconds, mode } = state;
  if (!mode) return null;
  return (
    <div className="timer">
      <span className="timer-icon">⏱</span>
      <span className="timer-display">{formatTime(elapsedSeconds)}</span>
    </div>
  );
}
