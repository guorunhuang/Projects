import React from 'react';
import './SudokuCell.css';

export default function SudokuCell({ row, col, cellState, onSelect, onKeyDown, size }) {
  const { isFixed, isSelected, isInvalid, isRelated, isHint, value, sameValue, isComplete } = cellState;

  const boxSize = size === 9 ? 3 : 2;
  const isBorderRight = (col + 1) % boxSize === 0 && col !== size - 1;
  const isBorderBottom = (row + 1) % boxSize === 0 && row !== size - 1;

  const classes = [
    'sudoku-cell',
    isFixed ? 'fixed' : 'editable',
    isSelected ? 'selected' : '',
    isInvalid ? 'invalid' : '',
    isRelated && !isSelected ? 'related' : '',
    isHint ? 'hint' : '',
    sameValue && !isSelected ? 'same-value' : '',
    isBorderRight ? 'border-box-right' : '',
    isBorderBottom ? 'border-box-bottom' : '',
    isComplete ? 'locked' : '',
  ].filter(Boolean).join(' ');

  return (
    <div
      className={classes}
      tabIndex={isFixed || isComplete ? -1 : 0}
      onClick={() => !isFixed && !isComplete && onSelect(row, col)}
      onKeyDown={(e) => !isFixed && !isComplete && onKeyDown(e, row, col)}
      role="gridcell"
      aria-label={`Row ${row + 1}, Column ${col + 1}, value ${value || 'empty'}`}
    >
      <span className="cell-value">{value !== 0 ? value : ''}</span>
    </div>
  );
}
