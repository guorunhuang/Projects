import React, { createContext, useContext, useReducer, useEffect, useCallback, useRef } from 'react';
import { isCellInvalid, isBoardComplete, findHintCell } from './sudoku.js';

const SudokuContext = createContext(null);

// ─────────────────────────────────────────────
//  INITIAL STATE
// ─────────────────────────────────────────────
const initialState = {
  gameId:       null,
  mode:         null,
  puzzle:       [],
  solution:     [],
  userBoard:    [],
  fixedCells:   [],
  selectedCell: null,
  hintCell:     null,
  isComplete:   false,
  elapsedSeconds: 0,
  timerActive:  false,
  pendingSave:  false,
};

// ─────────────────────────────────────────────
//  REDUCER
// ─────────────────────────────────────────────
function reducer(state, action) {
  switch (action.type) {

    case 'LOAD_GAME': {
      const { gameId, difficulty, puzzle, solution, userBoard, isComplete } = action;
      const fixedCells = puzzle.map(row => row.map(v => v !== 0));
      const board = userBoard && userBoard.length ? userBoard.map(r => [...r]) : puzzle.map(r => [...r]);
      return {
        ...initialState,
        gameId,
        mode:         difficulty,
        puzzle:       puzzle.map(r => [...r]),
        solution:     solution.map(r => [...r]),
        userBoard:    board,
        fixedCells,
        isComplete:   !!isComplete,
        timerActive:  !isComplete,
        elapsedSeconds: 0,
      };
    }

    case 'RESET_GAME': {
      return {
        ...state,
        userBoard:      state.puzzle.map(r => [...r]),
        selectedCell:   null,
        hintCell:       null,
        isComplete:     false,
        elapsedSeconds: 0,
        timerActive:    true,
        pendingSave:    true,
      };
    }

    case 'SELECT_CELL': {
      if (state.isComplete) return state;
      return { ...state, selectedCell: action.cell, hintCell: null };
    }

    case 'INPUT_VALUE': {
      const { row, col, value } = action;
      if (state.fixedCells[row]?.[col] || state.isComplete) return state;
      const size     = state.mode === 'easy' ? 6 : 9;
      const newBoard = state.userBoard.map(r => [...r]);
      newBoard[row][col] = value;
      const complete = isBoardComplete(newBoard, size);
      return {
        ...state,
        userBoard:   newBoard,
        isComplete:  complete,
        timerActive: complete ? false : state.timerActive,
        hintCell:    null,
        pendingSave:  true,
      };
    }

    case 'TICK': {
      if (!state.timerActive || state.isComplete) return state;
      return { ...state, elapsedSeconds: state.elapsedSeconds + 1 };
    }

    case 'SHOW_HINT': {
      const size = state.mode === 'easy' ? 6 : 9;
      const hint = findHintCell(state.userBoard, state.puzzle, size);
      return { ...state, hintCell: hint };
    }

    case 'CLEAR_PENDING_SAVE':
      return { ...state, pendingSave: false };

    default:
      return state;
  }
}

// ─────────────────────────────────────────────
//  PROVIDER
// ─────────────────────────────────────────────
export function SudokuProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const saveTimeoutRef = useRef(null);

  // Timer
  useEffect(() => {
    if (!state.timerActive) return;
    const id = setInterval(() => dispatch({ type: 'TICK' }), 1000);
    return () => clearInterval(id);
  }, [state.timerActive]);

  // Auto-save progress to backend (debounced); also records win via POST /api/highscore
  useEffect(() => {
    if (!state.pendingSave || !state.gameId) return;
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = setTimeout(async () => {
      try {
        await fetch(`/api/sudoku/${state.gameId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ board: state.userBoard, isComplete: state.isComplete }),
        });
        // Record win in highscore table when the board is freshly completed
        if (state.isComplete) {
          await fetch('/api/highscore', { method: 'POST' });
        }
        dispatch({ type: 'CLEAR_PENDING_SAVE' });
      } catch (err) {
        console.error('Failed to save progress:', err);
      }
    }, 800);
    return () => clearTimeout(saveTimeoutRef.current);
  }, [state.pendingSave, state.userBoard, state.isComplete, state.gameId]);

  // ── Action helpers ──
  const loadGame    = useCallback((data) => dispatch({ type: 'LOAD_GAME', ...data }), []);
  const resetGame   = useCallback(() => dispatch({ type: 'RESET_GAME' }), []);
  const selectCell  = useCallback((row, col) => dispatch({ type: 'SELECT_CELL', cell: { row, col } }), []);
  const inputValue  = useCallback((row, col, value) => dispatch({ type: 'INPUT_VALUE', row, col, value }), []);
  const showHint    = useCallback(() => dispatch({ type: 'SHOW_HINT' }), []);

  // ── Derived cell state ──
  const getCellState = useCallback((row, col) => {
    const { fixedCells, userBoard, selectedCell, hintCell, mode, isComplete, solution } = state;
    const size    = mode === 'easy' ? 6 : 9;
    const boxSize = size === 9 ? 3 : 2;
    const isFixed    = !!fixedCells[row]?.[col];
    const isSelected = selectedCell?.row === row && selectedCell?.col === col;
    const isHint     = hintCell?.row === row && hintCell?.col === col;
    const value      = userBoard[row]?.[col] ?? 0;
    const isInvalid  = !isFixed && value !== 0 && isCellInvalid(userBoard, row, col, size);
    const isReveal   = isComplete && solution[row]?.[col] !== undefined;

    let isRelated = false;
    if (selectedCell && !isSelected) {
      isRelated =
        selectedCell.row === row ||
        selectedCell.col === col ||
        (Math.floor(selectedCell.row / boxSize) === Math.floor(row / boxSize) &&
         Math.floor(selectedCell.col / boxSize) === Math.floor(col / boxSize));
    }

    const sameValue =
      selectedCell && !isSelected && value !== 0 &&
      userBoard[selectedCell.row]?.[selectedCell.col] === value;

    return { isFixed, isSelected, isInvalid, isRelated, isHint, value, sameValue, isComplete, isReveal };
  }, [state]);

  const formatTime = useCallback((secs) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }, []);

  return (
    <SudokuContext.Provider value={{
      state, dispatch,
      loadGame, resetGame,
      selectCell, inputValue,
      getCellState, showHint,
      formatTime,
    }}>
      {children}
    </SudokuContext.Provider>
  );
}

export function useSudoku() {
  const ctx = useContext(SudokuContext);
  if (!ctx) throw new Error('useSudoku must be used within SudokuProvider');
  return ctx;
}
