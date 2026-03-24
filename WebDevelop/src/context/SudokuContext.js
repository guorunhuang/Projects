import React, { createContext, useContext, useReducer, useEffect, useCallback } from 'react';
import { generatePuzzle, isCellInvalid, isBoardComplete, findHintCell } from '../utils/sudoku';

// ─────────────────────────────────────────────
//  STORE  —  the single source of truth
// ─────────────────────────────────────────────
const LS_KEY = 'sudoku_game_state';

/** Load persisted state from localStorage (only through Context) */
function loadFromStorage() {
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

/** Save state to localStorage (only through Context) */
function saveToStorage(state) {
  try {
    // Don't persist transient UI state (selectedCell, hintCell)
    const { selectedCell, hintCell, timerActive, ...persist } = state;
    window.localStorage.setItem(LS_KEY, JSON.stringify(persist));
  } catch { /* quota exceeded — ignore */ }
}

/** Clear localStorage (only through Context) */
function clearStorage() {
  try { window.localStorage.removeItem(LS_KEY); } catch { }
}

const initialState = {
  mode: null,          // 'easy' | 'normal'
  puzzle: [],          // original puzzle grid (immutable)
  solution: [],        // fully solved grid (for reference)
  userBoard: [],       // mutable player board
  fixedCells: [],      // boolean grid — true = pre-filled (cannot edit)
  selectedCell: null,  // { row, col } | null — transient UI
  hintCell: null,      // { row, col, value } | null — transient UI
  isComplete: false,
  elapsedSeconds: 0,
  timerActive: false,
};

// ─────────────────────────────────────────────
//  ACTIONS  —  plain objects describing events
// ─────────────────────────────────────────────
export const Actions = {
  NEW_GAME:    (mode)             => ({ type: 'NEW_GAME', mode }),
  RESET_GAME:  ()                 => ({ type: 'RESET_GAME' }),
  SELECT_CELL: (row, col)         => ({ type: 'SELECT_CELL', cell: { row, col } }),
  INPUT_VALUE: (row, col, value)  => ({ type: 'INPUT_VALUE', row, col, value }),
  TICK:        ()                 => ({ type: 'TICK' }),
  SHOW_HINT:   ()                 => ({ type: 'SHOW_HINT' }),
  RESTORE:     (saved)            => ({ type: 'RESTORE', saved }),
};

// ─────────────────────────────────────────────
//  REDUCER  —  pure function: (state, action) → state
// ─────────────────────────────────────────────
export function reducer(state, action) {
  switch (action.type) {

    case 'NEW_GAME': {
      const { mode } = action;
      const size  = mode === 'easy' ? 6 : 9;
      const clues = mode === 'easy' ? 18 : 29;
      const { puzzle, solution } = generatePuzzle(size, clues);
      const fixedCells = puzzle.map(row => row.map(v => v !== 0));
      const userBoard  = puzzle.map(row => [...row]);
      return {
        ...initialState,
        mode,
        puzzle:       puzzle.map(r => [...r]),
        solution,
        userBoard,
        fixedCells,
        timerActive:  true,
        elapsedSeconds: 0,
      };
    }

    case 'RESET_GAME': {
      // Revert userBoard to original puzzle; restart timer
      return {
        ...state,
        userBoard:      state.puzzle.map(row => [...row]),
        selectedCell:   null,
        hintCell:       null,
        isComplete:     false,
        elapsedSeconds: 0,
        timerActive:    true,
      };
    }

    case 'SELECT_CELL': {
      if (state.isComplete) return state;
      return { ...state, selectedCell: action.cell, hintCell: null };
    }

    case 'INPUT_VALUE': {
      const { row, col, value } = action;
      if (state.fixedCells[row]?.[col]) return state;
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
      };
    }

    case 'TICK': {
      if (!state.timerActive || state.isComplete) return state;
      return { ...state, elapsedSeconds: state.elapsedSeconds + 1 };
    }

    case 'SHOW_HINT': {
      const size = state.mode === 'easy' ? 6 : 9;
      const hint = findHintCell(state.userBoard, size);
      return { ...state, hintCell: hint };
    }

    case 'RESTORE': {
      // Rehydrate from localStorage; resume timer
      return { ...initialState, ...action.saved, timerActive: true, selectedCell: null, hintCell: null };
    }

    default:
      return state;
  }
}

// ─────────────────────────────────────────────
//  CONTEXT  —  wires Store + Actions + Reducer
// ─────────────────────────────────────────────
const SudokuContext = createContext(null);

export function SudokuProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // ── On mount: rehydrate from localStorage ──
  useEffect(() => {
    const saved = loadFromStorage();
    if (saved && saved.mode && saved.puzzle?.length) {
      dispatch(Actions.RESTORE(saved));
    }
  }, []); // eslint-disable-line

  // ── After every meaningful state change: persist to localStorage ──
  useEffect(() => {
    if (!state.mode) return; // nothing to persist before a game starts
    if (state.isComplete) {
      clearStorage(); // game over → clear
    } else {
      saveToStorage(state);
    }
  }, [state]);

  // ── Timer tick ──
  useEffect(() => {
    if (!state.timerActive) return;
    const id = setInterval(() => dispatch(Actions.TICK()), 1000);
    return () => clearInterval(id);
  }, [state.timerActive]);

  // ── Action dispatchers ──
  const newGame    = useCallback((mode) => {
    clearStorage();
    dispatch(Actions.NEW_GAME(mode));
  }, []);

  const resetGame  = useCallback(() => {
    clearStorage();
    dispatch(Actions.RESET_GAME());
  }, []);

  const selectCell = useCallback((row, col) => dispatch(Actions.SELECT_CELL(row, col)), []);
  const inputValue = useCallback((row, col, value) => dispatch(Actions.INPUT_VALUE(row, col, value)), []);
  const showHint   = useCallback(() => dispatch(Actions.SHOW_HINT()), []);

  // ── Derived cell state (used by SudokuCell) ──
  const getCellState = useCallback((row, col) => {
    const { fixedCells, userBoard, selectedCell, hintCell, mode, isComplete } = state;
    const size       = mode === 'easy' ? 6 : 9;
    const boxSize    = size === 9 ? 3 : 2;
    const isFixed    = !!fixedCells[row]?.[col];
    const isSelected = selectedCell?.row === row && selectedCell?.col === col;
    const isHint     = hintCell?.row === row && hintCell?.col === col;
    const value      = userBoard[row]?.[col] ?? 0;
    const isInvalid  = !isFixed && value !== 0 && isCellInvalid(userBoard, row, col, size);

    let isRelated = false;
    if (selectedCell && !isSelected) {
      isRelated =
        selectedCell.row === row ||
        selectedCell.col === col ||
        (Math.floor(selectedCell.row / boxSize) === Math.floor(row / boxSize) &&
         Math.floor(selectedCell.col / boxSize) === Math.floor(col / boxSize));
    }

    const sameValue =
      selectedCell && !isSelected &&
      value !== 0 &&
      userBoard[selectedCell.row]?.[selectedCell.col] === value;

    return { isFixed, isSelected, isInvalid, isRelated, isHint, value, sameValue, isComplete };
  }, [state]);

  const formatTime = useCallback((secs) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }, []);

  return (
    <SudokuContext.Provider value={{
      state, dispatch,        // expose raw dispatch for advanced use
      newGame, resetGame,
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
