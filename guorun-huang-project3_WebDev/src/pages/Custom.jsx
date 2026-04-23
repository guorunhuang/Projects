import React, { useState, useCallback, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../sudoku/AuthContext.jsx';

const SIZE = 9;
const BOX  = 3;

function emptyBoard() {
  return Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
}

function isCellConflicted(board, row, col) {
  const val = board[row][col];
  if (val === 0) return false;

  for (let c = 0; c < SIZE; c++)
    if (c !== col && board[row][c] === val) return true;
  for (let r = 0; r < SIZE; r++)
    if (r !== row && board[r][col] === val) return true;

  const br = Math.floor(row / BOX) * BOX;
  const bc = Math.floor(col / BOX) * BOX;
  for (let r = br; r < br + BOX; r++)
    for (let c = bc; c < bc + BOX; c++)
      if ((r !== row || c !== col) && board[r][c] === val) return true;

  return false;
}

export default function Custom() {
  const navigate  = useNavigate();
  const { user }  = useAuth();
  const [board,    setBoard]    = useState(emptyBoard);
  const [selected, setSelected] = useState(null);  // { row, col }
  const [name,     setName]     = useState('');
  const [status,   setStatus]   = useState('idle'); // idle | submitting | error
  const [error,    setError]    = useState('');
  const cellRefs   = useRef({});

  // ── Input handlers ──────────────────────────────────────────────────────
  const setValue = useCallback((row, col, val) => {
    setBoard(prev => {
      const next = prev.map(r => [...r]);
      next[row][col] = val;
      return next;
    });
  }, []);

  const handleCellKey = useCallback((e, row, col) => {
    const digit = parseInt(e.key);
    if (!isNaN(digit) && digit >= 1 && digit <= 9) {
      setValue(row, col, digit);
      return;
    }
    if (e.key === 'Backspace' || e.key === 'Delete' || e.key === '0') {
      setValue(row, col, 0);
      return;
    }
    const moves = { ArrowUp:[-1,0], ArrowDown:[1,0], ArrowLeft:[0,-1], ArrowRight:[0,1] };
    if (moves[e.key]) {
      e.preventDefault();
      const [dr, dc] = moves[e.key];
      const nr = Math.max(0, Math.min(SIZE - 1, row + dr));
      const nc = Math.max(0, Math.min(SIZE - 1, col + dc));
      setSelected({ row: nr, col: nc });
      cellRefs.current[`${nr}-${nc}`]?.focus();
    }
  }, [setValue]);

  const handleCellClick = useCallback((row, col) => {
    setSelected({ row, col });
  }, []);

  // ── Number pad ──────────────────────────────────────────────────────────
  const handlePad = (n) => {
    if (!selected) return;
    setValue(selected.row, selected.col, n);
  };

  // ── Clear board ─────────────────────────────────────────────────────────
  const handleClear = () => {
    setBoard(emptyBoard());
    setSelected(null);
    setError('');
    setStatus('idle');
  };

  // ── Submit ──────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    setStatus('submitting');
    setError('');
    try {
      const res  = await fetch('/api/sudoku/custom', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ puzzle: board, name: name.trim() || undefined }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Validation failed.');
        setStatus('error');
        return;
      }
      navigate(`/game/${data.id}`);
    } catch {
      setError('Network error — please try again.');
      setStatus('error');
    }
  };

  // ── Derived state ───────────────────────────────────────────────────────
  const filledCells = board.flat().filter(v => v !== 0).length;
  const hasConflict = board.some((row, r) => row.some((_, c) => isCellConflicted(board, r, c)));
  const canSubmit   = filledCells > 0 && !hasConflict && status !== 'submitting';

  // ── Redirect guests ─────────────────────────────────────────────────────
  if (!user) {
    return (
      <div className="custom-page">
        <div className="page-header">
          <h1 className="page-title">Custom <span className="accent">Game</span></h1>
        </div>
        <div className="guest-banner">
          👀 You must be logged in to create a custom game.{' '}
          <Link to="/login" className="guest-link">Sign in</Link> or{' '}
          <Link to="/register" className="guest-link">register</Link>.
        </div>
      </div>
    );
  }

  return (
    <div className="custom-page">
      <div className="page-header">
        <h1 className="page-title">Create Custom <span className="accent">Game</span></h1>
        <p className="page-sub">
          Fill in your clues on the 9×9 grid, then hit Submit. The server will verify
          your puzzle has exactly one solution using backtracking.
        </p>
      </div>

      <div className="custom-layout">
        {/* ── Board ── */}
        <div className="custom-board-section">
          <div
            className="sudoku-board size-9 custom-board"
            style={{ '--grid-size': 9 }}
            role="grid"
            aria-label="Custom puzzle editor"
          >
            {board.map((row, r) =>
              row.map((val, c) => {
                const isSelected  = selected?.row === r && selected?.col === c;
                const isConflict  = isCellConflicted(board, r, c);
                const isRelated   = selected && !isSelected && (
                  selected.row === r ||
                  selected.col === c ||
                  (Math.floor(selected.row / BOX) === Math.floor(r / BOX) &&
                   Math.floor(selected.col / BOX) === Math.floor(c / BOX))
                );
                const isBorderRight  = (c + 1) % BOX === 0 && c !== SIZE - 1;
                const isBorderBottom = (r + 1) % BOX === 0 && r !== SIZE - 1;

                const classes = [
                  'sudoku-cell editable custom-cell',
                  isSelected  ? 'selected'   : '',
                  isConflict  ? 'invalid'    : '',
                  isRelated   ? 'related'    : '',
                  isBorderRight  ? 'border-box-right'  : '',
                  isBorderBottom ? 'border-box-bottom' : '',
                ].filter(Boolean).join(' ');

                return (
                  <div
                    key={`${r}-${c}`}
                    ref={el => { cellRefs.current[`${r}-${c}`] = el; }}
                    className={classes}
                    tabIndex={0}
                    role="gridcell"
                    aria-label={`Row ${r+1}, Col ${c+1}, value ${val || 'empty'}`}
                    onClick={() => handleCellClick(r, c)}
                    onKeyDown={(e) => handleCellKey(e, r, c)}
                  >
                    <span className="cell-value">{val !== 0 ? val : ''}</span>
                  </div>
                );
              })
            )}
          </div>

          {/* Number pad */}
          <div className="number-pad custom-pad">
            {[1,2,3,4,5,6,7,8,9].map(n => (
              <button key={n} className="pad-btn" onClick={() => handlePad(n)}>{n}</button>
            ))}
            <button className="pad-btn erase-btn" onClick={() => handlePad(0)}>⌫</button>
          </div>
        </div>

        {/* ── Sidebar ── */}
        <div className="custom-sidebar">
          <div className="controls-card">
            <div className="controls-title">Puzzle Settings</div>

            <div className="field-group">
              <label className="field-label" htmlFor="puzzle-name">Game Name (optional)</label>
              <input
                id="puzzle-name"
                type="text"
                className="field-input"
                placeholder="My Custom Puzzle"
                value={name}
                onChange={e => setName(e.target.value)}
                maxLength={60}
              />
            </div>

            <div className="custom-stats">
              <div className="custom-stat">
                <span className="custom-stat-num">{filledCells}</span>
                <span className="custom-stat-label">Clues placed</span>
              </div>
              <div className="custom-stat">
                <span className={`custom-stat-num ${hasConflict ? 'stat-bad' : 'stat-ok'}`}>
                  {hasConflict ? '✗' : '✓'}
                </span>
                <span className="custom-stat-label">No conflicts</span>
              </div>
            </div>

            <div className="custom-instructions">
              <p>• Click a cell, then type 1–9 or use the number pad.</p>
              <p>• Arrow keys navigate between cells.</p>
              <p>• Backspace erases a cell.</p>
              <p>• Red cells indicate rule conflicts.</p>
              <p>• Uniqueness is verified on the server.</p>
            </div>

            {error && <div className="form-error">{error}</div>}

            <div className="custom-actions">
              <button
                className="game-btn reset-btn"
                onClick={handleClear}
                disabled={status === 'submitting'}
              >
                ✕ Clear Board
              </button>
              <button
                className="custom-submit"
                onClick={handleSubmit}
                disabled={!canSubmit}
              >
                {status === 'submitting' ? '⏳ Verifying…' : '✓ Submit Puzzle'}
              </button>
            </div>

            <p className="custom-note">
              The backend uses backtracking to confirm your puzzle has exactly one valid solution before saving.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
