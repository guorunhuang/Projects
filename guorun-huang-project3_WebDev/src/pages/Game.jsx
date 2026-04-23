import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useSudoku } from '../sudoku/SudokuContext.jsx';
import { useAuth }   from '../sudoku/AuthContext.jsx';
import SudokuBoard   from '../components/SudokuBoard.jsx';
import NumberPad     from '../components/NumberPad.jsx';
import Timer         from '../components/Timer.jsx';

export default function Game() {
  const { gameid } = useParams();
  const navigate   = useNavigate();
  const { user }   = useAuth();
  const { state, loadGame, resetGame, showHint, formatTime } = useSudoku();
  const { isComplete, elapsedSeconds } = state;

  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState('');
  const [gameName,    setGameName]    = useState('');
  const [createdBy,   setCreatedBy]   = useState('');
  const [confirmDel,  setConfirmDel]  = useState(false);
  const [deleting,    setDeleting]    = useState(false);
  const [deleteError, setDeleteError] = useState('');

  // Guests can VIEW the board but cannot interact
  const readOnly  = !user;
  const isCreator = user && createdBy && user.username === createdBy;

  useEffect(() => {
    if (!gameid) { navigate('/games'); return; }
    fetchGame();
  }, [gameid]); // eslint-disable-line

  async function fetchGame() {
    setLoading(true);
    setError('');
    try {
      const res  = await fetch(`/api/sudoku/${gameid}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Game not found');
      setGameName(data.name);
      setCreatedBy(data.createdBy);
      loadGame({
        gameId:     data._id,
        difficulty: data.difficulty,
        puzzle:     data.puzzle,
        solution:   data.solution,
        userBoard:  data.userBoard,
        isComplete: data.isComplete,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete() {
    setDeleting(true);
    setDeleteError('');
    try {
      const res  = await fetch(`/api/sudoku/${gameid}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to delete game.');
      navigate('/games');
    } catch (err) {
      setDeleteError(err.message);
      setDeleting(false);
      setConfirmDel(false);
    }
  }

  if (loading) return <div className="game-page"><div className="loading-state">Loading puzzle…</div></div>;
  if (error)   return <div className="game-page"><div className="error-state">Error: {error}</div></div>;

  return (
    <div className="game-page">
      {/* ── Delete confirmation modal ── */}
      {confirmDel && (
        <div className="modal-backdrop" onClick={() => !deleting && setConfirmDel(false)}>
          <div className="modal-card" onClick={e => e.stopPropagation()}>
            <h2 className="modal-title">Delete this game?</h2>
            <p className="modal-body">
              <strong>"{gameName}"</strong> will be permanently removed.
              All players who completed it will lose one win from their score.
            </p>
            {deleteError && <div className="form-error">{deleteError}</div>}
            <div className="modal-actions">
              <button
                className="game-btn"
                onClick={() => setConfirmDel(false)}
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                className="game-btn delete-confirm-btn"
                onClick={handleDelete}
                disabled={deleting}
              >
                {deleting ? '⏳ Deleting…' : '🗑 Yes, Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="game-header">
        <div className="game-title-row">
          <h1 className="game-title">
            {gameName || 'Sudoku'}{' '}
            <span className="difficulty-chip">{state.mode}</span>
          </h1>
          <Timer />
        </div>

        {readOnly && (
          <div className="guest-banner" style={{ marginTop: '0.75rem' }}>
            👀 You're viewing this puzzle as a guest.{' '}
            <Link to="/login" className="guest-link">Sign in</Link> or{' '}
            <Link to="/register" className="guest-link">register</Link> to play.
          </div>
        )}

        {isComplete && (
          <div className="congrats-banner">
            🎉 Puzzle Complete! Solved in {formatTime(elapsedSeconds)}
          </div>
        )}
      </div>

      <div className="game-layout">
        <div className="board-section">
          <SudokuBoard readOnly={readOnly} />
          {!readOnly && <NumberPad />}
        </div>

        <div className="game-controls">
          <div className="controls-card">
            <div className="controls-title">Controls</div>

            {readOnly ? (
              <p className="controls-hint">
                Log in to select cells, enter numbers, and track your progress.
              </p>
            ) : (
              <>
                <p className="controls-hint">Click a cell, then type or tap a number.</p>
                <p className="controls-hint">Arrow keys navigate. Backspace erases.</p>
              </>
            )}

            <div className="legend">
              <div className="legend-item"><div className="legend-swatch swatch-fixed" /><span>Given clue</span></div>
              <div className="legend-item"><div className="legend-swatch swatch-selected" /><span>Selected</span></div>
              <div className="legend-item"><div className="legend-swatch swatch-invalid" /><span>Invalid</span></div>
              <div className="legend-item"><div className="legend-swatch swatch-hint" /><span>Hint</span></div>
            </div>

            {!readOnly && (
              <div className="game-buttons">
                <button className="game-btn hint-btn"  onClick={showHint}  disabled={isComplete}>💡 Hint</button>
                <button className="game-btn reset-btn" onClick={resetGame} disabled={isComplete}>↺ Reset</button>
              </div>
            )}

            {/* DELETE — visible only to the game's creator */}
            {isCreator && (
              <div className="delete-section">
                <div className="delete-divider" />
                <p className="delete-label">You created this game</p>
                <button
                  className="game-btn delete-btn"
                  onClick={() => setConfirmDel(true)}
                >
                  🗑 Delete Game
                </button>
              </div>
            )}

            <div className="mode-switch">
              <button className="mode-link" onClick={() => navigate('/games')}>
                ← Back to Games
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
