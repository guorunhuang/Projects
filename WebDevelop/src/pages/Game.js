import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useSudoku } from '../context/SudokuContext';
import SudokuBoard from '../components/SudokuBoard';
import NumberPad from '../components/NumberPad';
import Timer from '../components/Timer';
import './Game.css';

export default function Game() {
  const { mode } = useParams();
  const navigate = useNavigate();
  const { state, newGame, resetGame, showHint } = useSudoku();
  const { isComplete, elapsedSeconds, mode: activeMode } = state;

  useEffect(() => {
    if (mode !== 'easy' && mode !== 'normal') {
      navigate('/games');
      return;
    }
    // Only start a new game if mode differs from current active game
    if (mode !== activeMode) {
      newGame(mode);
    }
  }, [mode]); // eslint-disable-line

  const fmt = (s) => {
    const m = Math.floor(s / 60).toString().padStart(2,'0');
    const sec = (s % 60).toString().padStart(2,'0');
    return `${m}:${sec}`;
  };

  return (
    <div className="game-page">
      <div className="game-header">
        <div className="game-title-row">
          <h1 className="game-title">
            {mode === 'easy' ? 'Easy' : 'Normal'}{' '}
            <span className="accent">Sudoku</span>
          </h1>
          <Timer />
        </div>
        {isComplete && (
          <div className="congrats-banner">
            Puzzle Complete! Solved in {fmt(elapsedSeconds)}
          </div>
        )}
      </div>

      <div className="game-layout">
        <div className="board-section">
          <SudokuBoard />
          <NumberPad />
        </div>

        <div className="game-controls">
          <div className="controls-card">
            <div className="controls-title">Controls</div>
            <p className="controls-hint">Click a cell, then type a number (or use the pad below).</p>
            <p className="controls-hint">Arrow keys navigate. Backspace erases.</p>

            <div className="legend">
              <div className="legend-item">
                <div className="legend-swatch swatch-fixed" />
                <span>Given clue</span>
              </div>
              <div className="legend-item">
                <div className="legend-swatch swatch-selected" />
                <span>Selected cell</span>
              </div>
              <div className="legend-item">
                <div className="legend-swatch swatch-invalid" />
                <span>Invalid entry</span>
              </div>
              <div className="legend-item">
                <div className="legend-swatch swatch-hint" />
                <span>Hint cell</span>
              </div>
            </div>

            <div className="game-buttons">
              <button className="game-btn hint-btn"  onClick={showHint}         disabled={isComplete}>💡 Hint</button>
              <button className="game-btn reset-btn" onClick={resetGame}        disabled={isComplete}>↺ Reset</button>
              <button className="game-btn new-btn"   onClick={() => newGame(mode)}>⊕ New Game</button>
            </div>

            <div className="mode-switch">
              <span className="mode-label">Switch difficulty:</span>
              <button className="mode-link" onClick={() => navigate(`/games/${mode === 'easy' ? 'normal' : 'easy'}`)}>
                {mode === 'easy' ? 'Try Normal →' : 'Try Easy →'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
