import React from 'react';
import { Link } from 'react-router-dom';
import './Select.css';

const GAMES = [
  { id: 1, title: 'Classic Sudoku — Easy', author: 'Guorun Huang', mode: 'easy', difficulty: 'Easy', rating: 4.8, plays: 12340 },
  { id: 2, title: 'Classic Sudoku — Normal', author: 'Guorun Huang', mode: 'normal', difficulty: 'Normal', rating: 4.9, plays: 8920 },
];

export default function Select() {
  return (
    <div className="select-page">
      <div className="page-header">
        <h1 className="page-title">Choose a <span className="accent">Game</span></h1>
        <p className="page-sub">Select a puzzle to start playing. Every game generates a fresh board.</p>
      </div>
      
      <div className="filter-tabs">
        <span className="filter-tab active">All</span>
        <span className="filter-tab">Easy</span>
        <span className="filter-tab">Normal</span>
      </div>

      <div className="games-grid">
        {GAMES.map(game => (
          <Link to={`/games/${game.mode}`} key={game.id} className="game-card">
            <div className="card-top">
              <span className={`difficulty-badge ${game.mode}`}>{game.difficulty}</span>
              <div className="card-rating">★ {game.rating}</div>
            </div>
            <h3 className="card-title">{game.title}</h3>
            <p className="card-author">by {game.author}</p>
            <div className="card-footer">
              <span className="card-plays">{game.plays.toLocaleString()} plays</span>
              <span className="card-cta">Play →</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
