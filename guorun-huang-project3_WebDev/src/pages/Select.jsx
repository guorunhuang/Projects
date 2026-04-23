import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../sudoku/AuthContext.jsx';

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export default function Select() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [games,    setGames]    = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [creating, setCreating] = useState(false);
  const [filter,   setFilter]   = useState('all');
  const [error,    setError]    = useState('');

  useEffect(() => { fetchGames(); }, []);

  async function fetchGames() {
    try {
      const res  = await fetch('/api/sudoku');
      const data = await res.json();
      setGames(data);
    } catch {
      setError('Failed to load games.');
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate(difficulty) {
    if (!user) return;
    setCreating(true);
    setError('');
    try {
      const res = await fetch('/api/sudoku', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ difficulty }),
      });
      const { id } = await res.json();
      if (!res.ok) throw new Error('Failed to create game.');
      navigate(`/game/${id}`);
    } catch (err) {
      setError(err.message || 'Failed to create game.');
      setCreating(false);
    }
  }

  const displayed = filter === 'all' ? games : games.filter(g => g.difficulty === filter);

  return (
    <div className="select-page">
      <div className="page-header">
        <h1 className="page-title">Choose a <span className="accent">Game</span></h1>
        <p className="page-sub">Create a new puzzle or jump into an existing one.</p>
      </div>

      {!user && (
        <div className="guest-banner">
          <span>
            👀 Browsing as guest —{' '}
            <Link to="/login" className="guest-link">sign in</Link> or{' '}
            <Link to="/register" className="guest-link">register</Link>{' '}
            to create and play games.
          </span>
        </div>
      )}

      <div className="create-buttons">
        <button
          className="create-btn easy-create"
          onClick={() => handleCreate('easy')}
          disabled={creating || !user}
          title={!user ? 'Log in to create a game' : ''}
        >
          {creating ? '⏳ Creating…' : '＋ Create Easy Game'}
        </button>
        <button
          className="create-btn normal-create"
          onClick={() => handleCreate('normal')}
          disabled={creating || !user}
          title={!user ? 'Log in to create a game' : ''}
        >
          {creating ? '⏳ Creating…' : '＋ Create Normal Game'}
        </button>
        <button
          className="create-btn custom-create"
          onClick={() => user && navigate('/custom')}
          disabled={!user}
          title={!user ? 'Log in to create a custom game' : ''}
        >
          ✎ Create Custom Game
        </button>
      </div>

      {error && <p className="form-error select-error">{error}</p>}

      <div className="filter-tabs">
        {['all', 'easy', 'normal'].map(f => (
          <button
            key={f}
            className={`filter-tab ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading-state">Loading games…</div>
      ) : displayed.length === 0 ? (
        <div className="empty-state">
          No games found.{user ? ' Create one above!' : ''}
        </div>
      ) : (
        <div className="games-grid">
          {displayed.map(game => (
            <button
              key={game._id}
              className={`game-card ${!user ? 'game-card-locked' : ''}`}
              onClick={() => user && navigate(`/game/${game._id}`)}
              title={!user ? 'Log in to play' : ''}
            >
              <div className="card-top">
                <span className={`difficulty-badge ${game.difficulty}`}>
                  {game.difficulty.charAt(0).toUpperCase() + game.difficulty.slice(1)}
                </span>
                {!user && <span className="lock-icon">🔒</span>}
              </div>
              <h3 className="card-title">{game.name}</h3>
              <p className="card-author">by {game.createdBy}</p>
              <div className="card-footer">
                <span className="card-date">{formatDate(game.createdAt)}</span>
                <span className="card-cta">{user ? 'Play →' : 'Login to play'}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
