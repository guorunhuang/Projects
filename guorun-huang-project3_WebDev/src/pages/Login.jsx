import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../sudoku/AuthContext.jsx';

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error,    setError]    = useState('');
  const [loading,  setLoading]  = useState(false);

  const disabled = !username.trim() || !password.trim() || loading;

  async function handleSubmit() {
    if (disabled) return;
    setLoading(true);
    setError('');
    try {
      await login(username.trim(), password);
      navigate('/games');
    } catch (err) {
      setError(err.message || 'Login failed.');
    } finally {
      setLoading(false);
    }
  }

  const handleKey = (e) => { if (e.key === 'Enter') handleSubmit(); };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">Go<span className="accent">Sudoku</span></div>
        <h1 className="auth-title">Welcome back</h1>
        <p className="auth-sub">Sign in to track your progress and compete on the leaderboard.</p>

        {error && <div className="form-error">{error}</div>}

        <div className="auth-form">
          <div className="field-group">
            <label className="field-label" htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              className="field-input"
              placeholder="your_username"
              autoComplete="username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              className="field-input"
              placeholder="••••••••"
              autoComplete="current-password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <button
            className="auth-submit"
            onClick={handleSubmit}
            disabled={disabled}
          >
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </div>

        <p className="auth-switch">
          Don't have an account? <Link to="/register" className="auth-link">Register →</Link>
        </p>
      </div>
    </div>
  );
}
