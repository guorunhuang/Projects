import React from 'react';
import { Link } from 'react-router-dom';
import './Auth.css';

export default function Register() {
  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">Numer<span className="accent">iX</span></div>
        <h1 className="auth-title">Create account</h1>
        <p className="auth-sub">Join GoSudoku and start solving puzzles today.</p>

        <div className="auth-form">
          <div className="field-group">
            <label className="field-label" htmlFor="reg-username">Username</label>
            <input
              id="reg-username"
              type="text"
              className="field-input"
              placeholder="choose_a_username"
              autoComplete="username"
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="reg-password">Password</label>
            <input
              id="reg-password"
              type="password"
              className="field-input"
              placeholder="••••••••"
              autoComplete="new-password"
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="reg-confirm">Confirm Password</label>
            <input
              id="reg-confirm"
              type="password"
              className="field-input"
              placeholder="••••••••"
              autoComplete="new-password"
            />
          </div>

          <button className="auth-submit" type="button">Create Account</button>
        </div>

        <p className="auth-switch">
          Already have an account? <Link to="/login" className="auth-link">Sign in →</Link>
        </p>
      </div>
    </div>
  );
}
