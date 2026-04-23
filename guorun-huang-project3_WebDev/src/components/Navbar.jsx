import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../sudoku/AuthContext.jsx';

export default function Navbar() {
  const location  = useLocation();
  const navigate  = useNavigate();
  const { user, logout } = useAuth();
  const [open, setOpen]  = useState(false);

  const navLinks = [
    { to: '/',       label: 'Home'   },
    { to: '/games',  label: 'Games'  },
    { to: '/rules',  label: 'Rules'  },
    { to: '/scores', label: 'Scores' },
  ];

  const handleLogout = async () => {
    await logout();
    setOpen(false);
    navigate('/');
  };

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand" onClick={() => setOpen(false)}>
        Go<span className="brand-ix">Sudoku</span>
      </Link>

      <button
        className="navbar-burger"
        onClick={() => setOpen(o => !o)}
        aria-label="Toggle menu"
      >
        <span className={`burger-bar ${open ? 'open' : ''}`} />
        <span className={`burger-bar ${open ? 'open' : ''}`} />
        <span className={`burger-bar ${open ? 'open' : ''}`} />
      </button>

      <ul className={`navbar-links ${open ? 'open' : ''}`}>
        {navLinks.map(({ to, label }) => (
          <li key={to}>
            <Link
              to={to}
              className={`nav-link ${location.pathname === to ? 'active' : ''}`}
              onClick={() => setOpen(false)}
            >
              {label}
            </Link>
          </li>
        ))}

        {user ? (
          <>
            <li className="nav-username">
              <span className="username-pill">👤 {user.username}</span>
            </li>
            <li>
              <button className="nav-btn logout-btn" onClick={handleLogout}>
                Log Out
              </button>
            </li>
          </>
        ) : (
          <>
            <li>
              <Link to="/login" className={`nav-link ${location.pathname === '/login' ? 'active' : ''}`} onClick={() => setOpen(false)}>
                Login
              </Link>
            </li>
            <li>
              <Link to="/register" className="nav-btn" onClick={() => setOpen(false)}>
                Register
              </Link>
            </li>
          </>
        )}
      </ul>
    </nav>
  );
}
