import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SudokuProvider } from './context/SudokuContext';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Select from './pages/Select';
import Game from './pages/Game';
import Rules from './pages/Rules';
import Scores from './pages/Scores';
import Login from './pages/Login';
import Register from './pages/Register';

function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-brand">
        Go<span style={{ color: 'var(--accent)' }}>Sudoku</span>
      </div>
      <div className="footer-copy">© 2026 GoSudoku · All rights reserved</div>
    </footer>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <SudokuProvider>
        <Navbar />
        <Routes>
          <Route path="/"              element={<Home />} />
          <Route path="/games"         element={<Select />} />
          <Route path="/games/:mode"   element={<Game />} />
          <Route path="/rules"         element={<Rules />} />
          <Route path="/scores"        element={<Scores />} />
          <Route path="/login"         element={<Login />} />
          <Route path="/register"      element={<Register />} />
        </Routes>
        <Footer />
      </SudokuProvider>
    </BrowserRouter>
  );
}
