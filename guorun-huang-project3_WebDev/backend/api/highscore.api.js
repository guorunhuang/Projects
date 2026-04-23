import express from 'express';
import * as model from './db/model/sudoku.model.js';

const router = express.Router();
const COOKIE_NAME = 'GoSudoku_user';

function getUser(req) {
  return req.cookies[COOKIE_NAME] || null;
}

// ── GET /api/highscore — all users with wins > 0, ordered by wins desc ─────
router.get('/', async (req, res) => {
  try {
    const scores = await model.getLeaderboard();
    return res.json(scores);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── POST /api/highscore — record a win for the logged-in user ─────────────
router.post('/', async (req, res) => {
  try {
    const username = getUser(req);
    if (!username) return res.status(401).json({ error: 'Must be logged in to record a win.' });

    const user = await model.recordWin(username);
    if (!user) return res.status(404).json({ error: 'User not found.' });

    return res.status(201).json({ username: user.username, wins: user.wins });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── GET /api/highscore/:gameId — scores for users who completed a specific game ──
router.get('/:gameId', async (req, res) => {
  try {
    const scores = await model.getScoresByGame(req.params.gameId);
    return res.json(scores);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

export default router;
