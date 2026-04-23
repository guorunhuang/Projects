import express from 'express';
import * as model from './db/model/sudoku.model.js';
import { generatePuzzle, generateGameName, validateCustomPuzzle } from '../utils.js';

const router = express.Router();
const COOKIE_NAME = 'GoSudoku_user';

function getUser(req) {
  return req.cookies[COOKIE_NAME] || null;
}

// ── GET /api/sudoku — list all games ──────────────────────────────────────
router.get('/', async (req, res) => {
  try {
    const games = await model.getAllGames();
    return res.json(games);
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── POST /api/sudoku/custom — validate & create a custom game ─────────────
// Must be declared BEFORE /:id so Express doesn't treat "custom" as an id.
router.post('/custom', async (req, res) => {
  try {
    const username = getUser(req);
    if (!username) return res.status(401).json({ error: 'Must be logged in to create a custom game.' });

    const { puzzle, name: customName } = req.body;
    if (!puzzle || !Array.isArray(puzzle))
      return res.status(400).json({ error: 'puzzle is required.' });

    // Run backtracking uniqueness check
    const result = validateCustomPuzzle(puzzle.map(r => [...r]));
    if (!result.valid)
      return res.status(422).json({ error: result.reason });

    const name = (customName && customName.trim()) ? customName.trim() : generateGameName();
    const game = await model.createGame({
      name,
      difficulty: 'normal',
      puzzle,
      solution: result.solution,
      createdBy: username,
    });

    return res.status(201).json({ id: game._id });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── POST /api/sudoku — create a new generated game ────────────────────────
router.post('/', async (req, res) => {
  try {
    const { difficulty } = req.body;
    if (difficulty !== 'easy' && difficulty !== 'normal')
      return res.status(400).json({ error: 'difficulty must be "easy" or "normal".' });

    const size  = difficulty === 'easy' ? 6 : 9;
    const clues = difficulty === 'easy' ? 18 : 29;
    const { puzzle, solution } = generatePuzzle(size, clues);
    const name  = generateGameName();
    const createdBy = getUser(req) || 'anonymous';

    const game = await model.createGame({ name, difficulty, puzzle, solution, createdBy });
    return res.status(201).json({ id: game._id });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── GET /api/sudoku/:id — get a single game ───────────────────────────────
router.get('/:id', async (req, res) => {
  try {
    const game = await model.getGameById(req.params.id);
    if (!game) return res.status(404).json({ error: 'Game not found.' });

    const username = getUser(req);
    // Return user-specific progress if available
    const progress = username && game.userProgress.get(username);

    return res.json({
      _id:        game._id,
      name:       game.name,
      difficulty: game.difficulty,
      puzzle:     game.puzzle,
      solution:   game.solution,
      createdBy:  game.createdBy,
      createdAt:  game.createdAt,
      userBoard:  progress ? progress.board      : null,
      isComplete: progress ? progress.isComplete : false,
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── PUT /api/sudoku/:id — save user progress on a game ────────────────────
router.put('/:id', async (req, res) => {
  try {
    const username = getUser(req);
    if (!username) return res.status(401).json({ error: 'Must be logged in to save progress.' });

    const { board, isComplete } = req.body;
    if (!board) return res.status(400).json({ error: 'board is required.' });

    const game = await model.saveProgress(req.params.id, username, board, !!isComplete);
    return res.json({ message: 'Progress saved.', isComplete: !!isComplete });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── DELETE /api/sudoku/:id — delete a game (creator only) ────────────────
router.delete('/:id', async (req, res) => {
  try {
    const username = getUser(req);
    if (!username) return res.status(401).json({ error: 'Must be logged in to delete a game.' });

    const game = await model.getGameById(req.params.id);
    if (!game) return res.status(404).json({ error: 'Game not found.' });

    if (game.createdBy !== username)
      return res.status(403).json({ error: 'Only the creator of this game can delete it.' });

    await model.deleteGame(req.params.id);
    return res.json({ message: 'Game deleted.' });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});


export default router;
