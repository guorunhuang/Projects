import express      from 'express';
import mongoose     from 'mongoose';
import cookieParser from 'cookie-parser';
import path         from 'path';

import sudokuRouter    from './backend/api/sudoku.api.js';
import userRouter      from './backend/api/user.api.js';
import highscoreRouter from './backend/api/highscore.api.js';

const app = express();

app.use(express.json());
app.use(cookieParser());
app.use(express.urlencoded({ extended: true }));

// ── MongoDB Username r38932537_db_user Password vntYNvd4cUNjM7bA────────────────────────────────────────────────────────────────
// mongodb+srv://r38932537_db_user:dGdI12m8anymp4SX@sudokuprojectcluster.fq7dpgl.mongodb.net/?appName=SudokuProjectCluster
const MONGODB_URL = process.env.MONGODB_URL || 'mongodb+srv://r38932537_db_user:dGdI12m8anymp4SX@sudokuprojectcluster.fq7dpgl.mongodb.net/?appName=SudokuProjectCluster';
mongoose.connect(MONGODB_URL);
const db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));
db.once('open', () => console.log('Connected to MongoDB'));

// ── API Routes ─────────────────────────────────────────────────────────────
app.use('/api/sudoku',     sudokuRouter);
app.use('/api/users',      userRouter);
app.use('/api/highscore',  highscoreRouter);

// ── Serve React frontend (production) ─────────────────────────────────────
const __dirname     = path.resolve();
const frontend_dir  = path.join(__dirname, 'dist');

app.use(express.static(frontend_dir));
app.get('*', (req, res) => {
  res.sendFile(path.join(frontend_dir, 'index.html'));
});

app.listen(8000, () => console.log('GoSudoku server running on port 8000'));
