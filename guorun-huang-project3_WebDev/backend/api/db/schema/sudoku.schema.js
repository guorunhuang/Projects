import mongoose from 'mongoose';

const sudokuSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  difficulty: {
    type: String,
    enum: ['easy', 'normal'],
    required: true,
  },
  // The original puzzle (0 = empty cell)
  puzzle: {
    type: [[Number]],
    required: true,
  },
  // The fully solved board
  solution: {
    type: [[Number]],
    required: true,
  },
  // Created by which user
  createdBy: {
    type: String,
    default: 'anonymous',
  },
  // Per-user progress: { username -> { board: [[Number]], isComplete: Boolean } }
  userProgress: {
    type: Map,
    of: new mongoose.Schema({
      board: [[Number]],
      isComplete: { type: Boolean, default: false },
    }, { _id: false }),
    default: {},
  },
}, { timestamps: true });

export default mongoose.model('Sudoku', sudokuSchema);
