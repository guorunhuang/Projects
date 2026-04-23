import mongoose from 'mongoose';

const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    minlength: 1,
  },
  password: {
    type: String,
    required: true,
  },
  wins: {
    type: Number,
    default: 0,
  },
}, { timestamps: true });

export default mongoose.model('User', userSchema);
