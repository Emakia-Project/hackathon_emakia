const fs = require('fs');
const path = require('path');
const redditLoader = require('./reddit/redditLoader');

async function loadRedditInputs(subreddit = 'politics', limit = 5) {
  return await redditLoader.getPosts(subreddit, limit);
}

function loadMockInputs(filepath = path.join(__dirname, 'input_config.json')) {
  if (!fs.existsSync(filepath)) {
    throw new Error(`Input file not found at ${filepath}`);
  }
  const rawData = fs.readFileSync(filepath);
  return JSON.parse(rawData);
}

module.exports = { loadRedditInputs, loadMockInputs };
