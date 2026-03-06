require('dotenv').config();
const { loadRedditInputs } = require('../inputs/inputLoader');
const classifyPost = require('./classifyPost');

async function runClassifier(subreddit = 'politics', limit = 3) {
  try {
    const posts = await loadRedditInputs(subreddit, limit);

    if (!posts || posts.length === 0) {
      console.warn(`⚠️ No posts found for subreddit: ${subreddit}`);
      return;
    }

    for (let [index, post] of posts.entries()) {
      const result = await classifyPost(post); // in case it's async
      console.log(`\n🔍 Post ${index + 1}: ${post.title}`);
      console.log(`🧠 Result: ${JSON.stringify(result, null, 2)}`);
    }
  } catch (err) {
    console.error('🚨 Error running classifier:', err);
  }
}

runClassifier();
