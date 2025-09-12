

require('dotenv').config();
const Snoowrap = require('snoowrap');
const axios = require('axios');
const cheerio = require('cheerio');

const reddit = new Snoowrap({
  userAgent: 'enaelle-moderator-bot/1.0',
  clientId: process.env.REDDIT_CLIENT_ID,
  clientSecret: process.env.REDDIT_CLIENT_SECRET,
  refreshToken: process.env.REDDIT_REFRESH_TOKEN
});

async function getArticleTextFromUrl(url) {
  try {
    const response = await axios.get(url, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
      timeout: 10000
    });
    const $ = cheerio.load(response.data);
    const candidates = ['article', 'main', '#content', '.article-body', '#mainArticleDiv'];
    for (const selector of candidates) {
      const block = $(selector);
      if (block.text().trim()) return block.text().trim();
    }
    const paragraphs = $('p');
    return paragraphs.length ? paragraphs.map((_, p) => $(p).text()).get().join('\n') : 'No article content found.';
  } catch (err) {
    return `[link](${url})`;
  }
}

async function getRedditPosts(subreddit = 'politics', limit = 3) {
  const posts = [];
  try {
    const fetched = await reddit.getSubreddit(subreddit).getNew({ limit });
    for (const post of fetched) {
      let content = post.selftext?.trim();
      if (!content && post.url) content = await getArticleTextFromUrl(post.url);
      posts.push({
        title: post.title,
        content: content || `[link](${post.url})`,
        link: post.url
      });
    }
  } catch (err) {
    console.error('Reddit fetch error:', err.message);
  }
  return posts;
}

// Example usage
getRedditPosts().then(posts => console.log(posts));
