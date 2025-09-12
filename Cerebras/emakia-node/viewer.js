const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express(); // ← This line must be here
const PORT = process.env.PORT || 3000;


const useMockInputs = process.env.USE_MOCK_INPUTS === 'true';
const data = useMockInputs ? loadInputs() : require('./inputs/reddit_posts.json');

// Then your route setup
app.get('/', (req, res) => {
  fs.readFile(path.join(__dirname, 'inputs/reddit_posts.json'), 'utf8', (err, data) => {
    if (err || !data.trim()) {
      return res.send('<h2>No post data found. Try rerunning the scraper and check "inputs/reddit_posts.json"</h2>');
    }

    let posts = [];
    try {
      posts = JSON.parse(data);
    } catch (parseErr) {
      return res.send('<h2>Could not parse JSON. Make sure reddit_posts.json is valid.</h2>');
    }

    const html = posts.map(post => `
      <div style="margin-bottom: 2em;">
        <h3>${post.title}</h3>
        <p>${post.content}</p>
        <a href="${post.link}" target="_blank">🔗 View Post</a>
        <hr/>
      </div>
    `).join('');

    res.send(`
      <html>
        <head><title>Reddit Posts Viewer</title></head>
        <body style="font-family:sans-serif; padding:2em;">
          <h1>🧵 Reddit Posts</h1>
          ${html}
        </body>
      </html>
    `);
  });
});

// And finally your server bootstrap
app.listen(PORT, () => {
  console.log(`🚀 Viewer running at http://localhost:${PORT}`);
});
app.get('/', (req, res) => {
  fs.readFile(path.join(__dirname, 'inputs/reddit_posts.json'), 'utf8', (err, data) => {
    if (err || !data.trim()) {
      return res.send('<h2>No post data found. Try rerunning the scraper and check "inputs/reddit_posts.json"</h2>');
    }

    let posts = [];
    try {
      posts = JSON.parse(data);
    } catch (parseErr) {
      return res.send('<h2>Could not parse JSON. Make sure reddit_posts.json is valid.</h2>');
    }

    const html = posts.map(post => `
      <div style="margin-bottom: 2em;">
        <h3>${post.title}</h3>
        <p>${post.content}</p>
        <a href="${post.link}" target="_blank">🔗 View Post</a>
        <hr/>
      </div>
    `).join('');

    res.send(`
      <html>
        <head><title>Reddit Posts Viewer</title></head>
        <body style="font-family:sans-serif; padding:2em;">
          <h1>🧵 Reddit Posts</h1>
          ${html}
        </body>
      </html>
    `);
  });
});
