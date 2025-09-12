const axios = require('axios');
const cheerio = require('cheerio');

async function getArticleTextFromUrl(url) {
  try {
    const response = await axios.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' }, timeout: 10000 });
    const $ = cheerio.load(response.data);

    const candidates = ['article', 'main', '#content', '.article-body', '#mainArticleDiv'];
    for (const selector of candidates) {
      const block = $(selector);
      if (block.length && block.text().trim()) {
        return block.text().trim().replace(/\n+/g, '\n');
      }
    }

    const paragraphs = $('p');
    if (paragraphs.length) {
      return paragraphs.map((_, p) => $(p).text()).get().join('\n');
    }

    return 'No article content found.';
  } catch (error) {
    return `[link](${url})`;
  }
}

module.exports = { getArticleTextFromUrl };
