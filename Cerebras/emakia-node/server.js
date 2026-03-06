const express = require('express');
const app = express();

app.get('/', (req, res) => {
  const code = req.query.code;
  const state = req.query.state;

  if (code) {
    console.log(`✅ Received OAuth code: ${code}`);
    res.send(`OAuth code received: ${code}`);
  } else {
    console.warn('⚠️ No code found in redirect');
    res.send('No code received in redirect');
  }
});

app.listen(8080, () => {
  console.log('🔊 Server listening on http://localhost:8080');
});
