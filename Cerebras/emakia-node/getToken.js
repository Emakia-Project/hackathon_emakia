(async () => {
  const snoowrap = require('snoowrap');
  const open = (await import('open')).default;

  const clientId = 'dzCa67fTQ5GiTi2viFUOFA';
  const clientSecret = 'xtDgLTap_dNCxukibWvyFnPz_iJV-A';
  const redirectUri = 'http://localhost:8080';
  const scopes = ['identity', 'read'];

  const authUrl = `https://www.reddit.com/api/v1/authorize?client_id=${clientId}&response_type=code&state=randomString&redirect_uri=${redirectUri}&duration=permanent&scope=${scopes.join(',')}`;
 
  await open(authUrl);
})();
