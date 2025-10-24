import { Devvit, SettingScope } from '@devvit/public-api';

Devvit.addSettings([
  {
    type: 'string',
    name: 'OPENAI_API_KEY',
    label: 'OpenAI API Key',
    scope: SettingScope.App,
    isSecret: true,
  },
]);

export default Devvit.configure({
  redditAPI: true,
});
