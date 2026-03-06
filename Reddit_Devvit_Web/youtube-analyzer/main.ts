import { Devvit } from '@devvit/public-api';

Devvit.addMenuItem({
  label: 'Analyze YouTube Video',
  location: 'post',
  onPress: async (_event, context) => {
    await context.ui.navigateTo('https://your-deployed-web-url'); // Replace with your actual frontend URL
  },
});

export default Devvit;
