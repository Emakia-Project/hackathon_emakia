import express from 'express';
import { InitResponse, IncrementResponse, DecrementResponse, YouTubeAnalysisRequest, YouTubeAnalysisResponse, YouTubeAnalysisError } from '../shared/types/api';
import { redis, reddit, createServer, context, getServerPort, settings } from '@devvit/web/server';
import { createPost } from './core/post';
import { analyzeVideo, extractVideoId, getVideoMetadata } from '../services/analyzer';

const app = express();

// Middleware for JSON body parsing
app.use(express.json());
// Middleware for URL-encoded body parsing
app.use(express.urlencoded({ extended: true }));
// Middleware for plain text body parsing
app.use(express.text());

const router = express.Router();

router.get<{ postId: string }, InitResponse | { status: string; message: string }>(
  '/api/init',
  async (_req, res): Promise<void> => {
    const { postId } = context;

    if (!postId) {
      console.error('API Init Error: postId not found in devvit context');
      res.status(400).json({
        status: 'error',
        message: 'postId is required but missing from context',
      });
      return;
    }

    try {
      const [count, username] = await Promise.all([
        redis.get('count'),
        reddit.getCurrentUsername(),
      ]);

      res.json({
        type: 'init',
        postId: postId,
        count: count ? parseInt(count) : 0,
        username: username ?? 'anonymous',
      });
    } catch (error) {
      console.error(`API Init Error for post ${postId}:`, error);
      let errorMessage = 'Unknown error during initialization';
      if (error instanceof Error) {
        errorMessage = `Initialization failed: ${error.message}`;
      }
      res.status(400).json({ status: 'error', message: errorMessage });
    }
  }
);

router.post<{ postId: string }, IncrementResponse | { status: string; message: string }, unknown>(
  '/api/increment',
  async (_req, res): Promise<void> => {
    const { postId } = context;
    if (!postId) {
      res.status(400).json({
        status: 'error',
        message: 'postId is required',
      });
      return;
    }

    res.json({
      count: await redis.incrBy('count', 1),
      postId,
      type: 'increment',
    });
  }
);

router.post<{ postId: string }, DecrementResponse | { status: string; message: string }, unknown>(
  '/api/decrement',
  async (_req, res): Promise<void> => {
    const { postId } = context;
    if (!postId) {
      res.status(400).json({
        status: 'error',
        message: 'postId is required',
      });
      return;
    }

    res.json({
      count: await redis.incrBy('count', -1),
      postId,
      type: 'decrement',
    });
  }
);

router.post<{ postId: string }, YouTubeAnalysisResponse | YouTubeAnalysisError, YouTubeAnalysisRequest>(
  '/api/youtube-analyze',
  async (req, res): Promise<void> => {
    const { postId } = context;
    if (!postId) {
      res.status(400).json({
        status: 'error',
        message: 'postId is required',
      });
      return;
    }

    const { url } = req.body;
    if (!url) {
      res.status(400).json({
        status: 'error',
        message: 'YouTube URL is required',
      });
      return;
    }

    try {
      // Get OpenAI API key from Devvit settings
      const openaiKey = await settings.get('OPENAI_API_KEY');
      if (!openaiKey) {
        res.status(500).json({
          status: 'error',
          message: 'OpenAI API key not configured. Please set OPENAI_API_KEY in app settings.',
        });
        return;
      }

      // Extract video ID from YouTube URL
      const videoId = extractVideoId(url);
      if (!videoId) {
        res.status(400).json({
          status: 'error',
          message: 'Invalid YouTube URL format',
        });
        return;
      }

      // Get video metadata (optional)
      const metadata = await getVideoMetadata(videoId);
      
      // Analyze video content using OpenAI
      const analysisResult = await analyzeVideo(videoId, openaiKey);
      
      if (!analysisResult.success) {
        res.status(400).json({
          status: 'error',
          message: analysisResult.error || 'Failed to analyze video',
        });
        return;
      }

      // Extract toxicity score from analysis (simple regex to find score)
      const toxicityMatch = analysisResult.analysis?.match(/Toxicity Score:\s*(\d+)\/10/);
      const toxicityScore = toxicityMatch ? parseInt(toxicityMatch[1]) / 10 : 0.5;

      const response: YouTubeAnalysisResponse = {
        type: 'youtube_analysis',
        postId,
        title: metadata?.title || 'Unknown Title',
        channel: metadata?.channelTitle || 'Unknown Channel',
        duration: 'Unknown Duration',
        views: metadata?.viewCount || 'Unknown',
        toxicityScore,
        summary: analysisResult.analysis,
      };

      res.json(response);
    } catch (error) {
      console.error('YouTube analysis error:', error);
      res.status(500).json({
        status: 'error',
        message: 'Failed to analyze YouTube video',
      });
    }
  }
);

router.post('/internal/on-app-install', async (_req, res): Promise<void> => {
  try {
    const post = await createPost();

    res.json({
      status: 'success',
      message: `Post created in subreddit ${context.subredditName} with id ${post.id}`,
    });
  } catch (error) {
    console.error(`Error creating post: ${error}`);
    res.status(400).json({
      status: 'error',
      message: 'Failed to create post',
    });
  }
});

router.post('/internal/menu/post-create', async (_req, res): Promise<void> => {
  try {
    const post = await createPost();

    res.json({
      navigateTo: `https://reddit.com/r/${context.subredditName}/comments/${post.id}`,
    });
  } catch (error) {
    console.error(`Error creating post: ${error}`);
    res.status(400).json({
      status: 'error',
      message: 'Failed to create post',
    });
  }
});

// Use router middleware
app.use(router);

// Get port from environment variable with fallback
const port = getServerPort();

const server = createServer(app);
server.on('error', (err) => console.error(`server error; ${err.stack}`));
server.listen(port);
