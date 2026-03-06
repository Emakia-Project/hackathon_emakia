// src/services/analyzer.ts
import { Devvit } from '@devvit/public-api';

// Extract video ID from various YouTube URL formats
export function extractVideoId(url: string): string | null {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
    /^([a-zA-Z0-9_-]{11})$/
  ];
  
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

// Fetch YouTube transcript via multiple methods
async function fetchTranscript(videoId: string): Promise<string> {
  const methods = [
    // Method 1: Primary transcript API with better error handling
    async () => {
      const response = await fetch(
        `https://youtube-transcript-api.vercel.app/api/transcript?videoId=${videoId}`
      );
      if (!response.ok) throw new Error('Method 1 failed');
      
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Response is not JSON');
      }
      
      const data = await response.json();
      if (data.transcript && Array.isArray(data.transcript)) {
        return data.transcript.map((t: any) => t.text).join(' ');
      }
      throw new Error('Invalid transcript format');
    },
    
    // Method 2: Alternative transcript API with error handling
    async () => {
      const response = await fetch(
        `https://www.youtube.com/api/timedtext?v=${videoId}&lang=en&fmt=json3`
      );
      if (!response.ok) throw new Error('Method 2 failed');
      
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Response is not JSON');
      }
      
      const data = await response.json();
      if (data.events && Array.isArray(data.events)) {
        return data.events.map((e: any) => e.segs?.map((s: any) => s.utf8).join('')).join(' ');
      }
      throw new Error('Invalid transcript format');
    },
    
    // Method 3: Try a different transcript service
    async () => {
      const response = await fetch(
        `https://youtubetranscript.com/?server_vid2=${videoId}`
      );
      if (!response.ok) throw new Error('Method 3 failed');
      
      const text = await response.text();
      if (text.includes('transcript') || text.includes('caption')) {
        return text;
      }
      throw new Error('No transcript found');
    },
    
    // Method 4: Fallback - return a generic analysis prompt
    async () => {
      return `This video does not have captions available. Please analyze based on the video title and description. Video ID: ${videoId}`;
    }
  ];
  
  for (let i = 0; i < methods.length; i++) {
    try {
      const transcript = await methods[i]();
      if (transcript && transcript.length > 10) {
        return transcript;
      }
    } catch (error) {
      console.log(`Transcript method ${i + 1} failed:`, error);
      if (i === methods.length - 1) {
        throw new Error('All transcript methods failed. Video may not have captions available.');
      }
    }
  }
  
  throw new Error('Failed to fetch transcript. Video may not have captions available.');
}

// Analyze video using OpenAI
export async function analyzeVideo(
  videoId: string,
  apiKey: string
): Promise<{
  success: boolean;
  analysis?: string;
  error?: string;
  transcriptLength?: number;
}> {
  try {
    // Get transcript
    let transcript: string;
    let hasTranscript = true;
    
    try {
      transcript = await fetchTranscript(videoId);
    } catch (error) {
      // If transcript fails, try to get video metadata for analysis
      console.log('Transcript not available, trying metadata analysis:', error);
      hasTranscript = false;
      
      // Get video metadata as fallback
      try {
        const metadata = await getVideoMetadata(videoId, apiKey);
        if (metadata) {
          transcript = `Video Title: ${metadata.title}\nChannel: ${metadata.channelTitle}\nViews: ${metadata.viewCount}\nPublished: ${metadata.publishedAt}\n\nThis video does not have captions available. Please provide a general analysis based on the title and channel information.`;
        } else {
          transcript = `Video ID: ${videoId}\n\nThis video does not have captions available and metadata could not be retrieved. Please provide a general analysis based on the video ID.`;
        }
      } catch (metadataError) {
        console.log('Metadata also failed:', metadataError);
        transcript = `Video ID: ${videoId}\n\nThis video does not have captions available and metadata could not be retrieved. Please provide a general analysis based on the video ID.`;
      }
    }
    
    if (!transcript || transcript.length < 10) {
      return {
        success: false,
        error: 'Unable to retrieve video content for analysis'
      };
    }
    
    // Truncate transcript to fit token limits
    const truncatedTranscript = transcript.slice(0, 12000);
    
    // Call OpenAI API
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini', // More cost-effective than gpt-4
        messages: [
          {
            role: 'system',
            content: `You are an expert content moderator analyzing YouTube video content. Provide analysis in the following structured format:

**Toxicity Score:** X/10
Brief explanation of toxicity level.

**Bias Assessment:**
- Identify any political, cultural, or ideological biases
- Note balanced vs one-sided perspectives

**Misinformation Detection:**
- Flag any claims that appear factually dubious
- Note unverified statements presented as facts

**Overall Summary:**
A brief 2-3 sentence overview of content quality and concerns.

${hasTranscript ? 'Analyze the following video transcript:' : 'Analyze the following video information (transcript not available):'}`
          },
          {
            role: 'user',
            content: `${hasTranscript ? 'Video Transcript:' : 'Video Information:'}\n\n${truncatedTranscript}`
          }
        ],
        temperature: 0.3,
        max_tokens: 800
      })
    });
    
    if (!response.ok) {
      let errorMessage = 'OpenAI API request failed';
      try {
        const errorData = await response.json();
        errorMessage = errorData.error?.message || errorMessage;
      } catch (jsonError) {
        const errorText = await response.text();
        errorMessage = `HTTP ${response.status}: ${errorText}`;
      }
      throw new Error(errorMessage);
    }
    
    let data;
    try {
      data = await response.json();
    } catch (jsonError) {
      throw new Error('Invalid JSON response from OpenAI API');
    }
    
    return {
      success: true,
      analysis: data.choices[0].message.content,
      transcriptLength: hasTranscript ? transcript.length : 0
    };
    
  } catch (error) {
    console.error('Analysis error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
}

// Get video metadata from YouTube API (optional enhancement)
export async function getVideoMetadata(videoId: string, apiKey?: string) {
  try {
    if (!apiKey) {
      return null;
    }
    
    const response = await fetch(
      `https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id=${videoId}&key=${apiKey}`
    );
    
    const data = await response.json();
    
    if (data.items && data.items.length > 0) {
      const video = data.items[0];
      return {
        title: video.snippet.title,
        channelTitle: video.snippet.channelTitle,
        publishedAt: video.snippet.publishedAt,
        viewCount: video.statistics.viewCount,
        likeCount: video.statistics.likeCount
      };
    }
    
    return null;
  } catch (error) {
    console.error('Failed to fetch video metadata:', error);
    return null;
  }
}