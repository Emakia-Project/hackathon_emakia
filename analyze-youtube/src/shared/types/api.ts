export type InitResponse = {
  type: 'init';
  postId: string;
  count: number;
  username: string;
};

export type IncrementResponse = {
  type: 'increment';
  postId: string;
  count: number;
};

export type DecrementResponse = {
  type: 'decrement';
  postId: string;
  count: number;
};

export type YouTubeAnalysisRequest = {
  url: string;
};

export type YouTubeAnalysisResponse = {
  type: 'youtube_analysis';
  postId: string;
  title: string;
  channel: string;
  duration: string;
  views: string;
  toxicityScore: number;
  summary?: string;
};

export type YouTubeAnalysisError = {
  status: 'error';
  message: string;
};
