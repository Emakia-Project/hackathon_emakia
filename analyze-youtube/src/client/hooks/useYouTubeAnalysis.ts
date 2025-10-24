import { useCallback, useState } from 'react';
import type { YouTubeAnalysisResponse, YouTubeAnalysisError } from '../../shared/types/api';

interface YouTubeAnalysisState {
  youtubeUrl: string;
  analysisResult: YouTubeAnalysisResponse | null;
  isAnalyzing: boolean;
  error: string | null;
}

export const useYouTubeAnalysis = () => {
  const [state, setState] = useState<YouTubeAnalysisState>({
    youtubeUrl: '',
    analysisResult: null,
    isAnalyzing: false,
    error: null,
  });

  const setYoutubeUrl = useCallback((url: string) => {
    setState(prev => ({ ...prev, youtubeUrl: url, error: null }));
  }, []);

  const analyzeVideo = useCallback(async () => {
    if (!state.youtubeUrl) {
      setState(prev => ({ ...prev, error: 'Please enter a YouTube URL' }));
      return;
    }

    setState(prev => ({ ...prev, isAnalyzing: true, error: null }));

    try {
      const response = await fetch('/api/youtube-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: state.youtubeUrl }),
      });

      if (!response.ok) {
        const errorData: YouTubeAnalysisError = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}`);
      }

      const data: YouTubeAnalysisResponse = await response.json();
      setState(prev => ({ 
        ...prev, 
        analysisResult: data, 
        isAnalyzing: false 
      }));
    } catch (err) {
      console.error('Failed to analyze YouTube video', err);
      setState(prev => ({ 
        ...prev, 
        isAnalyzing: false, 
        error: err instanceof Error ? err.message : 'Failed to analyze video' 
      }));
    }
  }, [state.youtubeUrl]);

  return {
    ...state,
    setYoutubeUrl,
    analyzeVideo,
  } as const;
};
