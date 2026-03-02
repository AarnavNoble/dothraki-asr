export interface TranslationWord {
  original: string;
  dothraki: string | null;
  english: string | null;
  confidence: number;
}

export interface ClipMatch {
  clip_id: string;
  dothraki: string;
  english: string;
  score: number;
  audio_file?: string;
  dtw_cost?: number;
}

export interface PipelineResult {
  strategy: string;
  quality: string;
  raw_dothraki: string | null;
  clip_matches: ClipMatch[] | null;
  transcription: {
    text: string;
    language: string | null;
    model: string;
  } | null;
  translation: {
    text: string;
    words: TranslationWord[];
  } | null;
  match_results?: Record<string, unknown>[];
}
