export interface CompletionRequest {
  prompt: string;
  max_tokens?: number;
}

export interface CompletionChoice {
  text: string;
  index: number;
  finish_reason: string;
}

export interface CompletionResponse {
  id: string;
  object: string;
  choices: CompletionChoice[];
}

export interface ModelInfo {
  eff_depth: number;
  n_params: number;
}
