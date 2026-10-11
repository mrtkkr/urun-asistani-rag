export interface Source {
  n: number;
  type: "product" | "faq";
  id: string;
  title: string;
  score: number;
  cited: boolean;
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  result: Record<string, unknown>;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  tool_calls: ToolCall[];
}

export type Message =
  | { id: number; role: "user"; content: string }
  | { id: number; role: "assistant"; content: string; sources: Source[]; toolCalls: ToolCall[] };

export interface HistoryItem {
  role: "user" | "assistant";
  content: string;
}
