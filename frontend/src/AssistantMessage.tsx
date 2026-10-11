import { useState } from "react";

import { AnswerText } from "./AnswerText";
import { Sources } from "./Sources";
import { ToolCalls } from "./ToolCalls";
import type { Source, ToolCall } from "./types";

interface Props {
  content: string;
  sources: Source[];
  toolCalls: ToolCall[];
}

export function AssistantMessage({ content, sources, toolCalls }: Props) {
  const [highlighted, setHighlighted] = useState<number | null>(null);
  return (
    <>
      <AnswerText text={content} sources={sources} onHighlight={setHighlighted} />
      <ToolCalls calls={toolCalls} />
      <Sources sources={sources} highlighted={highlighted} />
    </>
  );
}
