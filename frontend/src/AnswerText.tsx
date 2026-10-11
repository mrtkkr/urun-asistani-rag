import { parseCitations } from "./citations";
import type { Source } from "./types";

interface Props {
  text: string;
  sources: Source[];
  onHighlight: (n: number | null) => void;
}

/** Renders the answer as plain text (never HTML); [n] becomes a badge for a known source. */
export function AnswerText({ text, sources, onHighlight }: Props) {
  const known = new Set(sources.map((s) => s.n));
  return (
    <p className="answer-text">
      {parseCitations(text).map((segment, i) => {
        if (segment.kind === "text") return segment.text;
        if (!known.has(segment.n)) return `[${segment.n}]`;
        const n = segment.n;
        return (
          <button
            key={i}
            type="button"
            className="cite-badge"
            aria-label={`Kaynak ${n}`}
            onClick={() => onHighlight(n)}
            onMouseEnter={() => onHighlight(n)}
            onMouseLeave={() => onHighlight(null)}
            onFocus={() => onHighlight(n)}
            onBlur={() => onHighlight(null)}
          >
            {n}
          </button>
        );
      })}
    </p>
  );
}
