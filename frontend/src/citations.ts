export type Segment = { kind: "text"; text: string } | { kind: "cite"; n: number };

const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** Split an answer into plain text and citation markers; "[1, 2]" becomes two markers. */
export function parseCitations(text: string): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    if (match.index > last) segments.push({ kind: "text", text: text.slice(last, match.index) });
    for (const n of match[1].split(",")) segments.push({ kind: "cite", n: Number(n) });
    last = match.index + match[0].length;
  }
  if (last < text.length) segments.push({ kind: "text", text: text.slice(last) });
  return segments;
}
