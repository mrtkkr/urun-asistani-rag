import type { Source } from "./types";

const TYPE_LABEL: Record<Source["type"], string> = { product: "Ürün", faq: "SSS" };

const formatScore = (score: number) =>
  score.toLocaleString("tr-TR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

interface Props {
  sources: Source[];
  highlighted?: number | null;
}

export function Sources({ sources, highlighted = null }: Props) {
  if (sources.length === 0) return null;
  return (
    <section className="sources" aria-label="Kaynaklar">
      <h3>Kaynaklar</h3>
      <ol>
        {sources.map((s) => (
          <li
            key={s.n}
            className={`source ${s.cited ? "cited" : "uncited"}${highlighted === s.n ? " highlighted" : ""}`}
            data-cited={s.cited}
            data-highlighted={highlighted === s.n}
          >
            <span className="source-n">{s.n}</span>
            <span className="source-title">{s.title}</span>
            <span className="source-type">{TYPE_LABEL[s.type]}</span>
            <span className="source-score" title="Benzerlik skoru">{formatScore(s.score)}</span>
            {s.cited && <span className="visually-hidden">Yanıtta atıf yapıldı</span>}
          </li>
        ))}
      </ol>
    </section>
  );
}
