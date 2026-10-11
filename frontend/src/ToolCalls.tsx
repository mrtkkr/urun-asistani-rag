import type { ToolCall } from "./types";

const money = (value: unknown) =>
  `${Number(value).toLocaleString("tr-TR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} TL`;

function summarizeResult(call: ToolCall): string {
  const r = call.result;
  if (typeof r.error === "string") return "ürün bulunamadı";
  if (call.name === "get_price" && r.price !== undefined) return money(r.price);
  if (call.name === "check_stock" && typeof r.in_stock === "boolean") {
    return r.in_stock ? `stokta, ${r.stock} adet` : "stokta yok";
  }
  return JSON.stringify(r);
}

const summarizeArgs = (args: ToolCall["args"]) => {
  const values = Object.values(args);
  return values.length === 1 && typeof values[0] === "string" ? values[0] : JSON.stringify(args);
};

export function ToolCalls({ calls }: { calls: ToolCall[] }) {
  if (calls.length === 0) return null;
  return (
    <div className="tool-calls">
      {calls.map((call, i) => (
        <details key={i} className="tool-call">
          <summary>
            <span className="tool-name">{call.name}</span> · {summarizeArgs(call.args)} → {summarizeResult(call)}
          </summary>
          <dl>
            <dt>Araç</dt>
            <dd>{call.name}</dd>
            <dt>Argüman</dt>
            <dd><pre>{JSON.stringify(call.args, null, 2)}</pre></dd>
            <dt>Sonuç</dt>
            <dd><pre>{JSON.stringify(call.result, null, 2)}</pre></dd>
          </dl>
        </details>
      ))}
    </div>
  );
}
