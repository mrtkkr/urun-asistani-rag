import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { parseCitations } from "./citations";
import { Sources } from "./Sources";
import type { ChatResponse, Source } from "./types";

const source = (over: Partial<Source>): Source => ({
  n: 1, type: "product", id: "kasar-eski", title: "Eski Kaşar", score: 0.91, cited: false, ...over,
});

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

afterEach(() => vi.unstubAllGlobals());

describe("parseCitations", () => {
  it("splits [n] markers from text and expands lists", () => {
    expect(parseCitations("Soğukta saklayın [1]. Alerjen: süt [2, 3]")).toEqual([
      { kind: "text", text: "Soğukta saklayın " },
      { kind: "cite", n: 1 },
      { kind: "text", text: ". Alerjen: süt " },
      { kind: "cite", n: 2 },
      { kind: "cite", n: 3 },
    ]);
  });

  it("leaves text without markers alone", () => {
    expect(parseCitations("[a] ve [] düz metin")).toEqual([{ kind: "text", text: "[a] ve [] düz metin" }]);
  });
});

describe("Sources", () => {
  it("marks cited sources and fades the others", () => {
    render(
      <Sources
        sources={[
          source({ n: 1, title: "Eski Kaşar", cited: true }),
          source({ n: 2, type: "faq", id: "iade", title: "İade", score: 0.7, cited: false }),
        ]}
      />,
    );
    const [first, second] = screen.getAllByRole("listitem");
    expect(first).toHaveAttribute("data-cited", "true");
    expect(first).toHaveClass("cited");
    expect(within(first).getByText("Ürün")).toBeInTheDocument();
    expect(within(first).getByText("0,91")).toBeInTheDocument();
    expect(second).toHaveAttribute("data-cited", "false");
    expect(second).toHaveClass("uncited");
    expect(within(second).getByText("SSS")).toBeInTheDocument();
  });
});

describe("App", () => {
  it("renders [n] as a badge that highlights its source", async () => {
    const reply: ChatResponse = {
      answer: "Soğukta saklanır [1].\nİkinci satır.",
      sources: [source({ cited: true })],
      tool_calls: [],
    };
    vi.stubGlobal("fetch", vi.fn(() => json(reply)));
    render(<App />);
    await userEvent.type(screen.getByLabelText("Mesajınız"), "saklama?{Enter}");

    const badge = await screen.findByRole("button", { name: "Kaynak 1" });
    await userEvent.hover(badge);
    expect(document.querySelector(".source")).toHaveAttribute("data-highlighted", "true");
    await userEvent.unhover(badge);
    expect(document.querySelector(".source")).toHaveAttribute("data-highlighted", "false");
  });

  it("shows the server's message on 503 and retries with the same question", async () => {
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(() => json({ detail: "Asistan şu anda yanıt veremiyor." }, 503))
      .mockImplementationOnce(() => json({ answer: "Tamam.", sources: [], tool_calls: [] }));
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    await userEvent.type(screen.getByLabelText("Mesajınız"), "merhaba{Enter}");

    expect(await screen.findByRole("alert")).toHaveTextContent("Asistan şu anda yanıt veremiyor.");
    await userEvent.click(screen.getByRole("button", { name: "Tekrar dene" }));

    expect(await screen.findByText("Tamam.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getAllByText("merhaba")).toHaveLength(1); // not duplicated by the retry
    const bodies = fetchMock.mock.calls.map(([, init]) => JSON.parse(init.body));
    expect(bodies).toEqual([
      { message: "merhaba", history: [] },
      { message: "merhaba", history: [] },
    ]);
  });

  it("shows a generic message on network failure, with retry", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    render(<App />);
    await userEvent.type(screen.getByLabelText("Mesajınız"), "merhaba{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Sunucuya ulaşılamadı");
    expect(screen.getByRole("button", { name: "Tekrar dene" })).toBeInTheDocument();
  });

  it("sends only the last 10 messages as history", async () => {
    const fetchMock = vi.fn(() => json({ answer: "Cevap.", sources: [], tool_calls: [] }));
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    const box = screen.getByLabelText("Mesajınız");
    for (let i = 1; i <= 7; i++) {
      await userEvent.type(box, `soru ${i}{Enter}`);
      await screen.findAllByText("Cevap.").then((all) => expect(all).toHaveLength(i));
    }
    const last = JSON.parse((fetchMock.mock.calls.at(-1) as unknown as [string, { body: string }])[1].body);
    expect(last.history).toHaveLength(10);
    expect(last.history[0]).toEqual({ role: "user", content: "soru 2" });
  });

  it("renders model output as text, not HTML", async () => {
    const answer = "<img src=x onerror=alert(1)> merhaba";
    vi.stubGlobal("fetch", vi.fn(() => json({ answer, sources: [], tool_calls: [] })));
    render(<App />);
    await userEvent.type(screen.getByLabelText("Mesajınız"), "x{Enter}");
    expect(await screen.findByText(answer)).toBeInTheDocument();
    expect(document.querySelector("img")).toBeNull();
  });
});
