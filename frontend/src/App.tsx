import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";

import { ChatError, MESSAGE_LIMIT, sendChat } from "./api";
import { AssistantMessage } from "./AssistantMessage";
import { ErrorNotice } from "./ErrorNotice";
import type { Message } from "./types";

const EXAMPLES = [
  "Peynirler açıldıktan sonra nasıl saklanmalı?",
  "Eski kaşar alerjen içeriyor mu?",
  "Köy tereyağının fiyatı nedir?",
  "Ürünleri iade edebilir miyim?",
];

interface Failure {
  text: string;
  message: string;
}

export function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const nextId = useRef(1);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [messages, pending, failure]);

  /** Send `text`; `prior` is the conversation before it (the last 10 go to the server). */
  async function send(text: string, prior: Message[]) {
    const question: Message = { id: nextId.current++, role: "user", content: text };
    setMessages([...prior, question]);
    setFailure(null);
    setPending(true);
    try {
      const reply = await sendChat(text, prior.map((m) => ({ role: m.role, content: m.content })));
      setMessages((current) => [
        ...current,
        {
          id: nextId.current++,
          role: "assistant",
          content: reply.answer,
          sources: reply.sources,
          toolCalls: reply.tool_calls,
        },
      ]);
    } catch (error) {
      const message = error instanceof ChatError ? error.message : "Bir sorun oluştu. Lütfen tekrar deneyin.";
      setFailure({ text, message });
    } finally {
      setPending(false);
      inputRef.current?.focus();
    }
  }

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || pending) return;
    setInput("");
    void send(trimmed, messages);
  }

  function retry() {
    if (!failure || pending) return;
    // The failed question is the last message; resend it with the same history.
    void send(failure.text, messages.slice(0, -1));
  }

  function clear() {
    setMessages([]);
    setFailure(null);
    setInput("");
    inputRef.current?.focus();
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit(input);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    submit(input);
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Ürün Asistanı</h1>
        <button type="button" className="secondary" onClick={clear} disabled={messages.length === 0 && !failure}>
          Sohbeti temizle
        </button>
      </header>

      <main className="chat" aria-label="Sohbet">
        {messages.length === 0 && !pending && !failure && (
          <div className="welcome">
            <p>
              Merhaba! Ürünlerimiz, saklama koşulları, alerjenler, fiyat, stok ve teslimat hakkında
              soru sorabilirsiniz.
            </p>
            <div className="examples" role="group" aria-label="Örnek sorular">
              {EXAMPLES.map((q) => (
                <button key={q} type="button" className="example" onClick={() => submit(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <ul className="messages" aria-live="polite">
          {messages.map((m) => (
            <li key={m.id} className={`bubble ${m.role}`}>
              <span className="visually-hidden">{m.role === "user" ? "Siz: " : "Asistan: "}</span>
              {m.role === "user" ? (
                <p className="answer-text">{m.content}</p>
              ) : (
                <AssistantMessage content={m.content} sources={m.sources} toolCalls={m.toolCalls} />
              )}
            </li>
          ))}
          {pending && (
            <li className="bubble assistant loading" role="status">
              <span className="dots" aria-hidden="true"><i /><i /><i /></span>
              Yanıt hazırlanıyor…
            </li>
          )}
        </ul>

        {failure && <ErrorNotice message={failure.message} onRetry={retry} />}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <label htmlFor="message" className="visually-hidden">Mesajınız</label>
        <textarea
          id="message"
          ref={inputRef}
          rows={1}
          maxLength={MESSAGE_LIMIT}
          placeholder="Bir soru yazın… (Enter gönderir, Shift+Enter yeni satır)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button type="submit" disabled={pending || input.trim() === ""}>Gönder</button>
      </form>
    </div>
  );
}
