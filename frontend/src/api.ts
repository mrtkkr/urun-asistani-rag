import type { ChatResponse, HistoryItem } from "./types";

export const HISTORY_LIMIT = 10;
export const MESSAGE_LIMIT = 500;
const HISTORY_ITEM_LIMIT = 2000;

export const NETWORK_ERROR = "Sunucuya ulaşılamadı. Bağlantınızı kontrol edip tekrar deneyin.";
const GENERIC_ERROR = "Bir sorun oluştu. Lütfen tekrar deneyin.";
const THROTTLED_ERROR = "Çok fazla istek gönderildi. Lütfen biraz bekleyip tekrar deneyin.";

/** A failure with a message that is safe to show the user as is. */
export class ChatError extends Error {}

export async function sendChat(message: string, history: HistoryItem[]): Promise<ChatResponse> {
  const body = {
    message,
    history: history.slice(-HISTORY_LIMIT).map((h) => ({
      role: h.role,
      content: h.content.slice(0, HISTORY_ITEM_LIMIT),
    })),
  };
  let response: Response;
  try {
    response = await fetch("/api/chat/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ChatError(NETWORK_ERROR);
  }
  if (response.ok) return (await response.json()) as ChatResponse;

  // 503 carries the server's own Turkish message; other statuses get fixed texts.
  if (response.status === 503) {
    const data = await response.json().catch(() => null);
    if (data && typeof data.detail === "string") throw new ChatError(data.detail);
  }
  throw new ChatError(response.status === 429 ? THROTTLED_ERROR : GENERIC_ERROR);
}
