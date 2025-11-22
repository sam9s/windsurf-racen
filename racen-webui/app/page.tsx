"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import Header from "@/components/Header";
import ChatComposer from "@/components/ChatComposer";
import ChatMessage from "@/components/ChatMessage";
import { answerQuestion } from "@/lib/api";
import type { ChatMessage as ChatMessageType, AnswerResponse } from "@/lib/types";

// styles removed with slider; keep minimal shell CSS from globals.css

function stripFiller(text: string): string {
  return text
    .replace(/\b(please|kindly|thanks|thank you)\b/gi, "")
    .replace(/\b(hey|hi|hello)\b/gi, "")
    .replace(/\b(?:how\s+about|what\s+about|about|maybe)\b/gi, "")
    .replace(/\b(?:ok|okay|alright)\b/gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

function createId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function augmentWithContext(text: string, _meta?: any): string {
  // Simple cleaner for now; brand/catalogue context removed for RACEN UI
  return stripFiller(text);
}

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [pending, setPending] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, pending]);

  // No brand badge in RACEN UI

  const handleSubmit = useCallback(
    async (text: string) => {
      const userMessage: ChatMessageType = {
        id: createId(),
        role: "user",
        content: text,
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setPending(true);

      const augmentedText = augmentWithContext(text, undefined);
      const finalQuestion = (augmentedText || "").trim() || text.trim();

      if (!finalQuestion) {
        const assistantMessage: ChatMessageType = {
          id: createId(),
          role: "assistant",
          content: "Please type a question (e.g., 'What is your shipping policy?')",
          createdAt: Date.now(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setPending(false);
        return;
      }

      try {
        const started = performance.now();
        const res: AnswerResponse = await answerQuestion({ question: finalQuestion, k: 6 });
        const elapsed = Math.round(performance.now() - started);
        const parts = [res.answer, "", `Latency: ${elapsed} ms`];

        const assistantMessage: ChatMessageType = {
          id: createId(),
          role: "assistant",
          content: parts.join("\n"),
          createdAt: Date.now(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (error: unknown) {
        const detail = error instanceof Error ? error.message : "Answer request failed.";
        const assistantMessage: ChatMessageType = {
          id: createId(),
          role: "assistant",
          content: `Request failed: ${detail}`,
          createdAt: Date.now(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } finally {
        setPending(false);
      }
    },
    []
  );

  return (
    <div className="page-shell">
      <Header />
      <main className="main-area">
        <div className="scroll-region" ref={scrollRef}>
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {pending ? <div className="placeholder">Thinking…</div> : null}
        </div>
        <ChatComposer disabled={pending} onSubmit={handleSubmit} />
      </main>
    </div>
  );
}
