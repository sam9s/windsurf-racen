"use client";

import { memo, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import styles from "./ChatMessage.module.css";
import type { ChatMessage as ChatMessageType } from "@/lib/types";
import ResultGrid from "./ResultGrid";

interface ChatMessageProps {
  message: ChatMessageType;
}

function ChatMessage({ message }: ChatMessageProps) {
  const isAssistant = message.role === "assistant";
  const wrapperClass = `${styles.wrapper} ${isAssistant ? styles.assistant : styles.user}`;
  const payload = message.payload;

  return (
    <div className={wrapperClass}>
      <div className={styles.bubble}>
        {isAssistant ? (
          <div className={styles.summary}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ href, children }: { href?: string; children?: ReactNode }) => (
                  // eslint-disable-next-line jsx-a11y/anchor-has-content
                  <a href={href ?? "#"} target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        ) : (
          <p className={styles.text}>{message.content}</p>
        )}
        {isAssistant && payload ? (
          <ResultGrid items={payload.items} count={payload.count} meta={payload.meta} />
        ) : null}
      </div>
    </div>
  );
}

export default memo(ChatMessage);
