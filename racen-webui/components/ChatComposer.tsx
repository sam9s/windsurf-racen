"use client";

import { FormEvent, useState } from "react";

import styles from "./ChatComposer.module.css";

interface ChatComposerProps {
  disabled?: boolean;
  onSubmit(message: string): void | Promise<void>;
}

export default function ChatComposer({ disabled = false, onSubmit }: ChatComposerProps) {
  const [value, setValue] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) {
      return;
    }
    setValue("");
    await onSubmit(trimmed);
  }

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <input
        className={styles.input}
        placeholder="Welcome to RACEN - Ask -> What Can I Do"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        disabled={disabled}
        autoComplete="off"
      />
      <button type="submit" className={styles.submit} disabled={disabled}>
        <span>›</span>
      </button>
    </form>
  );
}
