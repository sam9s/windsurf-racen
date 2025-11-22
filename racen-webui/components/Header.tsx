"use client";

import { ReactNode } from "react";

import styles from "./Header.module.css";

interface HeaderProps {
  title?: string;
  caption?: string;
  children?: ReactNode;
}

export default function Header({
  title = "R . A . C . E . N",
  caption = "RAPID AUTOMATION CUSTOMER ENGAGEMENT NETWORK",
  children
}: HeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.logoWrapper}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/assets/RAVEN_logo.png"
          alt="RACEN logo"
          className={styles.logo}
          onError={(event) => {
            event.currentTarget.src = "/assets/RAVEN_logo.svg";
          }}
        />
      </div>
      <div className={styles.textBlock}>
        <h1 className={styles.title}>{title}</h1>
        <p className={styles.caption}>{caption}</p>
        {children ? <div className={styles.extra}>{children}</div> : null}
      </div>
    </header>
  );
}
