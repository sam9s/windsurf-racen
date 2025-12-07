import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RACEN – iPhone Pricing Console",
  description: "Internal console for managing iPhone pricing and specs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
