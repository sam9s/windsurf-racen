import { NextResponse } from "next/server";

export const runtime = "nodejs";

const backendBase = process.env.RACEN_BACKEND_BASE_URL;
const adminToken = process.env.RACEN_ADMIN_TOKEN;

export async function GET() {
  if (!backendBase || !adminToken) {
    return NextResponse.json(
      { detail: "Backend base URL or admin token is not configured." },
      { status: 500 },
    );
  }

  const base = backendBase.replace(/\/$/, "");
  const url = `${base}/admin/iphone-specs/list?token=${encodeURIComponent(adminToken)}`;

  try {
    const res = await fetch(url, { method: "GET" });
    const text = await res.text();

    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
    }

    if (!res.ok) {
      const payload =
        data && typeof data === "object"
          ? data
          : { detail: `Upstream error ${res.status}` };
      return NextResponse.json(payload, { status: res.status });
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : `Failed to reach backend: ${String(error)}`;
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}
