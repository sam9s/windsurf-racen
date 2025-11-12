<snippet lineStart=1 lineEnd=239 totalLineCount=239>
import React, { useState, useRef, useEffect } from "react";
import { Send, FileText, User, PhoneCall, AlertCircle, CheckCircle2, Clock, Package } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  citations?: { title: string; source: string }[];
  orderInfo?: {
    orderId: string;
    status: string;
    items: string[];
    deliveryDate?: string;
  };
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hello! I'm RACEN, your GREST service assistant. I can help you with order status, warranty claims, returns, and answer questions from our policies. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    // Simulate assistant response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I'm currently in demo mode. In production, I'll retrieve answers from verified GREST policies and provide citations for every response.",
        timestamp: new Date(),
        citations: [
          { title: "GREST Warranty Policy v2.3", source: "docs/warranty-policy.pdf" },
          { title: "Returns & Refunds FAQ", source: "kb/returns-faq" },
        ],
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsTyping(false);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const StatusBadge = ({ status }: { status: string }) => {
    const styles = {
      delivered: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400",
      shipped: "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400",
      processing: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400",
    };
    
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${styles[status as keyof typeof styles] || styles.processing}`}>
        {status === "delivered" && <CheckCircle2 className="w-3 h-3" />}
        {status === "shipped" && <Package className="w-3 h-3" />}
        {status === "processing" && <Clock className="w-3 h-3" />}
        {status}
      </span>
    );
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <span className="text-primary-foreground font-semibold text-lg">R</span>
            </div>
            <div>
              <h1 className="font-semibold text-foreground">RACEN Service Agent</h1>
              <p className="text-xs text-muted-foreground">Grounded answers · Order help · Warranty support</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 text-sm font-medium text-foreground hover:bg-accent rounded-md transition-colors flex items-center gap-1.5">
              <PhoneCall className="w-4 h-4" />
              Talk to human
            </button>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
          {messages.map((message) => (
            <div key={message.id} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              {message.role === "assistant" && (
                <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                  <span className="text-primary-foreground text-sm font-medium">R</span>
                </div>
              )}
              
              <div className={`flex flex-col gap-2 max-w-2xl ${message.role === "user" ? "items-end" : "items-start"}`}>
                <div className={`px-4 py-2.5 rounded-lg ${message.role === "user" ? "bg-primary text-primary-foreground" : "bg-card border border-border"}`}>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                </div>

                {/* Citations */}
                {message.citations && message.citations.length > 0 && (
                  <div className="bg-muted/50 border border-border rounded-md px-3 py-2 space-y-1.5">
                    <p className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                      <FileText className="w-3 h-3" />
                      Sources
                    </p>
                    {message.citations.map((citation, idx) => (
                      <div key={idx} className="text-xs">
                        <span className="font-medium text-foreground">{citation.title}</span>
                        <span className="text-muted-foreground ml-1.5 font-mono">{citation.source}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Order Info */}
                {message.orderInfo && (
                  <div className="bg-card border border-border rounded-md p-3 space-y-2 w-full">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-muted-foreground">Order #{message.orderInfo.orderId}</span>
                      <StatusBadge status={message.orderInfo.status} />
                    </div>
                    <div className="space-y-1">
                      {message.orderInfo.items.map((item, idx) => (
                        <p key={idx} className="text-sm text-foreground">{item}</p>
                      ))}
                    </div>
                    {message.orderInfo.deliveryDate && (
                      <p className="text-xs text-muted-foreground">Delivered: {message.orderInfo.deliveryDate}</p>
                    )}
                  </div>
                )}

                <span className="text-xs text-muted-foreground">
                  {message.timestamp.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>

              {message.role === "user" && (
                <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-muted-foreground" />
                </div>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="flex gap-3 items-start">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                <span className="text-primary-foreground text-sm font-medium">R</span>
              </div>
              <div className="bg-card border border-border rounded-lg px-4 py-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: "0ms" }}></div>
                  <div className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: "150ms" }}></div>
                  <div className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: "300ms" }}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Info Banner */}
      <div className="border-t border-border bg-muted/30">
        <div className="max-w-4xl mx-auto px-4 py-2 flex items-center gap-2 text-xs text-muted-foreground">
          <AlertCircle className="w-3.5 h-3.5" />
          <span>All answers are verified from GREST policies. Your data is protected and conversations are logged for quality.</span>
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about orders, warranty, returns, or policies..."
              className="flex-1 px-4 py-2.5 bg-background border border-input rounded-md text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="px-4 py-2.5 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              <span className="text-sm font-medium">Send</span>
            </button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Supports English, Hindi & Hinglish · Type your question or order number
          </p>
        </div>
      </div>
    </div>
  );
}