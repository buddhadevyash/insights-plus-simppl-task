"use client";
import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Send, User, Bot, Sparkles, RefreshCw, AlertCircle, Plus, Download, BarChart3, FileText, ChevronDown, TrendingUp, Target, Zap } from 'lucide-react';
import { cn } from "@/lib/utils";
import { Button } from '@/components/ui/button';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import html2canvas from 'html2canvas';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// --- NEW: Safe Color Palette ---
// We'll use these universally-supported hex codes to avoid the oklch() error.
const SAFE_COLORS = {
  primary: '#2563eb',
  primaryForeground: '#ffffff',
  card: '#ffffff',
  cardForeground: '#020817',
  background: '#f1f5f9', // A light gray for backgrounds
  muted: '#f1f5f9',
  mutedForeground: '#64748b',
  destructive: '#ef4444',
  border: '#e2e8f0',
  // Chart Colors
  chartBlue: '#3b82f6',
  chartGreen: '#22c55e',
  chartYellow: '#f59e0b',
  chartRed: '#ef4444',
  chartPurple: '#8b5cf6',
  // Stat Card Colors
  statBlueBg: '#dbeafe',
  statBlueText: '#1d4ed8',
  statGreenBg: '#dcfce7',
  statGreenText: '#166534',
  statOrangeBg: '#ffedd5',
  statOrangeText: '#9a3412',
};

// --- Constants ---
const API_ENDPOINT = process.env.FASTAPI;
const EXEMPLAR_PROMPTS = [
  "discussions regarding openai",
  "Generate a sales report for last quarter",
  "Summarize user engagement from the past month",
];
const MESSAGE_TYPES = {
  USER: 'user',
  BOT: 'bot',
  REPORT: 'report',
  ERROR: 'error'
};

// --- Charting Component (Updated with Safe Colors) ---
const ChartComponent = ({ chartData, chartId }) => {
    const data = useMemo(() => chartData.labels.map((label, index) => ({
        name: label,
        value: chartData.data[index],
    })), [chartData.labels, chartData.data]);

    // STYLE OVERRIDE: Using the safe color palette for charts
    const COLORS = useMemo(() => chartData.colors || [SAFE_COLORS.chartBlue, SAFE_COLORS.chartGreen, SAFE_COLORS.chartYellow, SAFE_COLORS.chartRed, SAFE_COLORS.chartPurple], [chartData.colors]);

    const renderChart = () => {
        switch (chartData.type) {
            case 'bar':
                return (
                    <BarChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-10} textAnchor="end" height={50} />
                        <YAxis />
                        <Tooltip /> <Legend />
                        <Bar dataKey="value">{data.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}</Bar>
                    </BarChart>
                );
            case 'pie':
                return (
                    <PieChart>
                        <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={120} label>{data.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}</Pie>
                        <Tooltip /><Legend />
                    </PieChart>
                );
            default: return <div className="text-center" style={{ color: SAFE_COLORS.mutedForeground }}>Unsupported chart type</div>;
        }
    };

    return (
        <div id={chartId} className="p-4 rounded-lg shadow-sm" style={{ backgroundColor: SAFE_COLORS.card, border: `1px solid ${SAFE_COLORS.border}` }}>
            <h4 className="font-semibold mb-4" style={{ color: SAFE_COLORS.cardForeground }}>{chartData.title}</h4>
            <div style={{ width: '100%', height: 400 }}>
                <ResponsiveContainer>{renderChart()}</ResponsiveContainer>
            </div>
            {chartData.insights && (
                <div className="mt-4 p-3 rounded-lg" style={{ backgroundColor: SAFE_COLORS.statBlueBg, border: `1px solid ${SAFE_COLORS.border}` }}>
                    <p className="text-sm" style={{ color: SAFE_COLORS.statBlueText }}><span className="font-medium">Insight:</span> {chartData.insights}</p>
                </div>
            )}
        </div>
    );
};

// --- UI Components for Report (Updated with Safe Colors) ---
const StatCard = ({ title, value, insight, icon: Icon, color = "blue" }) => {
    const colorStyles = {
        blue: { background: SAFE_COLORS.statBlueBg, text: SAFE_COLORS.statBlueText },
        green: { background: SAFE_COLORS.statGreenBg, text: SAFE_COLORS.statGreenText },
        orange: { background: SAFE_COLORS.statOrangeBg, text: SAFE_COLORS.statOrangeText },
    };
    return (
        <div className="p-6 rounded-xl shadow-sm h-full" style={{ backgroundColor: SAFE_COLORS.card, border: `1px solid ${SAFE_COLORS.border}` }}>
            <div className="flex items-center gap-3 mb-4">
                {Icon && <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ backgroundColor: colorStyles[color].background, color: colorStyles[color].text }}><Icon className="w-5 h-5" /></div>}
                <div>
                    <h3 className="text-sm font-semibold uppercase tracking-wider" style={{ color: SAFE_COLORS.mutedForeground }}>{title}</h3>
                    <p className="text-2xl font-bold" style={{ color: SAFE_COLORS.cardForeground }}>{value}</p>
                </div>
            </div>
            <p className="text-sm leading-relaxed" style={{ color: SAFE_COLORS.mutedForeground }}>{insight}</p>
        </div>
    );
};

const CollapsibleSection = ({ title, children, defaultOpen = true, icon: Icon }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <section className="rounded-xl shadow-sm overflow-hidden" style={{ backgroundColor: SAFE_COLORS.card, border: `1px solid ${SAFE_COLORS.border}` }}>
            <button onClick={() => setIsOpen(!isOpen)} className="w-full flex items-center justify-between p-6 text-left focus:outline-none">
                <div className="flex items-center gap-4">{Icon && <Icon className="w-6 h-6" style={{ color: SAFE_COLORS.primary }} />} <h2 className="text-xl font-bold" style={{ color: SAFE_COLORS.cardForeground }}>{title}</h2></div>
                <ChevronDown className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''}`} style={{ color: SAFE_COLORS.mutedForeground }} />
            </button>
            {isOpen && <div className="px-6 pb-6" style={{ borderTop: `1px solid ${SAFE_COLORS.border}`, backgroundColor: '#fafafa' }}><div className="pt-6">{children}</div></div>}
        </section>
    );
};

// --- Report Display Component (Updated with Safe Colors) ---
const ReportDisplay = ({ data }) => {
    const { summary, report, visualizations } = data || {};
    const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
    
    const handleDownload = useCallback(async () => {
        setIsGeneratingPdf(true);
        const filename = `InsightsPlus_Report_${new Date().toISOString().split('T')[0]}.pdf`;
        const doc = new jsPDF('p', 'mm', 'a4');
        let yPos = 15;

        try {
            doc.setFontSize(18); doc.text("InsightsPlus Analysis Report", 14, yPos); yPos += 15;
            if (summary) {
                doc.setFontSize(14); doc.text("Summary", 14, yPos); yPos += 7; doc.setFontSize(10);
                const summaryLines = doc.splitTextToSize(`Overview: ${summary.overview}\n\nKey Metrics: ${summary.key_metrics}\n\nPrimary Insight: ${summary.primary_insight}`, 180);
                doc.text(summaryLines, 14, yPos); yPos += summaryLines.length * 4 + 10;
            }
            if (report) {
                 if (yPos > 250) { doc.addPage(); yPos = 15; }
                 doc.setFontSize(14); doc.text("Detailed Report", 14, yPos); yPos += 7; doc.setFontSize(10);
                 doc.text("Key Findings:", 14, yPos); yPos += 5;
                 report.key_findings?.forEach(item => { const lines = doc.splitTextToSize(`- ${item}`, 170); doc.text(lines, 18, yPos); yPos += lines.length * 4 + 2; });
                 yPos += 4;
                 doc.text("Actionable Insights:", 14, yPos); yPos += 5;
                 report.actionable_insights?.forEach(item => { const lines = doc.splitTextToSize(`- ${item}`, 170); doc.text(lines, 18, yPos); yPos += lines.length * 4 + 2; });
            }
            if (visualizations && visualizations.length > 0) {
                if (yPos > 180) { doc.addPage(); yPos = 15; }
                doc.setFontSize(14); doc.text("Visualizations", 14, yPos); yPos += 10;
                for (const chart of visualizations) {
                    const chartElement = document.getElementById(`chart-container-${chart.id}`);
                    if (chartElement) {
                        const canvas = await html2canvas(chartElement, { scale: 2 });
                        const imgData = canvas.toDataURL('image/png');
                        const imgProps = doc.getImageProperties(imgData);
                        const pdfWidth = doc.internal.pageSize.getWidth();
                        const imgWidth = pdfWidth - 28;
                        const imgHeight = (imgProps.height * imgWidth) / imgProps.width;
                        if (yPos + imgHeight > 280) { doc.addPage(); yPos = 15; }
                        doc.addImage(imgData, 'PNG', 14, yPos, imgWidth, imgHeight);
                        yPos += imgHeight + 10;
                    }
                }
            }
            doc.save(filename);
        } catch (error) {
            console.error("Error generating PDF:", error);
            alert("Sorry, an error occurred while generating the PDF.");
        } finally {
            setIsGeneratingPdf(false);
        }
    }, [data, summary, report, visualizations]);

    if (!data || !summary || !report) return <div className="p-8 text-center rounded-lg" style={{ color: SAFE_COLORS.destructive, backgroundColor: '#fee2e2' }}>Invalid report data.</div>;

    return (
        <div className="rounded-lg p-4 sm:p-6 space-y-6 max-w-full" style={{ backgroundColor: SAFE_COLORS.background }}>
            <header className="flex justify-between items-center px-2">
                <h1 className="text-3xl font-bold" style={{ color: SAFE_COLORS.cardForeground }}>AI Analysis Summary</h1>
                <Button variant="outline" size="sm" onClick={handleDownload} disabled={isGeneratingPdf}>
                    {isGeneratingPdf ? <><RefreshCw className="h-4 w-4 mr-2 animate-spin" /> Generating...</> : <><Download className="h-4 w-4 mr-2" /> Download Report</>}
                </Button>
            </header>
            <section>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <StatCard title="Overview" value="Key Trends" insight={summary.overview} icon={TrendingUp} color="blue" />
                    <StatCard title="Key Metrics" value="Top Figures" insight={summary.key_metrics} icon={Target} color="green" />
                    <StatCard title="Primary Insight" value="Main Takeaway" insight={summary.primary_insight} icon={Zap} color="orange" />
                </div>
            </section>
            <CollapsibleSection title="Detailed Report" icon={FileText} defaultOpen={false}>
                <div className="prose prose-sm sm:prose-base max-w-none space-y-8" style={{ color: SAFE_COLORS.cardForeground }}>
                    {report.detailed_analysis && (<div><h3 className="font-semibold text-lg mb-2">Detailed Analysis</h3><p style={{ color: SAFE_COLORS.mutedForeground }}>{report.detailed_analysis}</p></div>)}
                    {report.key_findings?.length > 0 && (<div><h3 className="font-semibold text-lg mb-2">Key Findings</h3><ul className="list-disc pl-5 space-y-2" style={{ color: SAFE_COLORS.mutedForeground }}>{report.key_findings.map((item, index) => <li key={index}>{item}</li>)}</ul></div>)}
                    {report.actionable_insights?.length > 0 && (<div><h3 className="font-semibold text-lg mb-2">Actionable Insights</h3><ul className="list-disc pl-5 space-y-2" style={{ color: SAFE_COLORS.mutedForeground }}>{report.actionable_insights.map((item, index) => <li key={index}>{item}</li>)}</ul></div>)}
                </div>
            </CollapsibleSection>
            {visualizations && visualizations.length > 0 && (
                <CollapsibleSection title="Visualizations" icon={BarChart3} defaultOpen={true}>
                    <div className="space-y-6">{visualizations.map((chart, index) => <ChartComponent key={chart.id || index} chartData={chart} chartId={`chart-container-${chart.id}`} />)}</div>
                </CollapsibleSection>
            )}
        </div>
    );
};

// --- Helper and Custom Hooks ---
const useApiCall = () => {
  const abortControllerRef = useRef(null);
  const makeApiCall = useCallback(async (query) => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();
    try {
      const response = await fetch(`{API_ENDPOINT}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
        signal: abortControllerRef.current.signal,
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Request was cancelled');
      throw error;
    }
  }, []);
  const cancelRequest = useCallback(() => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
  }, []);
  return { makeApiCall, cancelRequest };
};

const useChatHistory = () => {
    const [chats, setChats] = useState([]);
    const [activeChatId, setActiveChatId] = useState(null);
    const generateId = useCallback(() => `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`, []);

    useEffect(() => {
        try {
            const savedHistory = sessionStorage.getItem('chatbot-history');
            const parsedHistory = savedHistory ? JSON.parse(savedHistory) : [];
            if (parsedHistory.length > 0) {
                setChats(parsedHistory);
                setActiveChatId(parsedHistory[0].id);
            } else {
                const newChatId = generateId();
                const newChat = { id: newChatId, title: 'New Chat', timestamp: new Date(), messages: [] };
                setChats([newChat]);
                setActiveChatId(newChatId);
            }
        } catch (e) { console.error("Failed to load chat history", e); }
    }, [generateId]);

    useEffect(() => {
        if(chats.length > 0) sessionStorage.setItem('chatbot-history', JSON.stringify(chats));
    }, [chats]);

    const startNewChat = useCallback(() => {
        const newChat = { id: generateId(), title: 'New Chat', timestamp: new Date(), messages: [] };
        setChats(prev => [newChat, ...prev]);
        setActiveChatId(newChat.id);
    }, [generateId]);

    const addMessage = useCallback((message) => {
        setChats(prevChats => {
            const newChats = prevChats.map(chat => {
                if (chat.id === activeChatId) {
                    const newMessages = [...chat.messages, { ...message, id: generateId() }];
                    const firstUserMessage = newMessages.find(m => m.type === MESSAGE_TYPES.USER);
                    return {
                        ...chat,
                        messages: newMessages,
                        title: chat.title === 'New Chat' && firstUserMessage ? firstUserMessage.content : chat.title,
                        timestamp: new Date()
                    };
                }
                return chat;
            });
            return newChats.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        });
    }, [activeChatId, generateId]);

    const switchChat = useCallback((id) => setActiveChatId(id), []);
    const activeChat = useMemo(() => chats.find(c => c.id === activeChatId) || null, [chats, activeChatId]);
    return { chats, activeChat, startNewChat, addMessage, switchChat };
};


// --- Core UI Components (Updated with Safe Colors) ---
const MessageBubble = ({ message, onRetry }) => {
  const formatTime = useCallback((date) => new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), []);

  const messageStyle = message.type === 'user'
    ? { backgroundColor: SAFE_COLORS.primary, color: SAFE_COLORS.primaryForeground }
    : { backgroundColor: SAFE_COLORS.card, color: SAFE_COLORS.cardForeground, border: `1px solid ${SAFE_COLORS.border}` };

  const timeStyle = message.type === 'user'
    ? { color: 'rgba(255, 255, 255, 0.7)' }
    : { color: SAFE_COLORS.mutedForeground };

  const renderMessageContent = useMemo(() => {
    switch (message.type) {
      case MESSAGE_TYPES.REPORT: return <ReportDisplay data={message.content} />;
      case MESSAGE_TYPES.ERROR:
        return (
          <div className="space-y-3 p-4">
            <div className="flex items-center gap-2" style={{ color: SAFE_COLORS.destructive }}><AlertCircle size={16} /><span className="font-medium">Something went wrong</span></div>
            <p className="text-sm leading-relaxed" style={{ color: SAFE_COLORS.mutedForeground }}>{message.content}</p>
            {onRetry && message.originalQuery && <Button variant="ghost" size="sm" onClick={() => onRetry(message.originalQuery)}><RefreshCw size={14} className="mr-2" /> Try again</Button>}
          </div>
        );
      default: return <p className="text-sm leading-relaxed whitespace-pre-wrap p-4">{String(message.content)}</p>;
    }
  }, [message, onRetry]);

  return (
    <div className={`flex items-start gap-4 ${message.type === 'user' ? "justify-end" : ""}`}>
      {message.type !== 'user' && <div className="w-8 h-8 rounded-full border flex items-center justify-center flex-shrink-0 mt-1 shadow-sm" style={{ backgroundColor: SAFE_COLORS.card, color: SAFE_COLORS.mutedForeground }}><Bot size={18} /></div>}
      <div className="max-w-4xl w-full rounded-2xl shadow-sm" style={message.type !== 'report' ? messageStyle : {}}>
          {renderMessageContent}
          {message.type !== 'report' && <div className="text-xs mt-1.5 flex justify-end p-2" style={timeStyle}>{formatTime(message.timestamp)}</div>}
      </div>
      {message.type === 'user' && <div className="w-8 h-8 rounded-full border flex items-center justify-center flex-shrink-0 mt-1 shadow-sm" style={{ backgroundColor: SAFE_COLORS.muted, color: SAFE_COLORS.mutedForeground }}><User size={18} /></div>}
    </div>
  );
};

const WelcomeScreen = ({ onPromptClick, isLoading }) => (
    <div className="flex flex-col items-center justify-center h-full text-center p-4">
        <Sparkles className="h-12 w-12 mx-auto mb-4" style={{ color: SAFE_COLORS.primary }} />
        <h2 className="text-3xl font-bold tracking-tight mb-1" style={{ color: SAFE_COLORS.cardForeground }}>InsightsPlus</h2>
        <p className="text-lg mb-8" style={{ color: SAFE_COLORS.mutedForeground }}>Your AI-powered analytics assistant</p>
        <div className="flex flex-wrap items-center justify-center gap-3 mt-4">
            {EXEMPLAR_PROMPTS.map((prompt, i) => <Button key={i} variant="outline" onClick={() => onPromptClick(prompt)} disabled={isLoading}>{prompt}</Button>)}
        </div>
    </div>
);

const ChatHistorySidebar = ({ chats, activeChatId, onSelectChat, onNewChat }) => (
    <div className="w-80 flex-shrink-0 border rounded-lg shadow-sm flex flex-col" style={{ backgroundColor: SAFE_COLORS.card, borderColor: SAFE_COLORS.border }}>
        <div className="p-4 border-b flex justify-between items-center" style={{ borderColor: SAFE_COLORS.border }}>
            <h3 className="font-semibold text-lg">Chat History</h3>
            <Button onClick={onNewChat} variant="ghost" size="sm"><Plus className="h-4 w-4 mr-2" />New Chat</Button>
        </div>
        <div className="flex-1 p-2 space-y-2 overflow-y-auto">
            {chats.map(chat => (
                <button key={chat.id} onClick={() => onSelectChat(chat.id)} className="w-full text-left p-3 rounded-md transition-colors hover:bg-gray-100" style={{ backgroundColor: activeChatId === chat.id ? SAFE_COLORS.muted : 'transparent' }}>
                    <p className="truncate text-sm font-medium">{chat.title}</p>
                    <p className="text-xs mt-1" style={{ color: SAFE_COLORS.mutedForeground }}>{new Date(chat.timestamp).toLocaleString()}</p>
                </button>
            ))}
        </div>
    </div>
);

// --- Main Page Component ---
export default function ChatbotPage() {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  
  const { makeApiCall, cancelRequest } = useApiCall();
  const { chats, activeChat, startNewChat, addMessage, switchChat } = useChatHistory();

  const scrollToBottom = useCallback(() => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }), []);
  useEffect(() => { scrollToBottom() }, [activeChat?.messages, scrollToBottom]);
  useEffect(() => { inputRef.current?.focus() }, [isLoading, activeChat]);

  const handleSendMessage = useCallback(async (messageContent = input) => {
    const content = messageContent.trim();
    if (!content || isLoading) return;
    addMessage({ type: MESSAGE_TYPES.USER, content, timestamp: new Date().toISOString() });
    setInput(''); setIsLoading(true);
    try {
        const data = await makeApiCall(content);
        addMessage({ type: MESSAGE_TYPES.REPORT, content: data, timestamp: new Date().toISOString() });
    } catch (error) {
        addMessage({ type: MESSAGE_TYPES.ERROR, content: error.message, originalQuery: content, timestamp: new Date().toISOString() });
    } finally {
        setIsLoading(false);
    }
  }, [input, isLoading, addMessage, makeApiCall]);

  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); }
  }, [handleSendMessage]);

  const hasMessages = activeChat && activeChat.messages.length > 0;

  return (
    <div className="h-full flex flex-col p-4 md:p-6 lg:p-8">
        <header className="flex-shrink-0 mb-6">
            <div className="flex items-center gap-3">
                <div className="relative flex h-3 w-3"><span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ backgroundColor: SAFE_COLORS.chartGreen }}></span><span className="relative inline-flex rounded-full h-3 w-3" style={{ backgroundColor: SAFE_COLORS.chartGreen }}></span></div>
                <div>
                  <h1 className="text-2xl font-bold tracking-tight">InsightsPlus Assistant</h1>
                  <p style={{ color: SAFE_COLORS.mutedForeground }}>Your AI-powered analytics assistant.</p>
                </div>
            </div>
        </header>
        <div className="flex flex-1 gap-6 overflow-hidden">
            <div className="flex flex-col flex-1 h-full border rounded-lg shadow-sm" style={{ backgroundColor: SAFE_COLORS.card, borderColor: SAFE_COLORS.border }}>
              <main className="flex-1 overflow-y-auto p-4 md:p-6">
                  {!hasMessages ? (<WelcomeScreen onPromptClick={handleSendMessage} isLoading={isLoading} />) : (
                      <div className="space-y-6">
                          {activeChat?.messages.map((msg) => <MessageBubble key={msg.id} message={msg} onRetry={handleSendMessage} />)}
                          {isLoading && (
                            <div className="flex items-start gap-4"><div className="w-8 h-8 rounded-full border flex items-center justify-center flex-shrink-0 mt-1 shadow-sm" style={{backgroundColor: SAFE_COLORS.card, color: SAFE_COLORS.mutedForeground}}><Bot size={18} /></div>
                                <div className="rounded-2xl border px-4 py-3 shadow-sm" style={{backgroundColor: SAFE_COLORS.card, borderColor: SAFE_COLORS.border}}><div className="flex items-center space-x-1.5" aria-label="AI is analyzing"><div className="w-2 h-2 rounded-full animate-bounce" style={{ animationDelay: '0s', backgroundColor: SAFE_COLORS.primary }}></div><div className="w-2 h-2 rounded-full animate-bounce" style={{ animationDelay: '0.1s', backgroundColor: SAFE_COLORS.primary }}></div><div className="w-2 h-2 rounded-full animate-bounce" style={{ animationDelay: '0.2s', backgroundColor: SAFE_COLORS.primary }}></div></div></div>
                            </div>
                          )}
                          <div ref={messagesEndRef} />
                      </div>
                  )}
              </main>
              <footer className="p-4 border-t" style={{ backgroundColor: 'rgba(255, 255, 255, 0.5)', borderColor: SAFE_COLORS.border }}>
                  <div className="relative">
                    <input ref={inputRef} type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={handleKeyPress} placeholder="Ask for a detailed analysis or insights..." className="w-full border shadow-inner rounded-xl py-3 pl-4 pr-14 focus:outline-none focus:ring-2" style={{ backgroundColor: SAFE_COLORS.card, borderColor: SAFE_COLORS.border, ringColor: SAFE_COLORS.primary }} disabled={isLoading} />
                    <Button onClick={() => handleSendMessage()} disabled={isLoading || !input.trim()} size="icon" className="absolute right-2 top-1/2 -translate-y-1/2" style={{ backgroundColor: SAFE_COLORS.primary }}><Send className="w-5 h-5" /></Button>
                  </div>
                  {isLoading && <div className="flex justify-center mt-2"><Button variant="ghost" size="sm" onClick={cancelRequest}>Cancel</Button></div>}
              </footer>
            </div>
            <ChatHistorySidebar chats={chats} activeChatId={activeChat?.id} onSelectChat={switchChat} onNewChat={startNewChat} />
        </div>
    </div>
  );
}