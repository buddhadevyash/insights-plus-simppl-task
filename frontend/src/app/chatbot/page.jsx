"use client";
import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Send, User, Bot, Sparkles, RefreshCw, AlertCircle, Plus, Download, BarChart3, FileText, ChevronDown, TrendingUp, Target, Zap, TestTube2, Info, ThumbsUp, Star } from 'lucide-react';
import { cn } from "@/lib/utils";
import { Button } from '@/components/ui/button';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
import html2canvas from 'html2canvas';
import {
    BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    ScatterChart, Scatter, LineChart, Line
} from 'recharts';

// --- Safe Color Palette ---
const SAFE_COLORS = {
    primary: '#2563eb',
    primaryForeground: '#ffffff',
    card: '#ffffff',
    cardForeground: '#020817',
    background: '#f8fafc',
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
    chartOrange: '#f97316',
    // Stat Card Colors
    statBlueBg: '#dbeafe',
    statBlueText: '#1d4ed8',
    statGreenBg: '#dcfce7',
    statGreenText: '#166534',
    statOrangeBg: '#ffedd5',
    statOrangeText: '#9a3412',
};

// --- Constants ---
const API_ENDPOINT = process.env.NEXT_PUBLIC_FASTAPI || "http://127.0.0.1:8000";
const EXEMPLAR_PROMPTS = [
    "Discussions about AI safety and ethics",
    "Generate a sales report for the last quarter",
    "Summarize user engagement from the past month",
];
const MESSAGE_TYPES = {
    USER: 'user',
    BOT: 'bot',
    REPORT: 'report',
    ERROR: 'error'
};

// --- DATA TRANSFORMATION HELPER ---
const transformApiResponse = (apiData) => {
    // Defensive check for all essential data keys
    const requiredKeys = ['natural_response', 'statistical_analysis', 'detailed_report', 'visualizations', 'metadata', 'top_results'];
    for (const key of requiredKeys) {
        if (!apiData || !apiData[key]) {
            console.error(`Incomplete or malformed API data: missing key "${key}"`, apiData);
            return {
                summary: {
                    overview: "Incomplete data received from the server.",
                    key_metrics: "Data could not be processed.",
                    primary_insight: "Could not process the API response due to missing data.",
                },
                report: { detailed_analysis: "The data from the server was malformed or incomplete, preventing a full report from being generated." },
                visualizations: [],
                metadata: apiData?.metadata || {},
                top_results: null,
            };
        }
    }

    // Destructure apiData for cleaner access
    const { natural_response, statistical_analysis, detailed_report, visualizations: apiVisualizations, metadata, top_results: apiTopResults } = apiData;

    // Safely parse the title from the top_posts JSON string first
    const top_results = {
        ...apiTopResults,
        top_posts: apiTopResults.top_posts.map(post => {
            try {
                const parsedTitle = JSON.parse(post.title);
                return { ...post, title: parsedTitle.title || post.title };
            } catch (e) {
                return post;
            }
        })
    };

    // --- UPDATED LOGIC: Create a new, more robust "Top Figures" metric ---
    const topUsers = apiTopResults?.top_users || [];
    const topPosts = top_results?.top_posts || [];
    
    const totalReactions = topUsers.reduce((acc, user) => acc + (user.reactions || 0), 0);
    const uniqueUsers = metadata?.unique_users_count || 0;
    
    const topPost = topPosts.find(post => post.rank === 1);
    let topPostTitle = topPost ? topPost.title : 'Not Found';
    
    // Truncate title if it's too long to fit in the card
    if (topPostTitle.length > 30) {
        topPostTitle = `${topPostTitle.substring(0, 30)}...`;
    }

    const keyMetrics = `Reactions: ${totalReactions.toLocaleString()} | Users: ${uniqueUsers} | Top Post: "${topPostTitle}"`;
    // --- END OF UPDATED LOGIC ---
    
    // Map the summary section
    const summary = {
        overview: natural_response.summary,
        key_metrics: keyMetrics,
        primary_insight: statistical_analysis.inferential_statistics,
    };

    // Map the detailed report section
    const report = {
        detailed_analysis: detailed_report.comprehensive_analysis,
        key_findings: natural_response.key_points.map(point => ({ text: point })),
        actionable_insights: natural_response.actionable_insights.map(insight => ({ text: insight })),
        statistical_summary: {
            correlation_analysis: statistical_analysis.correlation_analysis,
            trend_analysis: statistical_analysis.trend_analysis,
        },
        methodology_and_limits: {
            statistical_methodology: statistical_analysis.methodology,
            data_quality_assessment: statistical_analysis.data_quality_assessment,
            limitations_and_caveats: detailed_report.limitations_and_caveats,
        },
        additional_analysis: {
            platform_comparison: detailed_report.platform_comparison,
            user_behavior_analysis: detailed_report.user_behavior_analysis,
            content_performance: detailed_report.content_performance,
        }
    };

    // Map the visualizations
    const visualizations = apiVisualizations.map(chart => {
        if ((chart.type === 'bar' || chart.type === 'line') && chart.labels && chart.data) {
            return { ...chart, data: { x: chart.labels, y: chart.data } };
        }
        if (chart.type === 'pie' && chart.labels && chart.data) {
            return { ...chart, data: { labels: chart.labels, values: chart.data } };
        }
        if (chart.type === 'scatter' && chart.data && chart.secondary_data) {
            return { ...chart, data: { x: chart.data, y: chart.secondary_data } };
        }
        return chart;
    });

    return { summary, report, visualizations, top_results, metadata };
};


// --- Charting Component ---
const ChartComponent = ({ chartData, chartId }) => {
    const data = useMemo(() => {
        if (!chartData || !chartData.data) return [];
        try {
            switch (chartData.type) {
                case 'bar':
                case 'line':
                case 'boxplot':
                    return chartData.data.x.map((label, index) => ({ name: label, value: chartData.data.y[index] }));
                case 'pie':
                    return chartData.data.labels.map((label, index) => ({ name: label, value: chartData.data.values[index] }));
                case 'scatter':
                    return chartData.data.x.map((xVal, index) => ({ x: xVal, y: chartData.data.y[index] }));
                default:
                    return [];
            }
        } catch (e) {
            console.error("Error processing chart data:", e);
            return [];
        }
    }, [chartData]);

    const COLORS = useMemo(() => chartData.colors || [SAFE_COLORS.chartBlue, SAFE_COLORS.chartGreen, SAFE_COLORS.chartOrange, SAFE_COLORS.chartRed, SAFE_COLORS.chartPurple, SAFE_COLORS.chartYellow], [chartData.colors]);

    const renderChart = () => {
        if (!data || data.length === 0) return <div className="text-center text-sm" style={{ color: SAFE_COLORS.mutedForeground }}>No data available or chart type unsupported.</div>;

        switch (chartData.type) {
            case 'bar': return (<BarChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" angle={-10} textAnchor="end" height={50} interval={0} /><YAxis /><Tooltip /><Legend /><Bar dataKey="value">{data.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}</Bar></BarChart>);
            case 'pie': return (<PieChart><Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={120} label>{data.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}</Pie><Tooltip /><Legend /></PieChart>);
            case 'line': return (<LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis domain={['auto', 'auto']} /><Tooltip /><Legend /><Line type="monotone" dataKey="value" name="Engagement Score" stroke={COLORS[0] || SAFE_COLORS.chartBlue} strokeWidth={2} activeDot={{ r: 8 }} /></LineChart>);
            case 'scatter': return (<ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}><CartesianGrid /><XAxis type="number" dataKey="x" name={chartData.xAxisLabel || 'x'} domain={['auto', 'auto']} /><YAxis type="number" dataKey="y" name={chartData.yAxisLabel || 'y'} /><Tooltip cursor={{ strokeDasharray: '3 3' }} /><Legend /><Scatter name="Data Points" data={data} fill={SAFE_COLORS.chartBlue} /></ScatterChart>);
            case 'boxplot': return (<BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 30, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis dataKey="name" type="category" width={80} /><Tooltip /><Legend payload={[{ value: 'Mean Value', type: 'square', color: SAFE_COLORS.chartGreen }]} /><Bar dataKey="value" name="Mean Value">{data.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}</Bar></BarChart>);
            default: return <div className="text-center" style={{ color: SAFE_COLORS.mutedForeground }}>Unsupported chart type: {chartData.type}</div>;
        }
    };

    return (
        <div id={chartId} className="p-4 rounded-lg shadow-sm" style={{ backgroundColor: SAFE_COLORS.card, border: `1px solid ${SAFE_COLORS.border}` }}>
            <h4 className="font-semibold mb-2" style={{ color: SAFE_COLORS.cardForeground }}>{chartData.title}</h4>
            <p className="text-sm mb-4" style={{ color: SAFE_COLORS.mutedForeground }}>{chartData.description}</p>
            <div style={{ width: '100%', height: 400 }}>
                <ResponsiveContainer>{renderChart()}</ResponsiveContainer>
            </div>
            {chartData.insights && (<div className="mt-4 p-3 rounded-lg" style={{ backgroundColor: SAFE_COLORS.statBlueBg }}><p className="text-sm" style={{ color: SAFE_COLORS.statBlueText }}><span className="font-medium">Insight:</span> {chartData.insights}</p></div>)}
        </div>
    );
};

// --- UI Components for Report ---
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
            {isOpen && <div className="px-6 pb-6" style={{ borderTop: `1px solid ${SAFE_COLORS.border}`, backgroundColor: '#fafcff' }}><div className="pt-6">{children}</div></div>}
        </section>
    );
};

const ReportSection = ({ title, content }) => {
    if (!content || (typeof content === 'string' && content.trim() === "")) return null;
    return (
        <div>
            <h3 className="font-semibold text-lg mb-2">{title}</h3>
            <p style={{ color: SAFE_COLORS.mutedForeground }}>{content}</p>
        </div>
    );
};

// --- TopResultsDisplay ---
const TopResultsDisplay = ({ top_results }) => {
    if (!top_results || (!top_results.top_users?.length && !top_results.top_posts?.length)) return null;

    return (
        <CollapsibleSection title="Top Results" icon={Star} defaultOpen={true}>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {top_results.top_users?.length > 0 && (
                    <div>
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2"><User size={20} /> Top Users</h3>
                        <ul className="space-y-3 max-h-[450px] overflow-y-auto pr-2">
                            {top_results.top_users.map(user => (
                                <li key={user.rank} className="p-3 bg-white rounded-lg border flex items-center justify-between gap-4">
                                    <div className="flex items-center gap-3">
                                        <span className="text-sm font-bold w-6 text-center" style={{ color: SAFE_COLORS.mutedForeground }}>{user.rank}</span>
                                        <p className="font-medium text-sm">{user.username}</p>
                                    </div>
                                    <div className="flex items-center gap-2 text-sm" style={{ color: SAFE_COLORS.primary }}>
                                        <ThumbsUp size={14} />
                                        <span>{user.reactions}</span>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                {top_results.top_posts?.length > 0 && (
                    <div>
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2"><FileText size={20} /> Top Posts</h3>
                        <ul className="space-y-3 max-h-[450px] overflow-y-auto pr-2">
                            {top_results.top_posts.map(post => (
                                <li key={post.rank} className="p-3 bg-white rounded-lg border">
                                    <a href={post.link} target="_blank" rel="noopener noreferrer" className="block hover:bg-gray-50 -m-3 p-3 rounded-lg">
                                        <div className="flex items-start justify-between gap-4">
                                            <div className="flex items-start gap-3">
                                                <span className="text-sm font-bold w-6 text-center pt-0.5" style={{ color: SAFE_COLORS.mutedForeground }}>{post.rank}</span>
                                                <p className="text-sm flex-1">{post.title}</p>
                                            </div>
                                            <div className="flex items-center gap-2 text-sm flex-shrink-0" style={{ color: SAFE_COLORS.primary }}>
                                                <ThumbsUp size={14} />
                                                <span>{post.reactions}</span>
                                            </div>
                                        </div>
                                        <p className="text-xs text-right mt-2" style={{ color: SAFE_COLORS.mutedForeground }}>{new Date(post.timestamp).toLocaleString()}</p>
                                    </a>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </CollapsibleSection>
    );
};


// --- Report Display Component ---
const ReportDisplay = ({ data }) => {
    const { summary, report, visualizations, top_results, metadata } = data || {};
    const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
    const reportRef = useRef(null);

    const handleDownload = useCallback(async () => {
        const reportElement = reportRef.current;
        if (!reportElement) return;

        setIsGeneratingPdf(true);
        try {
            const canvas = await html2canvas(reportElement, {
                scale: 2,
                useCORS: true,
                backgroundColor: SAFE_COLORS.background,
            });

            const imgData = canvas.toDataURL('image/png');
            const pdf = new jsPDF('p', 'mm', 'a4');
            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = pdf.internal.pageSize.getHeight();

            const canvasWidth = canvas.width;
            const canvasHeight = canvas.height;
            const ratio = canvasWidth / canvasHeight;

            const imgWidth = pdfWidth - 20; // pdf width with margins
            const imgHeight = imgWidth / ratio;

            let heightLeft = imgHeight;
            let position = 10; // y-position on PDF

            pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
            heightLeft -= (pdfHeight - 20);

            while (heightLeft > 0) {
                position = heightLeft - imgHeight + 10;
                pdf.addPage();
                pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
                heightLeft -= (pdfHeight - 20);
            }

            const filename = `InsightsPlus_Report_${new Date().toISOString().split('T')[0]}.pdf`;
            pdf.save(filename);

        } catch (error) {
            console.error("Error generating PDF:", error);
            alert("Sorry, there was an error generating the PDF. Please try again.");
        } finally {
            setIsGeneratingPdf(false);
        }
    }, []);

    if (!data || !summary || !report) return <div className="p-8 text-center rounded-lg" style={{ color: SAFE_COLORS.destructive, backgroundColor: '#fee2e2' }}>Invalid report data received.</div>;

    return (
        <div ref={reportRef} className="rounded-lg p-4 sm:p-6 space-y-6 max-w-full" style={{ backgroundColor: SAFE_COLORS.background }}>
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

            <TopResultsDisplay top_results={top_results} />

            <CollapsibleSection title="Detailed Report" icon={FileText} defaultOpen={true}>
                <div className="prose prose-sm sm:prose-base max-w-none space-y-8" style={{ color: SAFE_COLORS.cardForeground }}>
                    <ReportSection title="Comprehensive Analysis" content={report.detailed_analysis} />
                    {report.key_findings?.length > 0 && (<div><h3 className="font-semibold text-lg mb-2">Key Findings</h3><ul className="space-y-4">{report.key_findings.map((item, index) => <li key={index} className="p-3 bg-white rounded-md border"><p>{item.text}</p></li>)}</ul></div>)}
                    {report.actionable_insights?.length > 0 && (<div><h3 className="font-semibold text-lg mb-2">Actionable Insights</h3><ul className="space-y-4">{report.actionable_insights.map((item, index) => <li key={index} className="p-3 bg-white rounded-md border"><p>{item.text}</p></li>)}</ul></div>)}
                    <ReportSection title="User Behavior Analysis" content={report.additional_analysis?.user_behavior_analysis} />
                    <ReportSection title="Content Performance" content={report.additional_analysis?.content_performance} />
                    <ReportSection title="Platform Comparison" content={report.additional_analysis?.platform_comparison} />
                </div>
            </CollapsibleSection>

            {visualizations && visualizations.length > 0 && (
                <CollapsibleSection title="Visualizations" icon={BarChart3} defaultOpen={true}>
                    <div className="space-y-6">{visualizations.map((chart, index) => <ChartComponent key={chart.id || index} chartData={chart} chartId={`chart-container-${chart.id}`} />)}</div>
                </CollapsibleSection>
            )}

            <CollapsibleSection title="Statistical Details & Methodology" icon={TestTube2} defaultOpen={false}>
                <div className="space-y-6">
                    <ReportSection title="Correlation Analysis" content={report.statistical_summary?.correlation_analysis} />
                    <ReportSection title="Trend Analysis" content={report.statistical_summary?.trend_analysis} />
                    <ReportSection title="Statistical Methodology" content={report.methodology_and_limits?.statistical_methodology} />
                    <ReportSection title="Data Quality & Limitations" content={`${report.methodology_and_limits?.data_quality_assessment} ${report.methodology_and_limits?.limitations_and_caveats}`} />
                </div>
            </CollapsibleSection>

            {metadata && (
                <CollapsibleSection title="Analysis Metadata" icon={Info} defaultOpen={false}>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        {Object.entries(metadata).map(([key, value]) => (
                            <div key={key} className="p-2 bg-white rounded">
                                <p className="font-semibold capitalize">{key.replace(/_/g, ' ')}</p>
                                <p style={{ color: SAFE_COLORS.mutedForeground }}>{Array.isArray(value) ? value.join(', ') : String(value)}</p>
                            </div>
                        ))}
                    </div>
                </CollapsibleSection>
            )}
        </div>
    );
};

// --- Helper and Custom Hooks ---
const useApiCall = () => {
    const abortControllerRef = useRef(null);

    const makeApiCall = useCallback(async (query) => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        abortControllerRef.current = new AbortController();

        try {
            // ** This is your dynamic API endpoint **
            const response = await fetch(`${API_ENDPOINT}/neo4j-chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP Error: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Fetch request was cancelled.');
                throw new Error('Request was cancelled by the user.');
            }
            // Add a more generic error for network issues
            console.error("API call failed:", error);
            throw new Error('Failed to connect to the analysis server. Please check your connection and try again.');
        }
    }, []);

    const cancelRequest = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
    }, []);

    return { makeApiCall, cancelRequest };
};

const useChatHistory = () => {
    const [chats, setChats] = useState([]);
    const [activeChatId, setActiveChatId] = useState(null);
    const generateId = useCallback(() => `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`, []);

    useEffect(() => {
        const newChatId = generateId();
        setChats([{ id: newChatId, title: 'New Chat', timestamp: new Date(), messages: [] }]);
        setActiveChatId(newChatId);
    }, [generateId]);

    const startNewChat = useCallback(() => {
        const newChat = { id: generateId(), title: 'New Chat', timestamp: new Date(), messages: [] };
        setChats(prev => [newChat, ...prev.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))]);
        setActiveChatId(newChat.id);
        return newChat.id;
    }, [generateId]);

    const addMessage = useCallback((chatId, message) => {
        setChats(prevChats => prevChats.map(chat => {
            if (chat.id === chatId) {
                const isFirstUserMessage = chat.messages.length === 0 && message.type === MESSAGE_TYPES.USER;
                return {
                    ...chat,
                    messages: [...chat.messages, { ...message, id: generateId() }],
                    title: isFirstUserMessage ? message.content.substring(0, 30) : chat.title,
                    timestamp: new Date()
                };
            }
            return chat;
        }));
    }, [generateId]);

    const switchChat = useCallback((id) => setActiveChatId(id), []);
    const activeChat = useMemo(() => chats.find(c => c.id === activeChatId) || null, [chats, activeChatId]);
    return { chats, activeChat, activeChatId, startNewChat, addMessage, switchChat };
};


// --- Core UI Components ---
const MessageBubble = ({ message, onRetry }) => {
    const renderMessageContent = useMemo(() => {
        switch (message.type) {
            case MESSAGE_TYPES.REPORT: return <ReportDisplay data={message.content} />;
            case MESSAGE_TYPES.ERROR: return (
                <div className="space-y-3 p-4">
                    <div className="flex items-center gap-2" style={{ color: SAFE_COLORS.destructive }}><AlertCircle size={16} /><span className="font-medium">Something went wrong</span></div>
                    <p className="text-sm leading-relaxed" style={{ color: SAFE_COLORS.mutedForeground }}>{message.content}</p>
                    {onRetry && message.originalQuery && <Button variant="ghost" size="sm" onClick={() => onRetry(message.originalQuery)}><RefreshCw size={14} className="mr-2" /> Try again</Button>}
                </div>
            );
            default: return <p className="text-sm leading-relaxed whitespace-pre-wrap p-4">{String(message.content)}</p>;
        }
    }, [message, onRetry]);

    const messageStyle = message.type === 'user'
        ? { backgroundColor: SAFE_COLORS.primary, color: SAFE_COLORS.primaryForeground }
        : { backgroundColor: SAFE_COLORS.card, color: SAFE_COLORS.cardForeground, border: `1px solid ${SAFE_COLORS.border}` };

    return (
        <div className={`flex items-start gap-4 ${message.type === 'user' ? "justify-end" : ""}`}>
            {message.type !== 'user' && <div className="w-8 h-8 rounded-full border flex items-center justify-center flex-shrink-0 mt-1 shadow-sm" style={{ backgroundColor: SAFE_COLORS.card, color: SAFE_COLORS.mutedForeground }}><Bot size={18} /></div>}
            <div className="max-w-4xl w-full rounded-2xl shadow-sm" style={message.type !== 'report' ? messageStyle : {}}>
                {renderMessageContent}
            </div>
            {message.type === 'user' && <div className="w-8 h-8 rounded-full border flex items-center justify-center flex-shrink-0 mt-1 shadow-sm" style={{ backgroundColor: SAFE_COLORS.muted, color: SAFE_COLORS.mutedForeground }}><User size={18} /></div>}
        </div>
    );
};

const WelcomeScreen = ({ onPromptClick, isLoading }) => (
    <div className="flex flex-col items-center justify-center h-full text-center p-4">
        <Sparkles className="h-12 w-12 mx-auto mb-4" style={{ color: SAFE_COLORS.primary }} />
        <h2 className="text-3xl font-bold tracking-tight mb-1">InsightsPlus</h2>
        <p className="text-lg mb-8" style={{ color: SAFE_COLORS.mutedForeground }}>Your AI-powered analytics assistant</p>
        <div className="flex flex-wrap items-center justify-center gap-3 mt-4">
            {EXEMPLAR_PROMPTS.map((prompt, i) => <Button key={i} variant="outline" onClick={() => onPromptClick(prompt)} disabled={isLoading}>{prompt}</Button>)}
        </div>
    </div>
);

const ChatHistorySidebar = ({ chats, activeChatId, onSelectChat, onNewChat }) => (
    <div className="hidden lg:flex w-80 flex-shrink-0 border rounded-lg shadow-sm flex-col" style={{ backgroundColor: SAFE_COLORS.card, borderColor: SAFE_COLORS.border }}>
        <div className="p-4 border-b flex justify-between items-center">
            <h3 className="font-semibold text-lg">Chat History</h3>
            <Button onClick={onNewChat} variant="ghost" size="sm"><Plus className="h-4 w-4 mr-2" />New Chat</Button>
        </div>
        <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
            {chats.map(chat => (
                <button key={chat.id} onClick={() => onSelectChat(chat.id)} className="w-full text-left p-3 rounded-md transition-colors hover:bg-gray-100" style={{ backgroundColor: activeChatId === chat.id ? SAFE_COLORS.muted : 'transparent' }}>
                    <p className="truncate text-sm font-medium">{chat.title}</p>
                    <p className="text-xs mt-1" style={{ color: SAFE_COLORS.mutedForeground }}>{new Date(chat.timestamp).toLocaleString()}</p>
                </button>
            ))}
        </nav>
    </div>
);


// --- Main Page Component ---
export default function ChatbotPage() {
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    const { makeApiCall, cancelRequest } = useApiCall();
    const { chats, activeChat, activeChatId, startNewChat, addMessage, switchChat } = useChatHistory();

    useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [activeChat?.messages]);
    useEffect(() => { inputRef.current?.focus(); }, [activeChatId, isLoading]);

    const handleSendMessage = useCallback(async (messageContent = input) => {
        const content = messageContent.trim();
        if (!content || isLoading) return;

        let targetChatId = activeChatId;
        if (!activeChat || activeChat.messages.length > 0) {
            targetChatId = startNewChat();
        }

        addMessage(targetChatId, { type: MESSAGE_TYPES.USER, content, timestamp: new Date().toISOString() });
        setInput('');
        setIsLoading(true);

        try {
            const data = await makeApiCall(content);
            const transformedData = transformApiResponse(data);
            addMessage(targetChatId, { type: MESSAGE_TYPES.REPORT, content: transformedData, timestamp: new Date().toISOString() });
        } catch (error) {
            addMessage(targetChatId, { type: MESSAGE_TYPES.ERROR, content: error.message, originalQuery: content, timestamp: new Date().toISOString() });
        } finally {
            setIsLoading(false);
        }
    }, [input, isLoading, addMessage, makeApiCall, startNewChat, activeChat, activeChatId]);


    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    return (
        <div className="h-full flex flex-col p-4 md:p-6 lg:p-8 bg-slate-50">
            <header className="flex-shrink-0 mb-6">
                <h1 className="text-2xl font-bold tracking-tight">InsightsPlus Assistant</h1>
            </header>
            <div className="flex flex-1 gap-6 overflow-hidden">
                <div className="flex flex-col flex-1 h-full border rounded-lg shadow-sm bg-white">
                    <main className="flex-1 overflow-y-auto p-4 md:p-6">
                        {(!activeChat || activeChat.messages.length === 0) ? (<WelcomeScreen onPromptClick={handleSendMessage} isLoading={isLoading} />) : (
                            <div className="space-y-6">
                                {activeChat?.messages.map((msg) => <MessageBubble key={msg.id} message={msg} onRetry={handleSendMessage} />)}
                                {isLoading && (
                                    <div className="flex items-start gap-4">
                                        <div className="w-8 h-8 rounded-full border flex items-center justify-center flex-shrink-0 mt-1 shadow-sm" style={{ backgroundColor: SAFE_COLORS.card, color: SAFE_COLORS.mutedForeground }}><Bot size={18} /></div>
                                        <div className="rounded-2xl border px-4 py-3 shadow-sm" style={{ backgroundColor: SAFE_COLORS.card, borderColor: SAFE_COLORS.border }}>
                                            <div className="flex items-center space-x-1.5" aria-label="AI is analyzing">
                                                <div className="w-2 h-2 rounded-full animate-bounce" style={{ animationDelay: '0s', backgroundColor: SAFE_COLORS.primary }}></div>
                                                <div className="w-2 h-2 rounded-full animate-bounce" style={{ animationDelay: '0.1s', backgroundColor: SAFE_COLORS.primary }}></div>
                                                <div className="w-2 h-2 rounded-full animate-bounce" style={{ animationDelay: '0.2s', backgroundColor: SAFE_COLORS.primary }}></div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                                <div ref={messagesEndRef} />
                            </div>
                        )}
                    </main>
                    <footer className="p-4 border-t bg-white/50">
                        <div className="relative">
                            <input ref={inputRef} type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={handleKeyPress} placeholder="Ask for a detailed analysis..." className="w-full border shadow-inner rounded-xl py-3 pl-4 pr-14 focus:outline-none focus:ring-2" disabled={isLoading} />
                            <Button onClick={() => handleSendMessage()} disabled={isLoading || !input.trim()} size="icon" className="absolute right-2 top-1/2 -translate-y-1/2"><Send className="w-5 h-5" /></Button>
                        </div>
                        {isLoading && <div className="flex justify-center mt-2"><Button variant="ghost" size="sm" onClick={cancelRequest}>Cancel Request</Button></div>}
                    </footer>
                </div>
                <ChatHistorySidebar chats={chats} activeChatId={activeChatId} onSelectChat={switchChat} onNewChat={startNewChat} />
            </div>
        </div>
    );
}