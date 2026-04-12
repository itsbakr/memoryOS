import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { SendHorizontal, Loader2, Sparkles, RefreshCw } from 'lucide-react';
import { clsx } from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

export function ChatWindow({
  sessionId,
  agentId,
  onMessageSent,
}: {
  sessionId: string;
  agentId: string;
  onMessageSent: () => void;
}) {
  const [messages, setMessages] = useState<{role: string, content: string, provenance?: any[]}[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [contradiction, setContradiction] = useState<any | null>(null);
  
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHistory();
    setContradiction(null);
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, contradiction, isLoading]);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`/api/chat/history?session_id=${sessionId}`);
      const data = await res.json();
      if (data.history && data.history.length > 0) {
        setMessages(data.history);
      } else {
        setMessages([{ role: 'agent', content: "Hello! How can I help you today?" }]);
      }
    } catch (e) {
      console.error(e);
      setMessages([{ role: 'agent', content: "Hello! How can I help you today?" }]);
    }
  };

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userText = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userText }]);
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: userText })
      });
      const data = await res.json();
      
      if (data.reply) {
        setMessages(prev => [
          ...prev,
          { role: 'agent', content: data.reply, provenance: data.provenance }
        ]);
      }
      if (data.contradiction) {
        setContradiction(data.contradiction);
      }
      onMessageSent(); // refresh titles
    } catch (e) {
      setMessages(prev => [...prev, { role: 'agent', content: "Error: Failed to fetch response." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const resolveContradiction = async (chosenFact: string) => {
    if (!contradiction) return;
    const cid = contradiction.contradiction_id;
    setContradiction({ ...contradiction, resolving: true });
    
    try {
      await fetch(`/api/contradictions/${cid}/resolve?agent_id=${encodeURIComponent(agentId)}&chosen_fact=${encodeURIComponent(chosenFact)}`, {
        method: 'POST'
      });
      
      setContradiction(null);
      setMessages(prev => [...prev, { 
        role: 'agent', 
        content: `*Memory updated:* I've saved "${chosenFact}" and removed the old conflict.` 
      }]);
      
      // Tell LLM
      const resolveMsg = `[System: The user resolved the contradiction. They confirmed the correct fact is: "${chosenFact}". Please acknowledge.]`;
      await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: resolveMsg })
      });
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      <div className="flex-1 overflow-y-auto p-4 sm:p-8 space-y-6 pb-48 scroll-smooth">
        {messages.map((msg, i) => (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            key={i} 
            className={clsx(
              "w-full max-w-3xl mx-auto flex items-start gap-4",
              msg.role === 'user' ? "justify-end" : "justify-start"
            )}
          >
            {msg.role === 'agent' && (
              <div className="w-8 h-8 rounded-full bg-emerald-600/20 text-emerald-500 flex items-center justify-center flex-shrink-0 mt-1">
                <Sparkles size={16} />
              </div>
            )}
            
            <div className={clsx(
              "px-5 py-3.5 max-w-[85%] text-[15px] leading-relaxed rounded-2xl",
              msg.role === 'user' 
                ? "bg-[#2f2f2f] text-gray-100 rounded-br-sm" 
                : "bg-transparent text-gray-100 prose prose-invert prose-p:leading-relaxed prose-pre:bg-[#1a1a1a] prose-pre:border prose-pre:border-gray-800"
            )}>
              {msg.role === 'agent' ? (
                <div>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                  {msg.provenance && msg.provenance.length > 0 && (
                    <div className="mt-3 text-xs text-gray-400">
                      <details className="group">
                        <summary className="cursor-pointer text-gray-500 hover:text-gray-300 transition">
                          Context used ({msg.provenance.length})
                        </summary>
                        <div className="mt-2 space-y-2">
                          {msg.provenance.map((item, idx) => (
                            <div key={idx} className="border border-[#333] rounded-lg p-2 bg-[#1a1a1a]">
                              <div className="text-gray-200 text-[12px] leading-snug">{item.content}</div>
                              <div className="mt-1 text-[10px] text-gray-500">
                                {item.category} · score {item.score} · {item.sources.join(', ')}
                              </div>
                            </div>
                          ))}
                        </div>
                      </details>
                    </div>
                  )}
                </div>
              ) : (
                msg.content
              )}
            </div>
          </motion.div>
        ))}

        <AnimatePresence>
          {contradiction && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-3xl mx-auto"
            >
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-5 ml-12">
                <h4 className="text-amber-500 font-medium flex items-center gap-2 mb-2">
                  <RefreshCw size={16} className={contradiction.resolving ? "animate-spin" : ""} />
                  Conflict Detected
                </h4>
                <div className="text-sm text-gray-300 mb-4 space-y-2">
                  <p><strong className="text-gray-400 font-medium">Old Memory:</strong> {contradiction.conflicts_with}</p>
                  <p><strong className="text-gray-400 font-medium">New Observation:</strong> {contradiction.new_fact}</p>
                  {contradiction.explanation && (
                    <p><strong className="text-gray-400 font-medium">Why:</strong> {contradiction.explanation}</p>
                  )}
                  {typeof contradiction.confidence_score === 'number' && (
                    <p><strong className="text-gray-400 font-medium">Confidence:</strong> {contradiction.confidence_score.toFixed(2)}</p>
                  )}
                </div>
                <div className="flex gap-3">
                  <button 
                    disabled={contradiction.resolving}
                    onClick={() => resolveContradiction(contradiction.new_fact)}
                    className="px-4 py-2 bg-[#2f2f2f] hover:bg-[#3f3f3f] disabled:opacity-50 text-gray-200 text-sm rounded-lg transition-colors"
                  >
                    Keep New
                  </button>
                  <button 
                    disabled={contradiction.resolving}
                    onClick={() => resolveContradiction(contradiction.conflicts_with)}
                    className="px-4 py-2 bg-transparent hover:bg-[#2f2f2f] disabled:opacity-50 text-gray-400 text-sm rounded-lg transition-colors border border-[#424242]"
                  >
                    Keep Old
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {isLoading && !contradiction && (
          <div className="w-full max-w-3xl mx-auto flex items-start gap-4">
            <div className="w-8 h-8 rounded-full bg-emerald-600/20 text-emerald-500 flex items-center justify-center flex-shrink-0 mt-1">
              <Sparkles size={16} />
            </div>
            <div className="px-5 py-3.5 text-gray-400 flex items-center gap-2">
              <Loader2 size={16} className="animate-spin" />
              Thinking...
            </div>
          </div>
        )}
        <div ref={bottomRef} className="h-4" />
      </div>

      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#212121] via-[#212121] to-transparent pt-10 pb-6 px-4">
        <form 
          onSubmit={handleSend}
          className="max-w-3xl mx-auto relative flex items-center bg-[#2f2f2f] border border-[#424242] focus-within:border-gray-500 transition-colors rounded-2xl shadow-xl overflow-hidden"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Message agent..."
            className="w-full max-h-48 bg-transparent text-gray-100 placeholder-gray-400 px-4 py-4 resize-none outline-none text-[15px]"
            rows={1}
          />
          <button 
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-2 bottom-2 p-2 bg-white text-black disabled:bg-[#424242] disabled:text-gray-500 rounded-full transition-colors flex items-center justify-center"
          >
            <SendHorizontal size={18} className="-ml-0.5" />
          </button>
        </form>
        <div className="text-center mt-3 text-xs text-gray-500">
          AI can make mistakes. Check important info.
        </div>
      </div>
    </div>
  );
}
