import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatWindow } from './components/ChatWindow';
import { MemoryPanel } from './components/MemoryPanel';

export default function App() {
  const [sessions, setSessions] = useState<{ session_id: string; title: string; agent_id: string }[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMemoryPanelOpen, setIsMemoryPanelOpen] = useState(true);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const res = await fetch('/api/sessions');
      const data = await res.json();
      setSessions(data.sessions);
      if (data.sessions.length > 0 && !currentSessionId) {
        setCurrentSessionId(data.sessions[0].session_id);
      } else if (data.sessions.length === 0) {
        await createSession();
      }
    } catch (e) {
      console.error('Error fetching sessions:', e);
    }
  };

  const createSession = async () => {
    const res = await fetch('/api/sessions', { method: 'POST' });
    const data = await res.json();
    setCurrentSessionId(data.session_id);
    await fetchSessions();
  };

  return (
    <div className="flex h-screen w-full bg-[#212121] overflow-hidden text-gray-100 font-sans">
      {/* Left Sidebar */}
      <Sidebar 
        sessions={sessions} 
        currentSessionId={currentSessionId} 
        onSelectSession={setCurrentSessionId}
        onNewSession={createSession}
        isOpen={isSidebarOpen}
        toggle={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#212121] relative h-full">
        {currentSessionId ? (
           <ChatWindow 
            sessionId={currentSessionId} 
            onMessageSent={fetchSessions} // Refresh sidebar titles
          />
        ) : (
           <div className="flex-1 flex items-center justify-center text-gray-500">Loading...</div>
        )}
      </main>

      {/* Right Memory Panel */}
      <MemoryPanel 
        isOpen={isMemoryPanelOpen}
        toggle={() => setIsMemoryPanelOpen(!isMemoryPanelOpen)}
      />
    </div>
  );
}
