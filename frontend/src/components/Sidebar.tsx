import { Plus, MessageSquare, Menu } from 'lucide-react';
import { clsx } from 'clsx';

export function Sidebar({ 
  sessions, 
  currentSessionId, 
  onSelectSession, 
  onNewSession, 
  isOpen, 
  toggle 
}: {
  sessions: any[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  isOpen: boolean;
  toggle: () => void;
}) {
  if (!isOpen) {
    return (
      <div className="absolute top-4 left-4 z-50">
        <button onClick={toggle} className="p-2 rounded-md hover:bg-[#2f2f2f] text-gray-400">
          <Menu size={20} />
        </button>
      </div>
    );
  }

  return (
    <aside className="w-[260px] bg-[#171717] flex-shrink-0 flex flex-col h-full border-r border-[#424242]/50 p-3 transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <button 
          onClick={onNewSession}
          className="flex-1 flex items-center gap-2 hover:bg-[#2f2f2f] transition px-3 py-2 rounded-lg text-sm text-gray-100"
        >
          <Plus size={16} />
          New chat
        </button>
        <button onClick={toggle} className="ml-2 p-2 hover:bg-[#2f2f2f] rounded-lg text-gray-400 transition">
          <Menu size={18} />
        </button>
      </div>
      
      <div className="flex-1 overflow-y-auto mt-2 space-y-1">
        <div className="text-xs font-semibold text-gray-500 mb-3 px-3 uppercase tracking-wider">
          Previous Chats
        </div>
        {sessions.map(s => (
          <button
            key={s.session_id}
            onClick={() => onSelectSession(s.session_id)}
            className={clsx(
              "w-full text-left px-3 py-2.5 rounded-lg text-[14px] flex items-center gap-3 transition-colors truncate",
              s.session_id === currentSessionId ? "bg-[#2f2f2f] text-gray-100 font-medium" : "text-gray-300 hover:bg-[#212121]"
            )}
          >
            <MessageSquare size={16} className="flex-shrink-0 opacity-70" />
            <span className="truncate">{s.title}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}
