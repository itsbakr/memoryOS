import { useState, useEffect } from 'react';
import { clsx } from 'clsx';
import { Settings, Zap, History, PanelRightClose, Database } from 'lucide-react';
import { motion } from 'framer-motion';

export function MemoryPanel({ isOpen, toggle }: { isOpen: boolean, toggle: () => void }) {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(() => {
      if (!document.hidden) fetchStats();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/memory/stats?agent_id=demo-agent');
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error(e);
    }
  };

  if (!isOpen) {
    return (
      <div className="absolute top-4 right-4 z-50">
        <button onClick={toggle} className="p-2 rounded-md hover:bg-[#2f2f2f] text-gray-400">
          <PanelRightClose size={20} />
        </button>
      </div>
    );
  }

  const savings = stats ? Math.max(0, (stats.total_memories * 150) - (stats.memories.length * 150)) : 0;

  return (
    <aside className="w-[300px] bg-[#171717] flex-shrink-0 border-l border-[#424242]/50 p-4 overflow-y-auto flex flex-col transition-all duration-300">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-2">
          <Database size={14} />
          Live Memory State
        </h3>
        <button onClick={toggle} className="p-1.5 hover:bg-[#2f2f2f] rounded text-gray-400 transition">
          <PanelRightClose size={16} />
        </button>
      </div>

      {/* Token Savings */}
      <div className="mb-6 p-4 rounded-xl bg-emerald-950/20 border border-emerald-900/30 flex items-center justify-between">
        <div className="text-emerald-500 font-medium flex items-center gap-2">
          <Zap size={16} />
          Tokens Saved
        </div>
        <div className="text-xl font-bold text-emerald-400">{savings.toLocaleString()}</div>
      </div>

      {/* Working Memory */}
      <div className="mb-8">
        <h4 className="text-[11px] font-bold text-gray-500 uppercase mb-3 flex items-center gap-2">
          <Settings size={12} /> Working Memory
        </h4>
        <div className="bg-[#212121] rounded-xl p-4 border border-[#333] shadow-inner">
          {stats?.working_memory ? (
            <>
              <div className="text-sm font-medium text-gray-200 mb-2">{stats.working_memory.task}</div>
              <div className="w-full bg-[#171717] h-1.5 rounded-full overflow-hidden mb-3">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${stats.working_memory.progress_pct}%` }}
                  className="bg-emerald-500 h-full rounded-full"
                />
              </div>
              <div className="text-xs text-gray-500 flex justify-between">
                <span>Action: {stats.working_memory.last_action || 'None'}</span>
                <span>{stats.working_memory.progress_pct}%</span>
              </div>
            </>
          ) : (
            <div className="text-sm text-gray-500 italic flex items-center gap-2">
              No active task
            </div>
          )}
        </div>
      </div>

      {/* Episodic Memory */}
      <div className="flex-1">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-[11px] font-bold text-gray-500 uppercase flex items-center gap-2">
            <History size={12} /> Episodic Facts
          </h4>
          <span className="text-xs text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full">
            {stats?.total_memories || 0} stored
          </span>
        </div>

        <div className="space-y-3">
          {!stats ? (
            <div className="text-sm text-gray-500">Loading...</div>
          ) : stats.memories.length === 0 ? (
            <div className="text-sm text-gray-500 italic p-4 text-center border border-dashed border-[#333] rounded-xl">
              No memories yet
            </div>
          ) : (
            stats.memories.map((m: any, i: number) => {
              let color = "border-emerald-500";
              if (m.confidence < 0.7) color = "border-amber-500";
              if (m.confidence < 0.4) color = "border-orange-500";
              if (m.confidence < 0.1) color = "border-red-500 opacity-60";

              return (
                <div key={i} className={clsx("bg-[#212121] p-3 rounded-lg border-l-2 shadow-sm", color)}>
                  <p className="text-[13px] text-gray-300 leading-snug mb-2">{m.content}</p>
                  <div className="flex justify-between text-[10px] text-gray-500 font-mono">
                    <span>conf: {m.confidence.toFixed(2)}</span>
                    <span>age: {m.age_hours}h</span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}
