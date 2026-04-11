import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { User, FolderGit2, Workflow, RefreshCw } from 'lucide-react';

interface MemoryItem {
  id: string;
  content: string;
  category: string;
  source: string;
  confidence: number;
  age_hours: number;
}

interface Snapshot {
  agent_id: string;
  profile: MemoryItem[];
  project: MemoryItem[];
  workflow: MemoryItem[];
  working_memory: {
    task: string;
    progress_pct: number;
    last_action?: string;
  } | null;
  generated_at: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  personal_context: 'border-violet-500 text-violet-400',
  user_preference: 'border-sky-500 text-sky-400',
  project_decision: 'border-amber-500 text-amber-400',
  workflow_pattern: 'border-teal-500 text-teal-400',
  codebase_knowledge: 'border-orange-500 text-orange-400',
  task_context: 'border-pink-500 text-pink-400',
  general: 'border-gray-500 text-gray-400',
};

const CATEGORY_LABELS: Record<string, string> = {
  personal_context: 'Personal',
  user_preference: 'Preference',
  project_decision: 'Decision',
  workflow_pattern: 'Workflow',
  codebase_knowledge: 'Codebase',
  task_context: 'Task',
  general: 'General',
};

function ConfidenceDots({ confidence }: { confidence: number }) {
  const filled = confidence >= 0.8 ? 3 : confidence >= 0.5 ? 2 : 1;
  return (
    <span className="flex gap-0.5 items-center">
      {[1, 2, 3].map((i) => (
        <span
          key={i}
          className={`w-1.5 h-1.5 rounded-full ${i <= filled ? 'bg-emerald-400' : 'bg-[#3a3a3a]'}`}
        />
      ))}
    </span>
  );
}

function AgeLabel({ hours }: { hours: number }) {
  const label =
    hours < 1 ? '< 1h' : hours < 24 ? `${Math.round(hours)}h` : `${Math.round(hours / 24)}d`;
  return <span className="text-[10px] text-gray-500 font-mono">{label} ago</span>;
}

function MemoryCard({ item }: { item: MemoryItem }) {
  const colorClass = CATEGORY_COLORS[item.category] ?? CATEGORY_COLORS.general;
  const label = CATEGORY_LABELS[item.category] ?? item.category;
  const [borderColor] = colorClass.split(' ');

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-[#212121] p-3 rounded-lg border-l-2 ${borderColor} shadow-sm`}
    >
      <p className="text-[13px] text-gray-200 leading-snug mb-2">{item.content}</p>
      <div className="flex items-center justify-between">
        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${colorClass} bg-white/5`}>
          {label}
        </span>
        <div className="flex items-center gap-2">
          <ConfidenceDots confidence={item.confidence} />
          <AgeLabel hours={item.age_hours} />
        </div>
      </div>
    </motion.div>
  );
}

function Section({
  icon,
  title,
  items,
  empty,
}: {
  icon: React.ReactNode;
  title: string;
  items: MemoryItem[];
  empty: string;
}) {
  return (
    <div className="mb-6">
      <h4 className="text-[11px] font-bold text-gray-500 uppercase mb-3 flex items-center gap-2">
        {icon} {title}
        <span className="ml-auto text-xs text-emerald-500 bg-emerald-500/10 px-1.5 py-0.5 rounded-full normal-case font-normal">
          {items.length}
        </span>
      </h4>
      {items.length === 0 ? (
        <p className="text-sm text-gray-600 italic px-2">{empty}</p>
      ) : (
        <div className="space-y-2">
          {items.map((item) => (
            <MemoryCard key={item.id} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}

export function DeveloperProfile({ agentId = 'demo-agent' }: { agentId?: string }) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchSnapshot = async () => {
    try {
      const res = await fetch(`/api/context/snapshot?agent_id=${agentId}`);
      const data = await res.json();
      setSnapshot(data);
      setLastRefresh(new Date());
    } catch (e) {
      console.error('Error fetching snapshot:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSnapshot();
    const interval = setInterval(() => {
      if (!document.hidden) fetchSnapshot();
    }, 10000);
    return () => clearInterval(interval);
  }, [agentId]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-sm text-gray-500 animate-pulse">Loading profile...</div>
      </div>
    );
  }

  if (!snapshot) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-sm text-gray-600 text-center px-4">
          Could not load profile. Is the backend running?
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-[10px] text-gray-600 font-mono">
          agent: {snapshot.agent_id}
        </p>
        <button
          onClick={fetchSnapshot}
          className="p-1 hover:bg-[#2f2f2f] rounded text-gray-500 hover:text-gray-300 transition"
          title="Refresh"
        >
          <RefreshCw size={12} />
        </button>
      </div>

      {/* Who I Am */}
      <Section
        icon={<User size={12} />}
        title="Who I Am"
        items={snapshot.profile}
        empty="Tell Claude about yourself — timezone, team, role — and it will remember."
      />

      {/* This Project */}
      <Section
        icon={<FolderGit2 size={12} />}
        title="This Project"
        items={snapshot.project}
        empty="Make architectural decisions with Claude and they'll be stored here."
      />

      {/* How I Work */}
      <Section
        icon={<Workflow size={12} />}
        title="How I Work"
        items={snapshot.workflow}
        empty="Mention your workflow patterns (deploy scripts, test commands) and Claude will remember them."
      />

      {/* Footer */}
      {lastRefresh && (
        <p className="text-[10px] text-gray-600 text-center mt-2 pb-2">
          Last synced {lastRefresh.toLocaleTimeString()}
        </p>
      )}
    </div>
  );
}
