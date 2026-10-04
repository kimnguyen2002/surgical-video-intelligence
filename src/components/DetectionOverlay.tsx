import React from 'react';
import { DetectedInstrument } from '../types';

interface DetectionOverlayProps {
  instruments: DetectedInstrument[];
  focusedInstrumentId: string | null;
  onSelectInstrument?: (id: string) => void;
  videoWidth?: number;
  videoHeight?: number;
}

export const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  instruments,
  focusedInstrumentId,
  onSelectInstrument,
}) => {
  if (!instruments || instruments.length === 0) return null;

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden select-none">
      {instruments.map((inst) => {
        const { box, label, confidence, type, id } = inst;
        const isFocused = focusedInstrumentId === id;

        // Normalized 0-1000 to percentages
        const top = `${(box.ymin / 1000) * 100}%`;
        const left = `${(box.xmin / 1000) * 100}%`;
        const width = `${((box.xmax - box.xmin) / 1000) * 100}%`;
        const height = `${((box.ymax - box.ymin) / 1000) * 100}%`;

        // Style based on type: predicted (dashed amber), recorded (solid green), refused (solid red)
        let borderClass = 'border-2 border-dashed border-amber-400 shadow-[0_0_14px_rgba(251,191,36,0.6),inset_0_0_12px_rgba(251,191,36,0.14)]';
        let badgeBg = 'bg-amber-400 text-black font-semibold';
        let badgeText = `${label} ${confidence !== undefined ? `${confidence}%` : ''}`;

        if (type === 'recorded') {
          borderClass = 'border-2 border-solid border-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.45)]';
          badgeBg = 'bg-emerald-600 text-white font-medium';
          badgeText = label;
        } else if (type === 'refused') {
          borderClass = 'border-2 border-solid border-red-500';
          badgeBg = 'bg-red-700 text-white font-medium';
          badgeText = `${label} (refused)`;
        }

        if (isFocused) {
          borderClass += ' ring-4 ring-amber-400/80 ring-offset-2 ring-offset-black animate-pulse';
        }

        return (
          <div
            key={id}
            id={`detection-box-${id}`}
            style={{
              top,
              left,
              width,
              height,
            }}
            onClick={(e) => {
              e.stopPropagation();
              onSelectInstrument?.(id);
            }}
            className={`absolute transition-all duration-200 pointer-events-auto cursor-pointer group ${borderClass} rounded-sm`}
          >
            {/* Tag Label Chip */}
            <div
              className={`absolute -top-6 left-0 px-2 py-0.5 text-xs leading-tight rounded-t-sm whitespace-nowrap shadow-md tracking-tight ${badgeBg} group-hover:brightness-110`}
            >
              {badgeText}
            </div>

            {/* Subtle Inner Glow on Hover */}
            <div className="w-full h-full bg-amber-400/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        );
      })}
    </div>
  );
};
