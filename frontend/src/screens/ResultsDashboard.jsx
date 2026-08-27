import OverallResultPanel from '../components/OverallResultPanel'
import ModalityCard from '../components/ModalityCard'
import EvidenceViewer from '../components/EvidenceViewer'
import ContextualAnalysis from '../components/ContextualAnalysis'
import TechnicalDetails from '../components/TechnicalDetails'
import InfoTip from '../components/ui/InfoTip'
import { MODALITY_META, MODALITY_ORDER, RULE_BASED_CAVEAT } from '../components/modalityMeta'

export default function ResultsDashboard({ result, onReset }) {
  return (
    <div className="stack" style={{ gap: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-5)' }}>
        <h1 style={{ fontSize: 22 }}>Analysis results</h1>
        <button className="btn btn-secondary btn-sm" onClick={onReset}>
          Analyze another video
        </button>
      </div>

      <OverallResultPanel result={result} />

      <div className="section-heading">
        <h2>Modality analysis</h2>
        <span className="section-heading__hint" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          Learned vs. rule-based signals
          <InfoTip text={RULE_BASED_CAVEAT} />
        </span>
      </div>
      <div className="grid grid-cols-5" style={{ gap: 'var(--space-4)' }}>
        {MODALITY_ORDER.map((key) => (
          <ModalityCard key={key} meta={MODALITY_META[key]} data={result.modalities[key]} />
        ))}
      </div>

      <div className="section-heading">
        <h2>Evidence</h2>
      </div>
      <EvidenceViewer evidence={result.evidence} />

      <div className="section-heading">
        <h2>Contextual analysis</h2>
      </div>
      <ContextualAnalysis contextualAnalysis={result.contextual_analysis} />

      <div className="section-heading">
        <h2>Technical details</h2>
      </div>
      <TechnicalDetails result={result} />
    </div>
  )
}
