import OverallResultPanel from '../components/OverallResultPanel'
import ModalityCard from '../components/ModalityCard'
import EvidenceViewer from '../components/EvidenceViewer'
import ContextualAnalysis from '../components/ContextualAnalysis'
import { MODALITY_META, MODALITY_ORDER } from '../components/modalityMeta'

export default function ResultsDashboard({ result, onReset }) {
  const modalities = result.modalities || {}

  return (
    <div className="stack" style={{ gap: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-5)' }}>
        <h1 style={{ fontSize: 22 }}>Deepfake Analysis</h1>
        <button className="btn btn-secondary btn-sm" onClick={onReset}>
          Analyze another video
        </button>
      </div>

      <OverallResultPanel result={result} />

      <div className="section-heading">
        <h2>Modality Analysis</h2>
      </div>
      <div className="grid grid-cols-5" style={{ gap: 'var(--space-4)' }}>
        {MODALITY_ORDER.map((key) => (
          <ModalityCard key={key} meta={MODALITY_META[key]} data={modalities[key]} />
        ))}
      </div>

      <div className="section-heading">
        <h2>Why This Video Was Flagged</h2>
      </div>
      <ContextualAnalysis contextualAnalysis={result.contextual_analysis} />

      <div className="section-heading">
        <h2>Evidence</h2>
      </div>
      <EvidenceViewer evidence={result.evidence} />
    </div>
  )
}
