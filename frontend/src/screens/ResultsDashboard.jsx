import OverallResultPanel from '../components/OverallResultPanel'
import ModalityCard from '../components/ModalityCard'
import EvidenceViewer from '../components/EvidenceViewer'
import ContextualAnalysis from '../components/ContextualAnalysis'
import { MODALITY_META, MODALITY_ORDER, SUPPLEMENTARY_ORDER } from '../components/modalityMeta'

export default function ResultsDashboard({ result, onReset }) {
  const modalities = result.modalities || {}

  return (
    <div className="stack" style={{ gap: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-5)' }}>
        <h1 style={{ fontSize: 22 }}>Video Analysis</h1>
        <button className="btn btn-secondary btn-sm" onClick={onReset}>
          Analyze another video
        </button>
      </div>

      <OverallResultPanel result={result} />

      <div className="section-heading">
        <h2>Modality Analysis</h2>
      </div>
      <div className="grid grid-cols-3" style={{ gap: 'var(--space-4)' }}>
        {MODALITY_ORDER.map((key) => (
          <ModalityCard key={key} meta={MODALITY_META[key]} data={modalities[key]} />
        ))}
      </div>

      <div className="section-heading">
        <h2>Proof of evidence</h2>
      </div>
      <EvidenceViewer evidence={result.evidence} />
      <ContextualAnalysis contextualAnalysis={result.contextual_analysis} transcript={result.transcript} segments={result.transcript_segments} />

      <div className="section-heading"><h2>Supplementary consistency checks</h2></div>
      <p>Eye-blink and lip-sync analyses are supplementary checks. They do not contribute to the final fake-likelihood score or verdict.</p>
      <div className="grid grid-cols-2" style={{ gap: 'var(--space-4)' }}>
        {SUPPLEMENTARY_ORDER.map((key) => <ModalityCard key={key} meta={MODALITY_META[key]} data={modalities[key]} supplementaryEvidence={result.evidence} />)}
      </div>
    </div>
  )
}
