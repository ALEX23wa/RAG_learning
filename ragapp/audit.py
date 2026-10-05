"""Finalize a frozen run only with a complete, independently recorded human/agent audit."""
import json
import math
from pathlib import Path


def measured_performance(performance, case_ids):
    """Recompute warm request p95 and simultaneous service+database peak RSS.

    Missing, client-only, nonfinite or inconsistent measurements stay unknown.
    Summary numbers supplied by the runner are never accepted as measurements.
    """
    unknown = {'p95_seconds': None, 'peak_rss_bytes': None}
    if not isinstance(performance, dict):
        return unknown
    def positive_pid(value):
        return type(value) is int and 0 < value <= 2**63 - 1
    def duration(value):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value) and value >= 0
        except OverflowError:
            return False
    service_pid, database_pid = performance.get('service_pid'), performance.get('database_pid')
    records = performance.get('records')
    if (performance.get('scope') not in {'service_and_database', 'offline_service_and_database'}
            or performance.get('measurement_method') != 'psutil_process_rss'
            or performance.get('warmup_excluded') is not True
            or not positive_pid(service_pid) or not positive_pid(database_pid) or service_pid == database_pid
            or not isinstance(records, list) or len(records) != len(case_ids)
            or type(performance.get('samples')) is not int or performance['samples'] != len(case_ids)):
        return unknown
    sessions = performance.get('process_sessions', [{'service_pid': service_pid, 'database_pid': database_pid,
                                                     'warmup_excluded': True}])
    if not isinstance(sessions, list) or not sessions:
        return unknown
    pairs = set()
    for session in sessions:
        if not isinstance(session, dict):
            return unknown
        s, d = session.get('service_pid'), session.get('database_pid')
        if not positive_pid(s) or not positive_pid(d) or s == d or session.get('warmup_excluded') is not True:
            return unknown
        pairs.add((s, d))
    if (service_pid, database_pid) not in pairs:
        return unknown
    times, peaks, ids = [], [], []
    def rss(sample):
        if not isinstance(sample, dict):
            return None
        s, d = sample.get('service_rss_bytes'), sample.get('database_rss_bytes')
        return s + d if positive_pid(s) and positive_pid(d) else None
    for record in records:
        if not isinstance(record, dict):
            return unknown
        elapsed = record.get('elapsed')
        peak = rss(record)
        pair = (record.get('service_pid', service_pid), record.get('database_pid', database_pid))
        if (not all(positive_pid(value) for value in pair) or pair not in pairs
                or not isinstance(record.get('id'), str) or not duration(elapsed) or peak is None):
            return unknown
        samples = record.get('rss_samples')
        if samples is not None:
            if not isinstance(samples, list) or not samples:
                return unknown
            values = [rss(sample) for sample in samples]
            if any(value is None for value in values) or max(values) != peak:
                return unknown
        ids.append(record['id']); times.append(elapsed); peaks.append(peak)
    if not times or len(set(ids)) != len(ids) or set(ids) != set(case_ids):
        return unknown
    return {'p95_seconds': sorted(times)[math.ceil(.95 * len(times)) - 1], 'peak_rss_bytes': max(peaks)}


def finalize(report_path,audit_path,expected=60):
    """将完整独立审核合并到冻结评测报告，并判定四项验收门槛。

    Args:
        report_path: Path | str：已收集问答输出和辅助评分的 JSON 报告，会原地更新。
        audit_path: Path | str：包含 dataset_sha256、reviewer 和逐题 reviews 的独立审核 JSON。
        expected: int：必须完成并审核的题目数，默认 60。

    Returns:
        dict：status 及 accuracy、evidence_recall_at_10、faithfulness、citation_location_validity 等正式指标。

    Raises:
        ValueError：题数、审核集合、题集哈希或审核字段不合规。

    Notes:
        旧版门槛为85/85/90；quality-v2为90/90真实span/95，并要求真实trace和拒答指标。没有数据不会降级通过。
    """
    path=Path(report_path);report=json.loads(path.read_text(encoding='utf-8'))
    audit=json.loads(Path(audit_path).read_text(encoding='utf-8'))
    rows=report['results'];reviews={r['id']:r for r in audit['reviews']}
    if len(reviews)!=len(audit['reviews']) or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate case IDs in run or audit')
    if len(rows)!=expected or set(reviews)!={r['id'] for r in rows}:
        raise ValueError('Incomplete run or incomplete independent audit')
    if audit.get('dataset_sha256')!=report['dataset_sha256']:
        raise ValueError('Audit does not match the frozen dataset')
    quality=report.get('evaluation_schema')=='quality-v2'
    if not audit.get('reviewer'):
        raise ValueError('An identified independent reviewer is required')
    if quality and audit.get('runtime_fingerprint')!=report.get('runtime_fingerprint'):
        raise ValueError('Audit does not match the frozen runtime')
    passed=total=supported=0;valid=True
    negative_count=refusal_count=correct_refusals=positive_count=0
    for row in rows:
        review=reviews[row['id']]
        if not review.get('reason') or review.get('source_checked') is not True:
            raise ValueError('Every question needs a reason and original-source verification')
        if quality and review.get('gold_verified') is not True:
            raise ValueError('Quality-v2 gold needs independent original-source verification')
        if any(type(review.get(k)) is not int for k in ('supported_claims','total_claims')):
            raise ValueError('Factual claim counts must be integers')
        if any(type(review.get(k)) is not bool for k in ('pass','citations_valid')):
            raise ValueError('Audit outcomes must be explicit booleans')
        if not 0<=review['supported_claims']<=review['total_claims']:
            raise ValueError('Invalid factual claim counts')
        row['independent_audit']=review
        passed+=bool(review['pass']);supported+=review['supported_claims'];total+=review['total_claims']
        valid &= bool(review['citations_valid'])
        if quality:
            answerable=row.get('case_metadata',{}).get('answerable',review.get('answerable'))
            if type(answerable) is not bool:
                raise ValueError('Quality-v2 needs frozen answerability for every case')
            if 'answerable' in review and review['answerable']!=answerable:
                raise ValueError('Audit answerability differs from the frozen case')
            refusal=row.get('answer',{}).get('status')=='abstained' and not row.get('answer',{}).get('citations')
            positive_count+=answerable;negative_count+=not answerable
            refusal_count+=refusal;correct_refusals+=refusal and not answerable
    recalled=[x for r in rows for x in r['grade']['evidence_recalled']]
    spans=[x for r in rows for x in (r['grade'].get('span_recalled_at_10') or [])]
    if quality and not spans:
        raise ValueError('Quality-v2 cannot be finalized using legacy locator recall')
    metrics={'completed':len(rows),'total':expected,'passed':passed,'accuracy':passed/expected,
        'evidence_recall_at_10':sum(spans)/len(spans) if quality and spans else (sum(recalled)/len(recalled) if recalled else 0),
        'legacy_locator_recall_at_10':sum(recalled)/len(recalled) if recalled else 0,
        'span_recall_at_10':sum(spans)/len(spans) if spans else None,
        'faithfulness':supported/total if total else 0,'citation_location_validity':1.0 if valid else 0.0,
        'supported_claims':supported,'total_claims':total}
    recall=metrics['span_recall_at_10'] if quality else metrics['legacy_locator_recall_at_10']
    if quality:
        candidate_groups=[];context_groups=[];trace_complete=True
        for row in rows:
            score=row.get('retrieval_score',{})
            case=row.get('case_metadata',{})
            if case.get('answerable') and case.get('scorable',True):
                trace=row.get('answer',{}).get('evaluation_trace')
                if not isinstance(trace,dict) or not isinstance(trace.get('trace'),dict):
                    trace_complete=False;continue
                if score.get('span_at_80') is None or score.get('context_groups') is None:
                    trace_complete=False;continue
                candidate_groups.extend(score['span_at_80']);context_groups.extend(score['context_groups'])
        metrics.update(candidate_recall_at_80=sum(candidate_groups)/len(candidate_groups) if candidate_groups and trace_complete else None,
            context_coverage=sum(context_groups)/len(context_groups) if context_groups and trace_complete else None,
            abstention_precision=correct_refusals/refusal_count if refusal_count else None,
            abstention_recall=correct_refusals/negative_count if negative_count else None,
            answerable_false_abstention_rate=(refusal_count-correct_refusals)/positive_count if positive_count else None,
            trace_measurements_complete=trace_complete)
        performance=measured_performance(report.get('performance_measurement'), [row['id'] for row in rows])
        metrics.update(performance)
        measured=all(value is not None for value in performance.values())
        required={'accuracy':.9,'span_recall_at_10':.9,'faithfulness':.95,'candidate_recall_at_80':.95,
                  'abstention_precision':.9,'abstention_recall':.9,'citation_location_validity':1.}
        def sufficient(key,minimum):
            value=metrics.get(key)
            return type(value) in (int,float) and math.isfinite(value) and value>=minimum
        missing=[k for k in required if metrics.get(k) is None]
        missing += [k for k in ('context_coverage','p95_seconds','peak_rss_bytes') if metrics.get(k) is None]
        quality_pass=all(sufficient(k,v) for k,v in required.items()) and not missing
        quality_pass &= bool(measured and type(metrics['p95_seconds']) in (int,float)
                             and type(metrics['peak_rss_bytes']) in (int,float)
                             and math.isfinite(metrics['p95_seconds']) and math.isfinite(metrics['peak_rss_bytes'])
                             and 0<=metrics['p95_seconds']<=60 and 0<metrics['peak_rss_bytes']<=12*1024**3)
        report['missing_acceptance_measurements']=missing
        report['status']='passed' if quality_pass else ('awaiting_measurements' if missing else 'failed')
    else:
        report['status']='passed' if metrics['accuracy']>=.85 and recall>=.85 and metrics['faithfulness']>=.9 and valid else 'failed'
    report['metrics']=metrics
    report['audit_reviewer']=audit.get('reviewer')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'status':report['status'],**metrics}
