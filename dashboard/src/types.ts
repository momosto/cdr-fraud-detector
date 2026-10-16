export type FraudType = 'wangiri' | 'irsf' | 'simbox' | 'unusual';

export interface Suspect {
  number: string;
  risk: number;
  fraud_type: FraudType;
  flagged: boolean;
  rule_hit: boolean;
  model_flag: boolean;
  model_percentile: number;
  reasons: string[];
  truth: string;
  features: Record<string, number>;
}

export interface MethodMetrics {
  flagged: number;
  true_positives: number;
  precision: number;
  recall: number;
  f1: number;
  by_variant: { fraud_type: string; variant: string; caught: number; planted: number }[];
  false_positives_by_cohort: Record<string, number>;
}

export interface Summary {
  period_start: string;
  days: number;
  cdrs: number;
  subscribers_and_callers: number;
  numbers_scored: number;
  fraud_planted: number;
  methods: Record<'rules' | 'model' | 'combined', MethodMetrics>;
  average_precision: { model_score: number; combined_risk: number };
}

export interface Benchmark {
  seeds: number[];
  methods: Record<string, Record<'precision' | 'recall' | 'f1' | 'flagged', { mean: number; min: number; max: number }>>;
  combined_average_precision: { mean: number; min: number; max: number };
}

export interface Timelines {
  start: string;
  series: Record<string, { made: number[]; received: number[] }>;
}

export const TYPE_LABEL: Record<FraudType, string> = {
  wangiri: 'Wangiri',
  irsf: 'IRSF',
  simbox: 'SIM box',
  unusual: 'Unusual',
};

export const FEATURE_LABEL: Record<string, string> = {
  out_calls_per_day: 'Calls made per active day',
  in_out_ratio: 'Incoming ÷ outgoing',
  distinct_dest_ratio: 'Share of calls to different numbers',
  short_call_share: 'Share of calls ≤ 3 s',
  mean_answered_s: 'Average answered call (s)',
  intl_share: 'Share international',
  high_risk_share: 'Share to premium-risk ranges',
  night_share: 'Share between 00:00 and 05:00',
  distinct_cells: 'Cells used',
  imei_sim_count: 'SIMs seen on the same handset',
  peak_hour_calls: 'Calls in busiest hour',
};

export const PERCENT_FEATURES = new Set(['distinct_dest_ratio', 'short_call_share', 'intl_share', 'high_risk_share', 'night_share']);
