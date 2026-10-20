// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Coach defined football actions and models learned from marked examples.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::eval::{
    self, Context, Goal, Measures, Placed, Profile, Reference, Rule, Shape, Snapshot, Verdict,
};

/// KPI keys measured by [`generic`].
pub const GENERIC_KPIS: &[&str] = &[
    "block_length",
    "width",
    "def_line_height",
    "line_gap",
    "centroid_depth",
    "pressers_10m",
    "nearest_ball",
    "behind_ball",
    "pressers_2s",
    "behind_ball_5s",
    "shift_follow",
    "progress_m",
    "support_15m",
    "superiority_zone",
    "regain",
];

/// Measure the generic catalogue across an arbitrary time window.
/// The window start is the first snapshot's time; only snapshots with shape
/// observability of at least 0.6 contribute. Unobservable values are `None`.
pub fn generic(window: &[Snapshot], team: u8, goal: Goal) -> Measures {
    let mut out = Measures::new();
    let start = window.first().map_or(0, |s| s.ms);
    let visible: Vec<&Snapshot> = window
        .iter()
        .filter(|s| eval::shape(s, team, goal, 20.0).observability >= 0.6)
        .collect();
    let mut put = |k: &str, v| {
        out.insert(k.to_owned(), v);
    };
    let shapes: Vec<Shape> = visible
        .iter()
        .map(|s| eval::shape(s, team, goal, 20.0))
        .collect();
    let avg = |f: fn(&Shape) -> Option<f64>| mean_values(shapes.iter().filter_map(f));
    put("block_length", avg(|s| s.block_length));
    put("width", avg(|s| s.width));
    put("def_line_height", avg(|s| s.def_line_height));
    put("line_gap", avg(|s| s.line_gap));
    put(
        "centroid_depth",
        mean_values(
            shapes
                .iter()
                .filter_map(|shape| shape.centroid.map(|(x, _)| eval::depth(goal, x))),
        ),
    );
    put(
        "pressers_10m",
        mean_values(visible.iter().filter_map(|s| {
            s.ball.map(|b| {
                outfield(s, team)
                    .filter(|p| distance(p.at, b) <= 10.0)
                    .count() as f64
            })
        })),
    );
    put(
        "nearest_ball",
        mean_values(visible.iter().filter_map(|s| {
            s.ball.and_then(|b| {
                outfield(s, team)
                    .map(|p| distance(p.at, b))
                    .min_by(f64::total_cmp)
            })
        })),
    );
    put(
        "behind_ball",
        mean_values(visible.iter().filter_map(|s| {
            s.ball.map(|b| {
                outfield(s, team)
                    .filter(|p| eval::depth(goal, p.at.0) < eval::depth(goal, b.0))
                    .count() as f64
            })
        })),
    );
    for (sec, key, count) in [(2.0, "pressers_2s", true), (5.0, "behind_ball_5s", false)] {
        let value = nearest(&visible, start, sec).and_then(|s| {
            let b = s.ball?;
            Some(if count {
                outfield(s, team)
                    .filter(|p| distance(p.at, b) <= 10.0)
                    .count() as f64
            } else {
                outfield(s, team)
                    .filter(|p| eval::depth(goal, p.at.0) < eval::depth(goal, b.0))
                    .count() as f64
            })
        });
        put(key, value);
    }
    let pairs: Vec<(f64, f64)> = visible
        .iter()
        .zip(&shapes)
        .filter_map(|(s, shape)| Some((shape.centroid?.1, s.ball?.1)))
        .collect();
    put("shift_follow", (pairs.len() >= 5).then(|| corr(&pairs)));
    let ball_depths: Vec<f64> = visible
        .iter()
        .filter_map(|s| s.ball.map(|b| eval::depth(goal, b.0)))
        .collect();
    put(
        "progress_m",
        ball_depths
            .first()
            .zip(ball_depths.last())
            .map(|(a, b)| b - a),
    );
    put(
        "support_15m",
        mean_values(visible.iter().filter_map(|s| {
            s.ball.map(|b| {
                outfield(s, team)
                    .filter(|p| distance(p.at, b) <= 15.0)
                    .count() as f64
            })
        })),
    );
    put(
        "superiority_zone",
        mean_values(visible.iter().filter_map(|s| {
            s.ball.map(|b| {
                outfield(s, team)
                    .filter(|p| distance(p.at, b) <= 20.0)
                    .count() as f64
                    - outfield(s, 3 - team)
                        .filter(|p| distance(p.at, b) <= 20.0)
                        .count() as f64
            })
        })),
    );
    let regained = visible
        .iter()
        .filter(|s| s.ms >= start.saturating_add(1000))
        .any(|s| {
            s.ball
                .is_some_and(|b| nearest_player(s, b).is_some_and(|(d, t)| t == team && d <= 2.5))
        });
    put(
        "regain",
        (!visible.is_empty()).then_some(if regained { 1.0 } else { 0.0 }),
    );
    put(
        "outside_box",
        mean_values(visible.iter().map(|s| {
            outfield(s, team)
                .filter(|p| (p.at.1 - 34.0).abs() > 18.0)
                .count() as f64
        })),
    );
    put(
        "interiors_press_wide",
        mean_values(visible.iter().filter_map(|s| {
            let ball = s.ball?;
            if (ball.1 - 34.0).abs() <= 17.0 {
                return None;
            }
            let players: Vec<&Placed> = outfield(s, team).collect();
            if !players.iter().any(|p| p.role.is_some()) {
                return None;
            }
            let nearest = players
                .iter()
                .min_by(|a, b| distance(a.at, ball).total_cmp(&distance(b.at, ball)))?;
            Some(
                if nearest
                    .role
                    .is_some_and(|r| matches!(r, super::Role::Interior | super::Role::Lateral))
                {
                    1.0
                } else {
                    0.0
                },
            )
        })),
    );
    put(
        "pivot_central",
        mean_values(visible.iter().filter_map(|s| {
            let pivots: Vec<&Placed> = outfield(s, team)
                .filter(|p| p.role == Some(super::Role::Pivote))
                .collect();
            (!pivots.is_empty()).then(|| {
                if pivots.iter().any(|p| (p.at.1 - 34.0).abs() <= 10.0) {
                    1.0
                } else {
                    0.0
                }
            })
        })),
    );
    put(
        "def_line_by_role",
        mean_values(visible.iter().flat_map(|s| {
            outfield(s, team)
                .filter(|p| {
                    p.role
                        .is_some_and(|r| matches!(r, super::Role::Central | super::Role::Lateral))
                })
                .map(|p| eval::depth(goal, p.at.0))
        })),
    );
    out
}

/// Spanish display label for generic KPIs, falling back to the built-in catalogue.
pub fn generic_label(key: &str) -> &'static str {
    match key {
        "centroid_depth" => "profundidad del centroide",
        "pressers_10m" => "jugadores a 10 m del balón",
        "nearest_ball" => "distancia del más cercano al balón",
        "behind_ball" => "jugadores por detrás del balón",
        "pressers_2s" => "presionando a los 2 s",
        "regain" => "recuperación del balón",
        "outside_box" => "jugadoras fuera de la caja",
        "interiors_press_wide" => "interiores o laterales presionando fuera",
        "pivot_central" => "pivote en zona central",
        "def_line_by_role" => "altura de centrales y laterales",
        _ => eval::kpi_label(key),
    }
}

/// Trigger that starts a custom action window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Trigger {
    /// The team loses possession.
    BallLost,
    /// The team wins possession.
    BallWon,
    /// The rival holds possession for at least this many seconds.
    RivalPossession {
        /// Shortest spell that counts, in seconds.
        min_s: f64,
    },
    /// The team possesses the ball in its own half-depth area for this long.
    OwnPossessionDeep {
        /// Deepest the ball may be from the own goal, in metres.
        max_depth: f64,
        /// Shortest spell that counts, in seconds.
        min_s: f64,
    },
    /// Window is entered manually by an analyst.
    Manual,
}

/// Definition of a coach-created action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionDef {
    /// Stable action identifier.
    pub key: String,
    /// Display name.
    pub name: String,
    /// Event that starts an action window.
    pub trigger: Trigger,
    /// Window duration in seconds.
    pub window_s: f64,
    /// KPI keys to display for this action.
    pub kpis: Vec<String>,
}

/// Time interval associated with a custom action occurrence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Window {
    /// Stable action identifier.
    pub key: String,
    /// Team being assessed.
    pub team: u8,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}

/// Find occurrences of a custom action in a match context.
pub fn find_windows(ctx: &Context, def: &ActionDef, team: u8) -> Vec<Window> {
    let duration = (def.window_s.max(0.0) * 1000.0) as u64;
    let mut starts = Vec::new();
    match &def.trigger {
        Trigger::BallLost | Trigger::BallWon => {
            for c in eval::changes(&ctx.possession) {
                if (matches!(&def.trigger, Trigger::BallLost) && c.from == team)
                    || (matches!(&def.trigger, Trigger::BallWon) && c.to == team)
                {
                    starts.push(c.ms);
                }
            }
        }
        Trigger::RivalPossession { min_s } => {
            for (begin, end) in spells(&ctx.possession, 3 - team) {
                if end.saturating_sub(begin) >= ((*min_s).max(0.0) * 1000.0) as u64 {
                    starts.push(begin);
                }
            }
        }
        Trigger::OwnPossessionDeep { max_depth, min_s } => {
            let mut last_emit = None;
            for (begin, end) in spells(&ctx.possession, team) {
                let deep: Vec<u64> = ctx
                    .possession
                    .iter()
                    .filter(|(ms, p)| *p == Some(team) && begin <= *ms && *ms <= end)
                    .filter_map(|(ms, _)| {
                        let ball = ctx.snaps.iter().min_by_key(|s| s.ms.abs_diff(*ms))?.ball?;
                        Some(*ms).filter(|_| {
                            ctx.sides
                                .goal(team, *ms)
                                .is_some_and(|g| eval::depth(g, ball.0) < *max_depth)
                        })
                    })
                    .collect();
                if let (Some(first), Some(last)) = (deep.first(), deep.last())
                    && last.saturating_sub(*first) >= ((*min_s).max(0.0) * 1000.0) as u64
                    && last_emit.is_none_or(|prior: u64| first.saturating_sub(prior) >= 10_000)
                {
                    starts.push(*first);
                    last_emit = Some(*first);
                }
            }
        }
        Trigger::Manual => {}
    }
    starts
        .into_iter()
        .map(|start_ms| Window {
            key: def.key.clone(),
            team,
            start_ms,
            end_ms: start_ms.saturating_add(duration),
        })
        .collect()
}

/// Measure a custom action window and return its mean shape observability.
pub fn measure_window(ctx: &Context, w: &Window) -> Option<(Measures, f64)> {
    let goal = ctx.sides.goal(w.team, w.start_ms)?;
    let snaps: Vec<Snapshot> = ctx
        .snaps
        .iter()
        .filter(|s| w.start_ms <= s.ms && s.ms <= w.end_ms)
        .cloned()
        .collect();
    let observability = if snaps.is_empty() {
        0.0
    } else {
        snaps
            .iter()
            .map(|s| eval::shape(s, w.team, goal, 20.0).observability)
            .sum::<f64>()
            / snaps.len() as f64
    };
    Some((generic(&snaps, w.team, goal), observability))
}

/// Load custom actions from `actions.json`, returning an empty list if unavailable.
pub fn load_actions(dir: &Path) -> Vec<ActionDef> {
    read_json(&dir.join("actions.json")).unwrap_or_default()
}
/// Save custom actions to `actions.json`.
pub fn save_actions(dir: &Path, actions: &[ActionDef]) -> Result<(), String> {
    write_json(&dir.join("actions.json"), actions)
}

/// Logistic model learned from coach-marked examples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// Action identifier this model assesses.
    pub action_key: String,
    /// KPI keys in model order.
    pub kpis: Vec<String>,
    /// Training means used for standardisation.
    pub means: Vec<f64>,
    /// Training scales used for standardisation.
    pub scales: Vec<f64>,
    /// Logistic coefficients.
    pub weights: Vec<f64>,
    /// Logistic intercept.
    pub bias: f64,
    /// Good example count.
    pub n_good: usize,
    /// Bad example count.
    pub n_bad: usize,
    /// Leave-one-out accuracy for datasets of at most 200 examples.
    pub loo_accuracy: Option<f64>,
}

/// Prediction from a learned action model.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    /// Estimated probability that the action is good.
    pub p_good: f64,
    /// KPI effects, largest absolute effect first.
    pub contributions: Vec<(String, f64)>,
    /// KPIs absent from the supplied measures.
    pub missing: Vec<String>,
}

impl Model {
    /// Predict whether measured KPIs resemble good coach-marked examples.
    pub fn predict(&self, m: &Measures) -> Prediction {
        let mut z = Vec::with_capacity(self.kpis.len());
        let mut missing = Vec::new();
        let mut contributions = Vec::new();
        for (i, kpi) in self.kpis.iter().enumerate() {
            let value = m.get(kpi).copied().flatten();
            if value.is_none() {
                missing.push(kpi.clone());
            }
            let value = value.unwrap_or(self.means.get(i).copied().unwrap_or(0.0));
            let zi = (value - self.means.get(i).copied().unwrap_or(0.0))
                / self.scales.get(i).copied().unwrap_or(1.0);
            z.push(zi);
            contributions.push((
                kpi.clone(),
                self.weights.get(i).copied().unwrap_or(0.0) * zi,
            ));
        }
        let logit = self.bias + z.iter().zip(&self.weights).map(|(x, w)| x * w).sum::<f64>();
        contributions.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        Prediction {
            p_good: sigmoid(logit),
            contributions,
            missing,
        }
    }

    /// Explain the strongest KPI effects in short Spanish phrases.
    pub fn explain(&self, p: &Prediction) -> Vec<String> {
        p.contributions
            .iter()
            .filter(|(_, v)| v.abs() > 1e-9)
            .take(3)
            .map(|(k, v)| {
                format!(
                    "{}: {}",
                    if *v >= 0.0 { "A favor" } else { "En contra" },
                    generic_label(k)
                )
            })
            .collect()
    }
}

/// Train a balanced, regularised logistic model for one action.
pub fn train(action_key: &str, examples: &[Reference]) -> Option<Model> {
    let rows: Vec<&Reference> = examples
        .iter()
        .filter(|r| r.action_key == action_key)
        .collect();
    let n_good = rows.iter().filter(|r| r.good).count();
    let n_bad = rows.len() - n_good;
    if n_good < 4 || n_bad < 4 {
        return None;
    }
    let keys: BTreeSet<String> = rows
        .iter()
        .flat_map(|r| r.measures.keys().cloned())
        .collect();
    let kpis: Vec<String> = keys
        .into_iter()
        .filter(|key| {
            rows.iter()
                .filter(|r| r.measures.get(key).is_some_and(Option::is_some))
                .count()
                * 10
                >= rows.len() * 7
        })
        .collect();
    if kpis.is_empty() {
        return None;
    }
    let means: Vec<f64> = kpis
        .iter()
        .map(|k| {
            rows.iter()
                .filter_map(|r| r.measures.get(k).copied().flatten())
                .sum::<f64>()
                / rows
                    .iter()
                    .filter(|r| r.measures.get(k).is_some_and(Option::is_some))
                    .count() as f64
        })
        .collect();
    let scales: Vec<f64> = kpis
        .iter()
        .zip(&means)
        .map(|(k, mean)| {
            let vals: Vec<f64> = rows
                .iter()
                .filter_map(|r| r.measures.get(k).copied().flatten())
                .collect();
            let sd =
                (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
            if sd < 1e-6 { 1.0 } else { sd }
        })
        .collect();
    let x: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            kpis.iter()
                .zip(&means)
                .zip(&scales)
                .map(|((k, mean), scale)| {
                    (r.measures.get(k).copied().flatten().unwrap_or(*mean) - mean) / scale
                })
                .collect()
        })
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| if r.good { 1.0 } else { 0.0 })
        .collect();
    let mut model = fit_logistic(action_key, &x, &y, &means, &scales, &kpis, n_good, n_bad);
    if rows.len() <= 200 {
        let mut correct = 0;
        // Leave one out, the scaling kept from the whole set: a small leak
        // that matters little next to how few examples there are.
        #[allow(clippy::needless_range_loop)]
        for held in 0..rows.len() {
            let xx: Vec<Vec<f64>> = x
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != held)
                .map(|(_, v)| v.clone())
                .collect();
            let yy: Vec<f64> = y
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != held)
                .map(|(_, v)| *v)
                .collect();
            let ng = yy.iter().filter(|v| **v == 1.0).count();
            let nb = yy.len() - ng;
            let fold = fit_logistic(action_key, &xx, &yy, &means, &scales, &kpis, ng, nb);
            if (fold.predict(&rows[held].measures).p_good >= 0.5) == rows[held].good {
                correct += 1;
            }
        }
        model.loo_accuracy = Some(correct as f64 / rows.len() as f64);
    }
    Some(model)
}

/// Load all learned models from `models.json`.
pub fn load_models(dir: &Path) -> Vec<Model> {
    read_json(&dir.join("models.json")).unwrap_or_default()
}
/// Save all learned models to `models.json`.
pub fn save_models(dir: &Path, models: &[Model]) -> Result<(), String> {
    write_json(&dir.join("models.json"), models)
}
/// Train one model for each action represented in the examples.
pub fn retrain_all(examples: &[Reference]) -> Vec<Model> {
    examples
        .iter()
        .map(|r| r.action_key.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|key| train(key, examples))
        .collect()
}

/// Suggested KPI threshold derived from coach examples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    /// KPI key.
    pub kpi: String,
    /// Suggested lower bound.
    pub min: Option<f64>,
    /// Suggested upper bound.
    pub max: Option<f64>,
    /// Share of good examples passing the suggestion.
    pub good_pass: f64,
    /// Share of bad examples failing the suggestion.
    pub bad_fail: Option<f64>,
    /// Spanish explanation.
    pub rationale: String,
}

/// Suggest separating thresholds from marked good and bad examples.
pub fn suggest(action_key: &str, examples: &[Reference], current: &[Rule]) -> Vec<Suggestion> {
    let rows: Vec<&Reference> = examples
        .iter()
        .filter(|r| r.action_key == action_key)
        .collect();
    let keys: BTreeSet<String> = rows
        .iter()
        .flat_map(|r| r.measures.keys().cloned())
        .collect();
    keys.into_iter().filter_map(|k| {
        let good: Vec<f64> = rows.iter().filter(|r| r.good).filter_map(|r| r.measures.get(&k).copied().flatten()).collect();
        let bad: Vec<f64> = rows.iter().filter(|r| !r.good).filter_map(|r| r.measures.get(&k).copied().flatten()).collect();
        if good.len() < 3 { return None; }
        let rule = current.iter().find(|r| r.kpi == k);
        if bad.len() >= 3 {
            let mut union = good.iter().chain(&bad).copied().collect::<Vec<_>>(); union.sort_by(f64::total_cmp); union.dedup();
            let mut best: Option<(f64, bool, f64)> = None;
            for pair in union.windows(2) {
                let cut = (pair[0] + pair[1]) / 2.0;
                for high in [false, true] {
                    let gp = good.iter().filter(|v| if high { **v >= cut } else { **v <= cut }).count() as f64 / good.len() as f64;
                    let bf = bad.iter().filter(|v| if high { **v < cut } else { **v > cut }).count() as f64 / bad.len() as f64;
                    let score = (gp + bf) / 2.0;
                    if best.is_none_or(|b| score > b.0) { best = Some((score, high, cut)); }
                }
            }
            let (score, high, cut) = best?; if score < 0.7 { return None; }
            let gp = good.iter().filter(|v| if high { **v >= cut } else { **v <= cut }).count() as f64 / good.len() as f64;
            let bf = bad.iter().filter(|v| if high { **v < cut } else { **v > cut }).count() as f64 / bad.len() as f64;
            let cut = round_half(cut);
            let (min, max) = if high { (Some(cut), None) } else { (None, Some(cut)) };
            Some(Suggestion { kpi: k.clone(), min, max, good_pass: gp, bad_fail: Some(bf), rationale: format!(
                "El {:.0} % de vuestros ejemplos buenos tiene {} {} {}{}; el {:.0} % de los malos queda fuera",
                gp * 100.0,
                generic_label(&k),
                if high { "â‰¥" } else { "â‰¤" },
                eval::fmt(cut),
                eval::unit_suffix(&k),
                bf * 100.0
            ) })
        } else if let Some(rule) = rule {
            let (min, max) = if rule.min.is_some() { (Some(round_half(percentile(&good, 0.1))), None) } else if rule.max.is_some() { (None, Some(round_half(percentile(&good, 0.9)))) } else { return None };
            let pass = good.iter().filter(|v| min.is_none_or(|x| **v >= x) && max.is_none_or(|x| **v <= x)).count() as f64 / good.len() as f64;
            Some(Suggestion { kpi: k.clone(), min, max, good_pass: pass, bad_fail: None, rationale: format!(
                "El {:.0} % de vuestros ejemplos buenos tiene {} {} {}{}",
                pass * 100.0,
                generic_label(&k),
                if min.is_some() { "â‰¥" } else { "â‰¤" },
                eval::fmt(min.or(max).unwrap_or(0.0)),
                eval::unit_suffix(&k)
            ) })
        } else { None }
    }).collect()
}

/// Apply suggested bounds to a profile, adding a unit-weight rule if needed.
pub fn apply_suggestions(profile: &mut Profile, action_key: &str, suggestions: &[Suggestion]) {
    let rules = profile.rules.entry(action_key.to_owned()).or_default();
    for suggestion in suggestions {
        if let Some(rule) = rules.iter_mut().find(|r| r.kpi == suggestion.kpi) {
            rule.min = suggestion.min;
            rule.max = suggestion.max;
        } else {
            rules.push(Rule {
                kpi: suggestion.kpi.clone(),
                min: suggestion.min,
                max: suggestion.max,
                weight: 1.0,
                major: false,
            });
        }
    }
}

/// Combine a rules verdict with an optional learned prediction.
pub fn blend(rules_verdict: Verdict, prediction: Option<&Prediction>) -> Verdict {
    let Some(p) = prediction else {
        return rules_verdict;
    };
    match rules_verdict {
        Verdict::Unknown => Verdict::Unknown,
        Verdict::Bad => Verdict::Bad,
        Verdict::Good if p.p_good <= 0.3 => Verdict::Improvable,
        Verdict::Improvable if p.p_good >= 0.7 => Verdict::Good,
        Verdict::Improvable if p.p_good <= 0.3 => Verdict::Bad,
        _ => rules_verdict,
    }
}

fn outfield(s: &Snapshot, team: u8) -> impl Iterator<Item = &Placed> {
    s.players
        .iter()
        .filter(move |p| p.team == team && !p.keeper)
}
fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}
fn nearest_player(s: &Snapshot, ball: (f64, f64)) -> Option<(f64, u8)> {
    s.players
        .iter()
        .filter(|p| !p.keeper && (p.team == 1 || p.team == 2))
        .map(|p| (distance(p.at, ball), p.team))
        .min_by(|a, b| a.0.total_cmp(&b.0))
}
fn mean_values(values: impl Iterator<Item = f64>) -> Option<f64> {
    let v: Vec<f64> = values.collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}
fn nearest<'a>(snaps: &[&'a Snapshot], start: u64, sec: f64) -> Option<&'a Snapshot> {
    let target = start.saturating_add((sec * 1000.0) as u64);
    snaps
        .iter()
        .copied()
        .min_by_key(|s| s.ms.abs_diff(target))
        .filter(|s| s.ms.abs_diff(target) <= 1000)
}
fn corr(v: &[(f64, f64)]) -> f64 {
    let mx = v.iter().map(|p| p.0).sum::<f64>() / v.len() as f64;
    let my = v.iter().map(|p| p.1).sum::<f64>() / v.len() as f64;
    let (a, b, c) = v.iter().fold((0.0, 0.0, 0.0), |(a, b, c), (x, y)| {
        (
            a + (x - mx) * (y - my),
            b + (x - mx).powi(2),
            c + (y - my).powi(2),
        )
    });
    if b * c > 0.0 { a / (b * c).sqrt() } else { 0.0 }
}
fn spells(poss: &[(u64, Option<u8>)], team: u8) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut begin = None;
    let mut last = 0;
    for (ms, p) in poss {
        if *p == Some(team) {
            begin.get_or_insert(*ms);
            last = *ms;
        } else if let Some(s) = begin.take() {
            out.push((s, last));
        }
    }
    if let Some(s) = begin {
        out.push((s, last));
    }
    out
}
fn percentile(v: &[f64], p: f64) -> f64 {
    let mut values = v.to_vec();
    values.sort_by(f64::total_cmp);
    let index = p * (values.len().saturating_sub(1)) as f64;
    let lo = index.floor() as usize;
    let hi = index.ceil() as usize;
    values[lo] + (values[hi] - values[lo]) * (index - lo as f64)
}
fn round_half(v: f64) -> f64 {
    (v * 2.0).round() / 2.0
}
fn sigmoid(v: f64) -> f64 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}
fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
}
fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}
fn fit_logistic(
    action_key: &str,
    x: &[Vec<f64>],
    y: &[f64],
    means: &[f64],
    scales: &[f64],
    kpis: &[String],
    n_good: usize,
    n_bad: usize,
) -> Model {
    let mut weights = vec![0.0; kpis.len()];
    let mut bias = 0.0;
    let good_weight = x.len() as f64 / (2.0 * n_good as f64);
    let bad_weight = x.len() as f64 / (2.0 * n_bad as f64);
    // The penalty is a unit prior on the standardised weights, spread over
    // the examples: firm with a handful, fading as the coach marks more.
    let n = x.len() as f64;
    for _ in 0..4000 {
        let mut gw = vec![0.0; weights.len()];
        let mut gb = 0.0;
        for (row, target) in x.iter().zip(y) {
            let sample_weight = if *target == 1.0 {
                good_weight
            } else {
                bad_weight
            };
            let error = (sigmoid(bias + row.iter().zip(&weights).map(|(a, b)| a * b).sum::<f64>())
                - *target)
                * sample_weight;
            gb += error;
            for (g, value) in gw.iter_mut().zip(row) {
                *g += error * value;
            }
        }
        for (w, gradient) in weights.iter_mut().zip(gw) {
            *w -= 0.1 * (gradient + *w) / n;
        }
        bias -= 0.1 * gb / n;
    }
    Model {
        action_key: action_key.to_owned(),
        kpis: kpis.to_vec(),
        means: means.to_vec(),
        scales: scales.to_vec(),
        weights,
        bias,
        n_good,
        n_bad,
        loo_accuracy: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::football::eval::{Action, Sides, possession};

    fn team(team: u8, xs: &[f64]) -> Vec<Placed> {
        xs.iter()
            .enumerate()
            .map(|(n, x)| Placed {
                id: u32::from(team) * 100 + n as u32,
                team,
                keeper: false,
                role: None,
                at: (*x, 10.0 + n as f64 * 5.0),
            })
            .collect()
    }

    fn example(good: bool, block: f64, pressers: f64) -> Reference {
        Reference {
            action_key: "press_after_loss".into(),
            good,
            measures: Measures::from([
                ("block_length".into(), Some(block)),
                ("pressers_3s".into(), Some(pressers)),
            ]),
        }
    }

    /// Good presses keep the block near 30 m, bad ones let it stretch to 45.
    fn examples() -> Vec<Reference> {
        let mut out = Vec::new();
        for i in 0..6 {
            let wobble = f64::from(i) - 2.5;
            out.push(example(true, 30.0 + wobble, 4.0 + (wobble / 3.0).round()));
            out.push(example(
                false,
                45.0 + wobble,
                1.0 + (wobble / 3.0).round().abs(),
            ));
        }
        out
    }

    #[test]
    fn generic_measures_a_compact_block() {
        let window: Vec<Snapshot> = (0..10u64)
            .map(|i| {
                let mut players = team(
                    1,
                    &[20.0, 22.0, 24.0, 26.0, 30.0, 32.0, 34.0, 36.0, 40.0, 42.0],
                );
                players.extend(team(2, &[60.0; 10]));
                Snapshot {
                    ms: i * 500,
                    players,
                    ball: Some((41.0, 52.0)),
                }
            })
            .collect();
        let m = generic(&window, 1, Goal::Left);
        let block = m["block_length"].expect("block");
        assert!((15.0..=22.0).contains(&block), "{block}");
        assert!(m["pressers_10m"].expect("pressers") >= 2.0);
        assert_eq!(m["regain"], Some(1.0), "a team 1 player is on the ball");
    }

    #[test]
    fn a_lost_ball_opens_a_window() {
        // Team 1 has the ball, then team 2 takes it.
        let snaps: Vec<Snapshot> = (0..12u64)
            .map(|i| {
                let mut players = team(1, &[30.0; 10]);
                let mut theirs = team(2, &[70.0; 10]);
                if i < 6 {
                    players[0].at = (50.0, 30.0);
                } else {
                    theirs[0].at = (50.0, 30.0);
                }
                players.extend(theirs);
                Snapshot {
                    ms: i * 1000,
                    players,
                    ball: Some((50.2, 30.0)),
                }
            })
            .collect();
        let ctx = Context {
            possession: possession(&snaps),
            sides: Sides::of(&snaps),
            snaps,
        };
        let def = ActionDef {
            key: "cierre".into(),
            name: "Cierre tras pérdida".into(),
            trigger: Trigger::BallLost,
            window_s: 5.0,
            kpis: vec!["pressers_10m".into()],
        };
        let windows = find_windows(&ctx, &def, 1);
        assert_eq!(windows.len(), 1, "{windows:?}");
        assert_eq!(windows[0].end_ms - windows[0].start_ms, 5000);
        assert!(
            find_windows(&ctx, &def, 2).is_empty(),
            "team 2 never lost it"
        );
    }

    #[test]
    fn a_model_learns_the_coach_line() {
        let model = train("press_after_loss", &examples()).expect("enough examples");
        assert!(
            model.loo_accuracy.expect("loo") >= 0.9,
            "{:?}",
            model.loo_accuracy
        );
        let good = model.predict(&Measures::from([
            ("block_length".into(), Some(29.0)),
            ("pressers_3s".into(), Some(4.0)),
        ]));
        let bad = model.predict(&Measures::from([
            ("block_length".into(), Some(46.0)),
            ("pressers_3s".into(), Some(1.0)),
        ]));
        assert!(good.p_good > 0.8, "{}", good.p_good);
        assert!(bad.p_good < 0.2, "{}", bad.p_good);
        assert!(!model.explain(&bad).is_empty());
        let back: Model =
            serde_json::from_str(&serde_json::to_string(&model).expect("json")).expect("parse");
        assert_eq!(back.kpis, model.kpis);
        for (a, b) in back.weights.iter().zip(&model.weights) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn too_few_examples_teach_nothing() {
        assert!(train("press_after_loss", &examples()[..6]).is_none());
        assert!(
            train("build_up", &examples()).is_none(),
            "no examples of that action"
        );
    }

    #[test]
    fn suggestions_split_good_from_bad() {
        let found = suggest("press_after_loss", &examples(), &[]);
        let block = found
            .iter()
            .find(|s| s.kpi == "block_length")
            .expect("block");
        let max = block.max.expect("an upper bound");
        assert!((35.0..=40.0).contains(&max), "{max}");
        assert!(block.min.is_none());
        assert!(block.good_pass >= 0.99 && block.bad_fail.expect("bad") >= 0.99);
        assert!(block.rationale.contains("â‰¤"), "{}", block.rationale);
        let mut profile = Profile::presets().remove(0);
        apply_suggestions(&mut profile, Action::PressAfterLoss.key(), &found);
        let rule = profile.rules[Action::PressAfterLoss.key()]
            .iter()
            .find(|r| r.kpi == "block_length")
            .expect("rule");
        assert_eq!(rule.max, Some(max));
    }

    #[test]
    fn blending_never_overrules_a_major_failure() {
        let p = |p_good| Prediction {
            p_good,
            contributions: Vec::new(),
            missing: Vec::new(),
        };
        assert_eq!(blend(Verdict::Good, None), Verdict::Good);
        assert_eq!(blend(Verdict::Good, Some(&p(0.1))), Verdict::Improvable);
        assert_eq!(blend(Verdict::Improvable, Some(&p(0.9))), Verdict::Good);
        assert_eq!(blend(Verdict::Improvable, Some(&p(0.1))), Verdict::Bad);
        assert_eq!(blend(Verdict::Bad, Some(&p(0.99))), Verdict::Bad);
        assert_eq!(blend(Verdict::Unknown, Some(&p(0.99))), Verdict::Unknown);
    }

    #[test]
    fn actions_round_trip() {
        let def = ActionDef {
            key: "salida_3".into(),
            name: "Salida con tres".into(),
            trigger: Trigger::OwnPossessionDeep {
                max_depth: 30.0,
                min_s: 2.0,
            },
            window_s: 8.0,
            kpis: GENERIC_KPIS.iter().map(|k| k.to_string()).collect(),
        };
        let back: ActionDef =
            serde_json::from_str(&serde_json::to_string(&def).expect("json")).expect("parse");
        assert_eq!(back, def);
    }

    fn placed(id: u32, team: u8, at: (f64, f64), role: Option<crate::football::Role>) -> Placed {
        Placed {
            id,
            team,
            keeper: false,
            at,
            role,
        }
    }

    #[test]
    fn the_caja_and_the_roles_are_measured() {
        use crate::football::Role;
        // Ten defending players: eight inside the central box, two outside;
        // the ball wide, an interior nearest it; the pivot central.
        let mut players = vec![
            placed(1, 1, (20.0, 26.0), Some(Role::Central)),
            placed(2, 1, (20.0, 42.0), Some(Role::Central)),
            placed(3, 1, (22.0, 8.0), Some(Role::Lateral)),
            placed(4, 1, (22.0, 60.0), Some(Role::Lateral)),
            placed(5, 1, (30.0, 34.0), Some(Role::Pivote)),
            placed(6, 1, (35.0, 22.0), Some(Role::Interior)),
            placed(7, 1, (35.0, 46.0), Some(Role::Interior)),
            placed(8, 1, (42.0, 34.0), Some(Role::Mediapunta)),
            placed(9, 1, (48.0, 28.0), Some(Role::Delantera)),
            placed(10, 1, (48.0, 40.0), Some(Role::Delantera)),
        ];
        players.extend((0..10).map(|n| placed(20 + n, 2, (60.0, 10.0 + n as f64 * 5.0), None)));
        let window: Vec<Snapshot> = (0..6u64)
            .map(|i| Snapshot {
                ms: i * 500,
                players: players.clone(),
                ball: Some((36.0, 14.0)),
            })
            .collect();
        let m = generic(&window, 1, Goal::Left);
        assert_eq!(m["outside_box"], Some(2.0), "the two laterals are outside");
        assert_eq!(
            m["interiors_press_wide"],
            Some(1.0),
            "an interior is nearest the wide ball"
        );
        assert_eq!(m["pivot_central"], Some(1.0));
        let line = m["def_line_by_role"].expect("by role");
        assert!((line - 21.0).abs() < 1e-9, "{line}");
    }

    /// The library learned outside the window, the way the window learns
    /// it: `CONCAT_LEARN_SPEC` names a JSON list of `{dir, key, good,
    /// colour}` - a clip's analysis folder, its action, its side, and the
    /// kit colour of the team judged - and `CONCAT_LEARN_LIBRARY` the
    /// library folder whose references and models are extended.
    #[test]
    #[ignore]
    fn learns_a_library() {
        use crate::football::eval::{Action, Occurrence};
        use crate::football::{Roster, TrackSet, pitch::Calibration};
        let (Ok(spec), Ok(library)) = (
            std::env::var("CONCAT_LEARN_SPEC"),
            std::env::var("CONCAT_LEARN_LIBRARY"),
        ) else {
            return;
        };
        let library = Path::new(&library);
        let spec: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(spec).expect("spec")).expect("json");
        let refs_file = library.join("references.json");
        let mut refs: Vec<Reference> = std::fs::read(&refs_file)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let builtin = [
            Action::PressAfterLoss,
            Action::RecoveryRun,
            Action::BuildUp,
            Action::DefensiveBlock,
        ];
        let profile = &Profile::presets()[0];
        for item in spec {
            let dir = Path::new(item["dir"].as_str().expect("dir"));
            let key = item["key"].as_str().expect("key").to_string();
            let good = item["good"].as_bool().expect("good");
            let colour: Vec<f32> = item["colour"]
                .as_array()
                .expect("colour")
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            let set = TrackSet::load(dir).expect("tracks");
            let calibration = Calibration::load(dir);
            let players = Roster::load(dir).effective(&set);
            let ctx = Context::with_players(&set, &calibration, &players);
            let dist = |k: &[u8; 3]| {
                (0..3)
                    .map(|i| (f32::from(k[i]) - colour[i]).powi(2))
                    .sum::<f32>()
            };
            let team = if dist(&set.team_colours[0]) <= dist(&set.team_colours[1]) {
                1
            } else {
                2
            };
            let end_ms = set.samples.last().map_or(0, |s| s.ms);
            let window = Window {
                key: key.clone(),
                team,
                start_ms: 0,
                end_ms,
            };
            let Some((generic_m, seen)) = measure_window(&ctx, &window) else {
                println!("{}: sin medir", dir.display());
                continue;
            };
            let mut measures = match builtin.iter().find(|a| a.key() == key) {
                Some(action) => ctx
                    .assess(
                        &Occurrence {
                            action: *action,
                            team,
                            start_ms: 0,
                            end_ms,
                        },
                        profile,
                        &[],
                    )
                    .map(|(m, _)| m)
                    .unwrap_or_default(),
                None => Measures::new(),
            };
            for (k, v) in generic_m {
                measures.entry(k).or_insert(v);
            }
            println!(
                "{key} {} equipo {team} visto {:.0}% | {}",
                if good { "así sí" } else { "así no" },
                seen * 100.0,
                [
                    "pressers_10m",
                    "nearest_ball",
                    "behind_ball",
                    "block_length",
                    "width",
                    "support_15m",
                    "superiority_zone",
                    "progress_m",
                    "regain"
                ]
                .iter()
                .filter_map(|k| measures
                    .get(*k)
                    .copied()
                    .flatten()
                    .map(|v| format!("{k}={v:.1}")))
                .collect::<Vec<_>>()
                .join(" ")
            );
            refs.push(Reference {
                action_key: key,
                good,
                measures,
            });
        }
        let models = retrain_all(&refs);
        std::fs::write(&refs_file, serde_json::to_vec_pretty(&refs).expect("json")).expect("write");
        save_models(library, &models).expect("models");
        println!("{} referencias, {} modelos", refs.len(), models.len());
    }
}
