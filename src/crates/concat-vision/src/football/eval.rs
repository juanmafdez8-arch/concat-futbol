// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Evaluation of collective football actions against a game model and
//! coach-marked reference examples.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::pitch::{self, Calibration, Motion};
use super::{BALL, Kind, Player, Role, TrackSet};

/// A tracked player placed on the pitch, in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    /// The track id.
    pub id: u32,
    /// The team, 1 or 2.
    pub team: u8,
    /// Whether this player is a goalkeeper.
    pub keeper: bool,
    /// Position in pitch metres.
    pub at: (f64, f64),
    /// Posición registrada.
    pub role: Option<Role>,
}

/// One mapped instant of the match.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// Source time in milliseconds.
    pub ms: u64,
    /// Players mapped to the pitch.
    pub players: Vec<Placed>,
    /// Ball centre in pitch metres, when visible.
    pub ball: Option<(f64, f64)>,
}

/// Map tracking samples onto the pitch, omitting instants without calibration.
pub fn snapshots(set: &TrackSet, calibration: &Calibration) -> Vec<Snapshot> {
    snapshots_with(set, calibration, &BTreeMap::new())
}
/// Map tracking samples and attach registered roles.
pub fn snapshots_with(
    set: &TrackSet,
    calibration: &Calibration,
    roster: &BTreeMap<u32, Player>,
) -> Vec<Snapshot> {
    let motion = Motion::of(set);
    set.samples
        .iter()
        .filter_map(|sample| {
            let map = pitch::map_at(calibration, &motion, sample.ms)?;
            let mut players = Vec::new();
            let mut ball = None;
            for item in &sample.boxes {
                match item.kind {
                    Kind::Player | Kind::Goalkeeper if item.team == 1 || item.team == 2 => {
                        if let Some(at) = pitch::apply(&map, pitch::feet(&item.bbox))
                            && in_bounds(at)
                        {
                            players.push(Placed {
                                id: item.id,
                                team: item.team,
                                keeper: item.kind == Kind::Goalkeeper,
                                at,
                                role: roster.get(&item.id).and_then(|p| p.role),
                            });
                        }
                    }
                    Kind::Ball if item.id == BALL || item.kind == Kind::Ball => {
                        if let Some(at) = pitch::apply(&map, pitch::feet(&item.bbox))
                            && in_bounds(at)
                        {
                            ball = Some(at);
                        }
                    }
                    _ => {}
                }
            }
            Some(Snapshot {
                ms: sample.ms,
                players,
                ball,
            })
        })
        .collect()
}

fn in_bounds((x, y): (f64, f64)) -> bool {
    (-3.0..=108.0).contains(&x) && (-3.0..=71.0).contains(&y)
}

/// A goal line defended by a team.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Goal {
    /// x = 0 m.
    Left,
    /// x = 105 m.
    Right,
}

/// Find a team's defended end, preferring its goalkeeper's average position.
pub fn defended_goal(snaps: &[Snapshot], team: u8) -> Option<Goal> {
    let keepers: Vec<f64> = snaps
        .iter()
        .flat_map(|s| s.players.iter())
        .filter(|p| p.team == team && p.keeper)
        .map(|p| p.at.0)
        .collect();
    let own = if keepers.is_empty() {
        let xs: Vec<f64> = snaps
            .iter()
            .flat_map(|s| s.players.iter())
            .filter(|p| p.team == team)
            .map(|p| p.at.0)
            .collect();
        let other: Vec<f64> = snaps
            .iter()
            .flat_map(|s| s.players.iter())
            .filter(|p| p.team != team && p.team != 0)
            .map(|p| p.at.0)
            .collect();
        if xs.is_empty() || other.is_empty() {
            return None;
        }
        if mean(&xs) < mean(&other) {
            Goal::Left
        } else {
            Goal::Right
        }
    } else if mean(&keepers) < 52.5 {
        Goal::Left
    } else {
        Goal::Right
    };
    Some(own)
}

/// Which end each team defends, instant by instant. The teams change ends
/// at half time, so a match is not one answer: each instant reads the ten
/// minutes either side of it, where the team whose outfield players stand
/// nearer an end on average is the one defending it. Kept as minute buckets
/// of summed x, so asking is cheap however long the match.
pub struct Sides {
    /// Per minute: summed x and count for teams 1 and 2.
    buckets: Vec<([f64; 2], [usize; 2])>,
}

impl Sides {
    const BUCKET_MS: u64 = 60_000;
    const REACH: usize = 10;

    /// The buckets of a match's snapshots.
    pub fn of(snaps: &[Snapshot]) -> Sides {
        let len = snaps
            .last()
            .map_or(0, |s| (s.ms / Self::BUCKET_MS) as usize + 1);
        let mut buckets = vec![([0.0; 2], [0usize; 2]); len];
        for snap in snaps {
            let bucket = &mut buckets[(snap.ms / Self::BUCKET_MS) as usize];
            for p in snap.players.iter().filter(|p| !p.keeper) {
                if let Some(t) = p.team.checked_sub(1).map(usize::from).filter(|t| *t < 2) {
                    bucket.0[t] += p.at.0;
                    bucket.1[t] += 1;
                }
            }
        }
        Sides { buckets }
    }

    /// The end `team` defends around `ms`, if both teams were seen near it.
    pub fn goal(&self, team: u8, ms: u64) -> Option<Goal> {
        let t = usize::from(team.checked_sub(1)?);
        if t > 1 || self.buckets.is_empty() {
            return None;
        }
        let at = ((ms / Self::BUCKET_MS) as usize).min(self.buckets.len() - 1);
        let from = at.saturating_sub(Self::REACH);
        let to = (at + Self::REACH).min(self.buckets.len() - 1);
        let (mut sum, mut count) = ([0.0; 2], [0usize; 2]);
        for (s, n) in &self.buckets[from..=to] {
            for i in 0..2 {
                sum[i] += s[i];
                count[i] += n[i];
            }
        }
        if count[0] == 0 || count[1] == 0 {
            return None;
        }
        let mine = sum[t] / count[t] as f64;
        let theirs = sum[1 - t] / count[1 - t] as f64;
        Some(if mine < theirs {
            Goal::Left
        } else {
            Goal::Right
        })
    }
}

/// Distance from a team's own goal line, in metres.
pub fn depth(goal: Goal, x: f64) -> f64 {
    match goal {
        Goal::Left => x,
        Goal::Right => 105.0 - x,
    }
}

/// A possession change, after the new team has held the ball for two instants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Change {
    /// Change time in milliseconds.
    pub ms: u64,
    /// Previous team.
    pub from: u8,
    /// New team.
    pub to: u8,
}

/// Infer the nearest outfield player's team, retaining the incumbent until
/// another team is nearest for two consecutive mapped snapshots.
pub fn possession(snaps: &[Snapshot]) -> Vec<(u64, Option<u8>)> {
    let mut result = Vec::with_capacity(snaps.len());
    let (mut current, mut candidate, mut candidate_count) = (None, None, 0usize);
    for snap in snaps {
        let raw = snap
            .ball
            .and_then(|ball| {
                snap.players
                    .iter()
                    .filter(|p| !p.keeper && (p.team == 1 || p.team == 2))
                    .map(|p| ((p.at.0 - ball.0).hypot(p.at.1 - ball.1), p.team))
                    .min_by(|a, b| a.0.total_cmp(&b.0))
            })
            .filter(|(d, _)| *d <= 2.5)
            .map(|(_, team)| team);
        if raw.is_none() || raw == current {
            candidate = None;
            candidate_count = 0;
        } else if raw == candidate {
            candidate_count += 1;
            if candidate_count >= 2 {
                current = raw;
                candidate = None;
                candidate_count = 0;
            }
        } else {
            candidate = raw;
            candidate_count = 1;
        }
        result.push((snap.ms, current));
    }
    result
}

/// Return changes between known possession teams.
pub fn changes(possession: &[(u64, Option<u8>)]) -> Vec<Change> {
    let mut prior = None;
    possession
        .iter()
        .filter_map(|(ms, team)| {
            let out = match (prior, *team) {
                (Some(from), Some(to)) if from != to => Some(Change { ms: *ms, from, to }),
                _ => None,
            };
            if team.is_some() {
                prior = *team;
            }
            out
        })
        .collect()
}

/// A team's per-instant shape and observability.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    /// Number of outfield players observed.
    pub count: usize,
    /// 90th minus 10th percentile depth, in metres.
    pub block_length: Option<f64>,
    /// 90th minus 10th percentile y, in metres.
    pub width: Option<f64>,
    /// Mean depth of the four deepest players, in metres.
    pub def_line_height: Option<f64>,
    /// Mean depth of the next four players, in metres.
    pub midfield_line_height: Option<f64>,
    /// Difference between the two line heights, in metres.
    pub line_gap: Option<f64>,
    /// Team centroid in metres.
    pub centroid: Option<(f64, f64)>,
    /// Outfield players within the supplied radius of the ball.
    pub near_ball: Option<usize>,
    /// Players goal-side of the ball.
    pub behind_ball: Option<usize>,
    /// Fractional observability, capped at one.
    pub observability: f64,
}

/// Measure one team's shape at one instant.
pub fn shape(snap: &Snapshot, team: u8, goal: Goal, radius: f64) -> Shape {
    let players: Vec<&Placed> = snap
        .players
        .iter()
        .filter(|p| p.team == team && !p.keeper)
        .collect();
    let depths: Vec<f64> = players.iter().map(|p| depth(goal, p.at.0)).collect();
    let ys: Vec<f64> = players.iter().map(|p| p.at.1).collect();
    let mut ordered = depths.clone();
    ordered.sort_by(f64::total_cmp);
    // The back line by role when the analyst named enough of it; else the
    // four nearest the own goal.
    let role_depths: Vec<f64> = players
        .iter()
        .filter(|p| {
            p.role
                .is_some_and(|r| matches!(r, Role::Central | Role::Lateral))
        })
        .map(|p| depth(goal, p.at.0))
        .collect();
    let def = if role_depths.len() >= 3 {
        Some(mean(&role_depths))
    } else {
        (!ordered.is_empty())
            .then(|| ordered.iter().take(4).sum::<f64>() / ordered.len().min(4) as f64)
    };
    let mid = (ordered.len() > 4)
        .then(|| ordered.iter().skip(4).take(4).sum::<f64>() / (ordered.len() - 4).min(4) as f64);
    let centroid = (!players.is_empty()).then(|| {
        (
            players.iter().map(|p| p.at.0).sum::<f64>() / players.len() as f64,
            players.iter().map(|p| p.at.1).sum::<f64>() / players.len() as f64,
        )
    });
    let ball_depth = snap.ball.map(|b| depth(goal, b.0));
    Shape {
        count: players.len(),
        block_length: spread(&depths),
        width: spread(&ys),
        def_line_height: def,
        midfield_line_height: mid,
        line_gap: def.zip(mid).map(|(a, b)| (a - b).abs()),
        centroid,
        near_ball: snap.ball.map(|b| {
            players
                .iter()
                .filter(|p| (p.at.0 - b.0).hypot(p.at.1 - b.1) <= radius)
                .count()
        }),
        behind_ball: ball_depth.map(|bd| depths.iter().filter(|d| **d < bd).count()),
        observability: (players.len() as f64 / 10.0).min(1.0),
    }
}

fn spread(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        Some(percentile(&v, 0.9) - percentile(&v, 0.1))
    }
}
fn percentile(v: &[f64], p: f64) -> f64 {
    if v.len() == 1 {
        v[0]
    } else {
        let i = p * (v.len() - 1) as f64;
        let lo = i.floor() as usize;
        v[lo] + (v[i.ceil() as usize] - v[lo]) * (i - lo as f64)
    }
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// The collective action being evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    /// Immediate pressure after losing possession.
    PressAfterLoss,
    /// Possession in the defensive build-up.
    BuildUp,
    /// Defensive block and lateral movement.
    DefensiveBlock,
    /// Recovery after a turnover.
    RecoveryRun,
}
impl Action {
    /// Stable persisted action identifier.
    pub fn key(self) -> &'static str {
        match self {
            Self::PressAfterLoss => "press_after_loss",
            Self::BuildUp => "build_up",
            Self::DefensiveBlock => "defensive_block",
            Self::RecoveryRun => "recovery_run",
        }
    }
    /// Spanish action name for display.
    pub fn name(self) -> &'static str {
        match self {
            Self::PressAfterLoss => "Presión tras pérdida",
            Self::BuildUp => "Salida de balón",
            Self::DefensiveBlock => "Bloque defensivo / basculación",
            Self::RecoveryRun => "Repliegue",
        }
    }
}

/// A detected action window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occurrence {
    /// Action type.
    pub action: Action,
    /// Team being judged.
    pub team: u8,
    /// Window start in milliseconds.
    pub start_ms: u64,
    /// Window end in milliseconds.
    pub end_ms: u64,
}

/// Find turnover, build-up and defensive-block windows.
pub fn find(snaps: &[Snapshot], poss: &[(u64, Option<u8>)], sides: &Sides) -> Vec<Occurrence> {
    let mut out = Vec::new();
    let cs = changes(poss);
    for c in cs {
        out.push(Occurrence {
            action: Action::PressAfterLoss,
            team: c.from,
            start_ms: c.ms,
            end_ms: c.ms + 6000,
        });
        out.push(Occurrence {
            action: Action::RecoveryRun,
            team: c.from,
            start_ms: c.ms,
            end_ms: c.ms + 10000,
        });
    }
    for team in [1, 2] {
        let mut start = None;
        let mut last_emit = 0u64;
        for (i, (ms, p)) in poss.iter().enumerate() {
            let deep = *p == Some(team)
                && snaps
                    .get(i)
                    .and_then(|s| s.ball)
                    .zip(sides.goal(team, *ms))
                    .is_some_and(|(b, g)| depth(g, b.0) < 35.0);
            if deep {
                let s = *start.get_or_insert(*ms);
                if ms.saturating_sub(s) >= 2000
                    && (last_emit == 0 || ms.saturating_sub(last_emit) >= 10000)
                {
                    out.push(Occurrence {
                        action: Action::BuildUp,
                        team,
                        start_ms: s,
                        end_ms: s + 8000,
                    });
                    last_emit = *ms;
                }
            } else {
                start = None;
            }
        }
    }
    for defending in [1, 2] {
        let opponent = 3 - defending;
        let mut start: Option<u64> = None;
        let mut last: Option<u64> = None;
        let sentinel = (u64::MAX, None);
        for (ms, p) in poss.iter().chain(std::iter::once(&sentinel)) {
            if *p == Some(opponent) {
                if start.is_none() {
                    start = Some(*ms);
                }
                last = Some(*ms);
                if let Some(s) = start
                    && ms.saturating_sub(s) >= 15000
                {
                    out.push(Occurrence {
                        action: Action::DefensiveBlock,
                        team: defending,
                        start_ms: s,
                        end_ms: s + 15000,
                    });
                    start = None;
                    last = None;
                }
            } else if let (Some(s), Some(end)) = (start, last) {
                if end.saturating_sub(s) >= 4000 {
                    out.push(Occurrence {
                        action: Action::DefensiveBlock,
                        team: defending,
                        start_ms: s,
                        end_ms: end.min(s + 15000),
                    });
                }
                start = None;
                last = None;
            }
        }
    }
    out.sort_by_key(|o| (o.start_ms, o.action));
    out
}

/// KPI values in their documented units; absent values are unobservable.
pub type Measures = BTreeMap<String, Option<f64>>;

/// Measure an action window using only snapshots with adequate visibility.
pub fn measure(action: Action, team: u8, snaps: &[Snapshot], goal: Goal) -> Measures {
    let mut m = Measures::new();
    // Times are from the window's start, not from its first visible instant.
    let start = snaps.first().map_or(0, |s| s.ms);
    let good: Vec<&Snapshot> = snaps
        .iter()
        .filter(|s| shape(s, team, goal, 20.0).observability >= 0.6)
        .collect();
    let put = |m: &mut Measures, k: &str, v: Option<f64>| {
        m.insert(k.into(), v);
    };
    match action {
        Action::PressAfterLoss => {
            let at = |sec: f64| nearest_time(&good, start, sec);
            let p3 = at(3.0);
            put(
                &mut m,
                "pressers_3s",
                p3.and_then(|s| {
                    s.ball.map(|b| {
                        s.players
                            .iter()
                            .filter(|p| {
                                p.team == team
                                    && !p.keeper
                                    && (p.at.0 - b.0).hypot(p.at.1 - b.1) <= 10.0
                            })
                            .count() as f64
                    })
                }),
            );
            put(
                &mut m,
                "nearest_2s",
                at(2.0).and_then(|s| {
                    s.ball.and_then(|b| {
                        s.players
                            .iter()
                            .filter(|p| p.team == team && !p.keeper)
                            .map(|p| (p.at.0 - b.0).hypot(p.at.1 - b.1))
                            .min_by(f64::total_cmp)
                    })
                }),
            );
            // Won back: a second on at least, and the player nearest the
            // ball, of either side, is ours and on it.
            let regained = good.iter().find(|s| {
                s.ms >= start + 1000
                    && s.ball.is_some_and(|b| {
                        s.players
                            .iter()
                            .filter(|p| !p.keeper)
                            .map(|p| ((p.at.0 - b.0).hypot(p.at.1 - b.1), p.team))
                            .min_by(|a, b| a.0.total_cmp(&b.0))
                            .is_some_and(|(d, t)| t == team && d <= 2.5)
                    })
            });
            put(
                &mut m,
                "regain_s",
                Some(regained.map_or(6.0, |s| s.ms.saturating_sub(start) as f64 / 1000.0)),
            );
            put(
                &mut m,
                "regained",
                Some(if regained.is_some() { 1.0 } else { 0.0 }),
            );
            put(
                &mut m,
                "block_length",
                average(&good, |s| shape(s, team, goal, 10.0).block_length),
            );
        }
        Action::RecoveryRun => {
            for (sec, key) in [(5.0, "behind_ball_5s"), (8.0, "behind_ball_8s")] {
                let s = nearest_time(&good, start, sec);
                put(
                    &mut m,
                    key,
                    s.and_then(|s| shape(s, team, goal, 10.0).behind_ball.map(|n| n as f64)),
                );
            }
            let reorganise = good.iter().find(|s| {
                shape(s, team, goal, 10.0)
                    .behind_ball
                    .is_some_and(|n| n >= 7)
            });
            put(
                &mut m,
                "reorganise_s",
                reorganise.map(|s| s.ms.saturating_sub(start) as f64 / 1000.0),
            );
            put(
                &mut m,
                "block_length_end",
                good.last()
                    .and_then(|s| shape(s, team, goal, 10.0).block_length),
            );
        }
        Action::BuildUp => {
            put(
                &mut m,
                "width",
                average(&good, |s| shape(s, team, goal, 10.0).width),
            );
            put(
                &mut m,
                "def_line_height",
                average(&good, |s| shape(s, team, goal, 10.0).def_line_height),
            );
            put(
                &mut m,
                "support_15m",
                average(&good, |s| {
                    s.ball.map(|b| {
                        s.players
                            .iter()
                            .filter(|p| {
                                p.team == team
                                    && !p.keeper
                                    && (p.at.0 - b.0).hypot(p.at.1 - b.1) <= 15.0
                            })
                            .count() as f64
                    })
                }),
            );
            put(
                &mut m,
                "superiority_zone",
                average(&good, |s| {
                    s.ball.map(|b| {
                        let n = |t| {
                            s.players
                                .iter()
                                .filter(|p| {
                                    p.team == t
                                        && !p.keeper
                                        && (p.at.0 - b.0).hypot(p.at.1 - b.1) <= 20.0
                                })
                                .count() as f64
                        };
                        n(team) - n(3 - team)
                    })
                }),
            );
            let first = good.first().and_then(|s| s.ball);
            let last = good.last().and_then(|s| s.ball);
            put(
                &mut m,
                "progress_m",
                first
                    .zip(last)
                    .map(|(a, b)| depth(goal, b.0) - depth(goal, a.0)),
            );
        }
        Action::DefensiveBlock => {
            put(
                &mut m,
                "block_length",
                average(&good, |s| shape(s, team, goal, 10.0).block_length),
            );
            put(
                &mut m,
                "width",
                average(&good, |s| shape(s, team, goal, 10.0).width),
            );
            put(
                &mut m,
                "def_line_height",
                average(&good, |s| shape(s, team, goal, 10.0).def_line_height),
            );
            put(
                &mut m,
                "line_gap",
                average(&good, |s| shape(s, team, goal, 10.0).line_gap),
            );
            let pairs: Vec<(f64, f64)> = good
                .iter()
                .filter_map(|s| Some((shape(s, team, goal, 10.0).centroid?.1, s.ball?.1)))
                .collect();
            put(
                &mut m,
                "shift_follow",
                (pairs.len() >= 5).then(|| correlation(&pairs)),
            );
        }
    }
    m
}
/// The visible instant nearest `sec` after `start`, if one is within a
/// second of it: further off, it measures another moment.
fn nearest_time<'a>(v: &[&'a Snapshot], start: u64, sec: f64) -> Option<&'a Snapshot> {
    let target = start + (sec * 1000.0) as u64;
    v.iter()
        .copied()
        .min_by_key(|s| s.ms.abs_diff(target))
        .filter(|s| s.ms.abs_diff(target) <= 1000)
}
fn average(v: &[&Snapshot], f: impl Fn(&Snapshot) -> Option<f64>) -> Option<f64> {
    let xs: Vec<f64> = v.iter().filter_map(|s| f(s)).collect();
    (!xs.is_empty()).then(|| mean(&xs))
}
fn correlation(v: &[(f64, f64)]) -> f64 {
    let (mx, my) = (
        mean(&v.iter().map(|p| p.0).collect::<Vec<_>>()),
        mean(&v.iter().map(|p| p.1).collect::<Vec<_>>()),
    );
    let (a, b, c) = v.iter().fold((0.0, 0.0, 0.0), |(a, b, c), (x, y)| {
        (
            a + (x - mx) * (y - my),
            b + (x - mx).powi(2),
            c + (y - my).powi(2),
        )
    });
    if b * c > 0.0 { a / (b * c).sqrt() } else { 0.0 }
}

/// A threshold rule for one KPI.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    /// KPI key.
    pub kpi: String,
    /// Inclusive lower limit.
    pub min: Option<f64>,
    /// Inclusive upper limit.
    pub max: Option<f64>,
    /// Weight in the rule score.
    pub weight: f64,
    /// Whether failure alone means a bad verdict.
    pub major: bool,
}
/// A named tactical game model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// Profile name.
    pub name: String,
    /// Rules by stable action key.
    pub rules: BTreeMap<String, Vec<Rule>>,
}
impl Profile {
    /// Two useful Spanish game-model starting points.
    pub fn presets() -> Vec<Profile> {
        let rule = |k: &str, min, max, weight, major| Rule {
            kpi: k.into(),
            min,
            max,
            weight,
            major,
        };
        let mut high = BTreeMap::new();
        high.insert(
            "press_after_loss".into(),
            vec![
                rule("pressers_3s", Some(3.0), None, 2.0, true),
                rule("nearest_2s", None, Some(5.0), 1.0, false),
                rule("regain_s", None, Some(6.0), 1.0, false),
                rule("block_length", None, Some(35.0), 1.0, false),
            ],
        );
        high.insert(
            "build_up".into(),
            vec![rule("width", Some(40.0), None, 1.0, false)],
        );
        high.insert(
            "defensive_block".into(),
            vec![
                rule("block_length", None, Some(35.0), 1.0, false),
                rule("def_line_height", Some(35.0), None, 1.0, false),
                rule("line_gap", None, Some(12.0), 1.0, false),
                rule("shift_follow", Some(0.5), None, 1.0, false),
            ],
        );
        high.insert(
            "recovery_run".into(),
            vec![
                rule("behind_ball_5s", Some(6.0), None, 2.0, true),
                rule("reorganise_s", None, Some(8.0), 1.0, false),
            ],
        );
        let mut mid = high.clone();
        for r in mid.get_mut("press_after_loss").into_iter().flatten() {
            if r.kpi == "block_length" {
                r.max = Some(40.0)
            }
        }
        for r in mid.get_mut("defensive_block").into_iter().flatten() {
            if r.kpi == "block_length" {
                r.max = Some(40.0)
            }
            if r.kpi == "def_line_height" {
                r.min = Some(25.0)
            }
        }
        vec![
            Profile {
                name: "Presión alta (1-4-3-3)".into(),
                rules: high,
            },
            Profile {
                name: "Bloque medio (1-4-4-2)".into(),
                rules: mid,
            },
        ]
    }
    /// Load profiles from `profiles.json`, falling back to built-in presets.
    pub fn load(dir: &Path) -> Vec<Profile> {
        std::fs::read(dir.join("profiles.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_else(Self::presets)
    }
    /// Save profiles as a JSON list in the supplied directory.
    pub fn save_all(dir: &Path, profiles: &[Profile]) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let bytes = serde_json::to_vec_pretty(profiles).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("profiles.json"), bytes).map_err(|e| e.to_string())
    }
}

/// A coach-marked good or bad reference example.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    /// Stable action key.
    pub action_key: String,
    /// Whether the example is good.
    pub good: bool,
    /// Measured KPIs.
    pub measures: Measures,
}
/// Distance from good references and the nearest bad example.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    /// Root mean square robust z-score.
    pub distance: f64,
    /// Largest KPI deviations, descending by magnitude.
    pub worst: Vec<(String, f64, f64)>,
    /// Number of good examples.
    pub n_good: usize,
    /// Distance to nearest bad reference.
    pub nearest_bad: Option<f64>,
}
/// Compare measures to the good references for the same action.
pub fn reference_fit(measures: &Measures, refs: &[Reference]) -> Option<Fit> {
    let good: Vec<&Reference> = refs.iter().filter(|r| r.good).collect();
    if good.len() < 3 {
        return None;
    }
    let mut norms = BTreeMap::new();
    for (key, value) in measures {
        let Some(value) = value else { continue };
        let mut vals: Vec<f64> = good
            .iter()
            .filter_map(|r| r.measures.get(key).and_then(|v| *v))
            .collect();
        if vals.len() < 3 {
            continue;
        }
        vals.sort_by(f64::total_cmp);
        let med = median(&vals);
        let mut dev: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
        dev.sort_by(f64::total_cmp);
        let scale = (median(&dev) * 1.4826).max(med.abs() * 0.05).max(0.5);
        norms.insert(key.clone(), ((*value - med) / scale, med));
    }
    if norms.is_empty() {
        return None;
    }
    let rms = |r: &Reference| -> Option<f64> {
        let zs: Vec<f64> = norms
            .iter()
            .filter_map(|(k, _)| {
                r.measures.get(k).and_then(|v| *v).map(|v| {
                    let (_, med) = norms[k];
                    let mut vals: Vec<f64> = good
                        .iter()
                        .filter_map(|g| g.measures.get(k).and_then(|x| *x))
                        .collect();
                    vals.sort_by(f64::total_cmp);
                    let mut d: Vec<f64> = vals.iter().map(|x| (x - med).abs()).collect();
                    d.sort_by(f64::total_cmp);
                    (v - med) / (median(&d) * 1.4826).max(med.abs() * 0.05).max(0.5)
                })
            })
            .collect();
        (!zs.is_empty()).then(|| (zs.iter().map(|z| z * z).sum::<f64>() / zs.len() as f64).sqrt())
    };
    let distance = (norms.values().map(|(z, _)| z * z).sum::<f64>() / norms.len() as f64).sqrt();
    let mut worst: Vec<_> = norms
        .iter()
        .map(|(k, (z, m))| (k.clone(), *z, *m))
        .collect();
    worst.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    let nearest_bad = refs
        .iter()
        .filter(|r| !r.good)
        .filter_map(rms)
        .min_by(f64::total_cmp);
    Some(Fit {
        distance,
        worst,
        n_good: good.len(),
        nearest_bad,
    })
}
fn median(v: &[f64]) -> f64 {
    if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) * 0.5
    }
}

/// Rule evaluation details.
#[derive(Clone, Debug, PartialEq)]
pub struct RuleOutcome {
    /// Number of passing rules.
    pub passed: usize,
    /// Failed rule and observed value.
    pub failed: Vec<(Rule, Option<f64>)>,
    /// Missing KPI keys.
    pub unobservable: Vec<String>,
}
/// A tactical verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// Meets the model.
    Good,
    /// Some improvement is needed.
    Improvable,
    /// Major or many failures.
    Bad,
    /// Insufficient observations.
    Unknown,
}
impl Verdict {
    /// Spanish label for display.
    pub fn label(self) -> &'static str {
        match self {
            Self::Good => "Bien",
            Self::Improvable => "Mejorable",
            Self::Bad => "Mal",
            Self::Unknown => "Sin datos",
        }
    }
}
/// Combined rule, reference and observability judgement.
#[derive(Clone, Debug, PartialEq)]
pub struct Judgement {
    /// Rule result.
    pub rules: RuleOutcome,
    /// Similarity to coach examples.
    pub reference: Option<Fit>,
    /// Fractional observability.
    pub observability: f64,
    /// Final verdict.
    pub verdict: Verdict,
    /// Short Spanish explanations.
    pub reasons: Vec<String>,
}
/// Spanish label for a KPI key.
pub fn kpi_label(key: &str) -> &'static str {
    match key {
        "pressers_3s" => "presionando a los 3 s",
        "nearest_2s" => "distancia del más cercano a los 2 s",
        "regain_s" => "tiempo de recuperación",
        "regained" => "recuperación",
        "block_length" => "longitud del bloque",
        "behind_ball_5s" => "jugadores tras el balón a los 5 s",
        "behind_ball_8s" => "jugadores tras el balón a los 8 s",
        "reorganise_s" => "tiempo para reorganizarse",
        "block_length_end" => "longitud final del bloque",
        "width" => "anchura",
        "def_line_height" => "altura de la línea defensiva",
        "support_15m" => "apoyos a 15 m",
        "superiority_zone" => "superioridad en zona",
        "progress_m" => "progresión",
        "line_gap" => "distancia entre líneas",
        "shift_follow" => "seguimiento lateral",
        "outside_box" => "jugadoras fuera de la caja",
        "interiors_press_wide" => "interiores o laterales presionando fuera",
        "pivot_central" => "pivote en zona central",
        "def_line_by_role" => "altura de centrales y laterales",
        _ => "indicador",
    }
}

/// Judge measured KPIs against rules and coach examples.
pub fn judge(
    action: Action,
    measures: &Measures,
    profile: &Profile,
    refs: &[Reference],
    observability: f64,
) -> Judgement {
    judge_key(action.key(), measures, profile, refs, observability)
}

/// [`judge`] for any action by its key: one of the four, or one the coach
/// defined, whose rules the profile keeps under the same key.
pub fn judge_key(
    key: &str,
    measures: &Measures,
    profile: &Profile,
    refs: &[Reference],
    observability: f64,
) -> Judgement {
    let rules = profile.rules.get(key).cloned().unwrap_or_default();
    let mut outcome = RuleOutcome {
        passed: 0,
        failed: Vec::new(),
        unobservable: Vec::new(),
    };
    let (mut total, mut failed_weight) = (0.0, 0.0);
    let mut major = false;
    let mut reasons = Vec::new();
    for rule in rules {
        total += rule.weight.max(0.0);
        let value = measures.get(&rule.kpi).copied().flatten();
        match value {
            None => {
                outcome.unobservable.push(rule.kpi.clone());
                reasons.push(format!("No observable: {}", kpi_label(&rule.kpi)));
            }
            Some(v) => {
                let failed =
                    rule.min.is_some_and(|min| v < min) || rule.max.is_some_and(|max| v > max);
                if failed {
                    failed_weight += rule.weight.max(0.0);
                    major |= rule.major;
                    outcome.failed.push((rule.clone(), Some(v)));
                    let bound = if rule.min.is_some_and(|x| v < x) {
                        format!("mín. {}", fmt(rule.min.map_or(0.0, |x| x)))
                    } else {
                        format!("máx. {}", fmt(rule.max.map_or(0.0, |x| x)))
                    };
                    let unit = unit(&rule.kpi);
                    let unit = if unit.is_empty() {
                        String::new()
                    } else {
                        format!(" {unit}")
                    };
                    reasons.push(format!("{} ({bound}{unit})", format_value(&rule.kpi, v)));
                } else {
                    outcome.passed += 1;
                }
            }
        }
    }
    let fit = reference_fit(measures, refs);
    let mut verdict = if observability < 0.5
        || outcome.unobservable.len() * 2
            > outcome.unobservable.len() + outcome.passed + outcome.failed.len()
    {
        Verdict::Unknown
    } else if major || total > 0.0 && failed_weight / total > 0.34 {
        Verdict::Bad
    } else if failed_weight > 0.0 {
        Verdict::Improvable
    } else {
        Verdict::Good
    };
    if let Some(f) = &fit
        && (f.distance > 2.5 || f.nearest_bad.is_some_and(|d| d < f.distance))
    {
        {
            if verdict == Verdict::Good {
                verdict = Verdict::Improvable;
            }
            if let Some((k, z, m)) = f.worst.first() {
                let side = if *z > 0.0 { "por encima" } else { "por debajo" };
                reasons.push(format!(
                    "Lejos de las referencias: {} {side} de lo habitual (ref. {} {})",
                    kpi_label(k),
                    fmt(*m),
                    unit(k)
                ));
            }
        }
    }
    Judgement {
        rules: outcome,
        reference: fit,
        observability,
        verdict,
        reasons,
    }
}
/// Whole numbers bare, the rest to a decimal: a correlation of 0.5 is
/// not 0, and 35 m is not 35.0.
pub fn fmt(v: f64) -> String {
    if (v - v.round()).abs() < 0.05 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}
/// A KPI's unit after a space, or nothing for a count or a ratio.
pub fn unit_suffix(k: &str) -> &'static str {
    match unit(k) {
        "m" => " m",
        "s" => " s",
        _ => "",
    }
}

fn unit(k: &str) -> &'static str {
    match k {
        "block_length" | "block_length_end" | "nearest_2s" | "width" | "def_line_height"
        | "line_gap" | "progress_m" | "centroid_depth" | "nearest_ball" => "m",
        "def_line_by_role" => "m",
        "regain_s" | "reorganise_s" => "s",
        _ => "",
    }
}
fn format_value(k: &str, v: f64) -> String {
    match k {
        "block_length" | "block_length_end" => format!("Bloque {:.0} m", v),
        "pressers_3s" => format!("Solo {:.0} jugador(es) presionando a los 3 s", v),
        _ => {
            let label = kpi_label(k);
            let mut chars = label.chars();
            let label: String = chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect())
                .unwrap_or_default();
            let unit = unit(k);
            if unit.is_empty() {
                format!("{label} {}", fmt(v))
            } else {
                format!("{label} {} {unit}", fmt(v))
            }
        }
    }
}

/// A match read once onto the pitch: its instants, who had the ball, and
/// which end each side defended - what every window is measured against.
pub struct Context {
    /// The calibrated instants.
    pub snaps: Vec<Snapshot>,
    /// Possession at each instant.
    pub possession: Vec<(u64, Option<u8>)>,
    /// The ends, through the match.
    pub sides: Sides,
}

impl Context {
    /// Reads a track set through its calibration.
    pub fn new(set: &TrackSet, calibration: &Calibration) -> Context {
        Self::with_players(set, calibration, &BTreeMap::new())
    }
    /// Reads a track set and attaches player roles.
    pub fn with_players(
        set: &TrackSet,
        calibration: &Calibration,
        players: &BTreeMap<u32, Player>,
    ) -> Context {
        let snaps = snapshots_with(set, calibration, players);
        let possession = possession(&snaps);
        let sides = Sides::of(&snaps);
        Context {
            snaps,
            possession,
            sides,
        }
    }

    /// The windows found in the match.
    pub fn occurrences(&self) -> Vec<Occurrence> {
        find(&self.snaps, &self.possession, &self.sides)
    }

    /// One window measured and judged: found by `occurrences`, or a play the
    /// analyst tagged by hand. None when the team's end cannot be told.
    pub fn assess(
        &self,
        occurrence: &Occurrence,
        profile: &Profile,
        refs: &[Reference],
    ) -> Option<(Measures, Judgement)> {
        let goal = self.sides.goal(occurrence.team, occurrence.start_ms)?;
        let window: Vec<Snapshot> = self
            .snaps
            .iter()
            .filter(|s| occurrence.start_ms <= s.ms && s.ms <= occurrence.end_ms)
            .cloned()
            .collect();
        let measures = measure(occurrence.action, occurrence.team, &window, goal);
        // How much of the window could be seen, on average.
        let observability = if window.is_empty() {
            0.0
        } else {
            window
                .iter()
                .map(|s| shape(s, occurrence.team, goal, 10.0).observability)
                .sum::<f64>()
                / window.len() as f64
        };
        let relevant: Vec<Reference> = refs
            .iter()
            .filter(|r| r.action_key == occurrence.action.key())
            .cloned()
            .collect();
        let judgement = judge(
            occurrence.action,
            &measures,
            profile,
            &relevant,
            observability,
        );
        Some((measures, judgement))
    }
}

/// Evaluate all detected action windows in a track set.
pub fn evaluate(
    set: &TrackSet,
    calibration: &Calibration,
    profile: &Profile,
    refs: &[Reference],
) -> Vec<(Occurrence, Measures, Judgement)> {
    let context = Context::new(set, calibration);
    context
        .occurrences()
        .into_iter()
        .filter_map(|o| {
            let (m, j) = context.assess(&o, profile, refs)?;
            Some((o, m, j))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn snaps() -> Vec<Snapshot> {
        (0..8)
            .map(|i| {
                let bx = if i < 3 { 25.0 } else { 30.0 };
                let mut players = (0..10)
                    .map(|n| Placed {
                        id: n,
                        team: 1,
                        keeper: false,
                        role: None,
                        at: (
                            if n < 4 { bx + 0.5 } else { 15.0 + n as f64 },
                            20.0 + n as f64 * 2.0,
                        ),
                    })
                    .collect::<Vec<_>>();
                players.extend((10..20).map(|n| Placed {
                    id: n,
                    team: 2,
                    keeper: false,
                    role: None,
                    at: (70.0 + n as f64 * 0.2, 20.0 + n as f64),
                }));
                Snapshot {
                    ms: i * 1000,
                    players,
                    ball: Some((bx, 30.0)),
                }
            })
            .collect()
    }
    #[test]
    fn hysteresis_needs_two_snapshots() {
        let mut s = snaps();
        for x in &mut s {
            x.ball = Some((25.0, 20.0));
        }
        let p = possession(&s);
        assert_eq!(p[0].1, None);
        assert_eq!(p[1].1, Some(1));
    }
    #[test]
    fn a_loss_produces_both_windows() {
        let s = snaps();
        let p = vec![
            (0, Some(1)),
            (1000, Some(1)),
            (2000, Some(2)),
            (3000, Some(2)),
        ];
        let f = find(&s, &p, &Sides::of(&s));
        assert!(
            f.iter()
                .any(|o| o.action == Action::PressAfterLoss && o.team == 1)
        );
        assert!(
            f.iter()
                .any(|o| o.action == Action::RecoveryRun && o.team == 1)
        );
    }
    #[test]
    fn presets_round_trip() {
        let dir = std::env::temp_dir().join(format!("concat-eval-{}", std::process::id()));
        let p = Profile::presets();
        Profile::save_all(&dir, &p).expect("save");
        assert_eq!(Profile::load(&dir), p);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn references_flag_outlying_kpi() {
        let refs = (0..3)
            .map(|_| Reference {
                action_key: "press_after_loss".into(),
                good: true,
                measures: BTreeMap::from([("block_length".into(), Some(20.0))]),
            })
            .collect::<Vec<_>>();
        let fit = reference_fit(
            &BTreeMap::from([("block_length".into(), Some(60.0))]),
            &refs,
        )
        .expect("fit");
        assert!(fit.distance > 2.5);
    }
    #[test]
    fn a_good_press_passes_and_a_poor_press_is_bad() {
        let profile = Profile::presets().remove(0);
        let good = BTreeMap::from([
            ("pressers_3s".into(), Some(4.0)),
            ("nearest_2s".into(), Some(3.0)),
            ("regain_s".into(), Some(2.0)),
            ("block_length".into(), Some(30.0)),
        ]);
        assert_eq!(
            judge(Action::PressAfterLoss, &good, &profile, &[], 1.0).verdict,
            Verdict::Good
        );
        let poor = BTreeMap::from([
            ("pressers_3s".into(), Some(1.0)),
            ("nearest_2s".into(), Some(8.0)),
            ("regain_s".into(), Some(6.0)),
            ("block_length".into(), Some(48.0)),
        ]);
        let result = judge(Action::PressAfterLoss, &poor, &profile, &[], 1.0);
        assert_eq!(result.verdict, Verdict::Bad);
        assert!(result.reasons.iter().any(|r| r.contains("Bloque 48 m")));
    }
    #[test]
    fn sparse_team_is_unknown() {
        let mut profile = Profile::presets().remove(0);
        let s = snaps();
        let mut sparse = s[0].clone();
        sparse.players.retain(|p| p.team != 1 || p.id < 3);
        assert!(shape(&sparse, 1, Goal::Left, 10.0).observability < 0.5);
        profile.rules.insert(
            "press_after_loss".into(),
            vec![Rule {
                kpi: "pressers_3s".into(),
                min: Some(3.0),
                max: None,
                weight: 1.0,
                major: true,
            }],
        );
        let m = BTreeMap::from([("pressers_3s".into(), None)]);
        assert_eq!(
            judge(Action::PressAfterLoss, &m, &profile, &[], 0.3).verdict,
            Verdict::Unknown
        );
    }
    #[test]
    fn lateral_follow_is_positive() {
        let s: Vec<Snapshot> = (0..5)
            .map(|i| {
                let y = 10.0 + i as f64 * 5.0;
                Snapshot {
                    ms: i * 1000,
                    ball: Some((50.0, y)),
                    players: (0..10)
                        .map(|n| Placed {
                            id: n,
                            team: 1,
                            keeper: false,
                            role: None,
                            at: (40.0 + n as f64, y + n as f64 * 0.1),
                        })
                        .chain((10..20).map(|n| Placed {
                            id: n,
                            team: 2,
                            keeper: false,
                            role: None,
                            at: (70.0, y),
                        }))
                        .collect(),
                }
            })
            .collect();
        let m = measure(Action::DefensiveBlock, 1, &s, Goal::Left);
        assert!(m["shift_follow"].expect("correlation") > 0.9);
    }

    fn team_at(team: u8, xs: &[f64]) -> Vec<Placed> {
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

    #[test]
    fn the_back_line_is_the_players_nearest_the_own_goal() {
        let snap = Snapshot {
            ms: 0,
            players: team_at(
                1,
                &[20.0, 21.0, 22.0, 23.0, 40.0, 41.0, 42.0, 43.0, 60.0, 61.0],
            ),
            ball: None,
        };
        let shape = shape(&snap, 1, Goal::Left, 10.0);
        assert!((shape.def_line_height.unwrap() - 21.5).abs() < 1e-9);
        assert!((shape.midfield_line_height.unwrap() - 41.5).abs() < 1e-9);
        assert!((shape.line_gap.unwrap() - 20.0).abs() < 1e-9);
        // Defending the other end, the same players' back line is the far one.
        let other = super::shape(&snap, 1, Goal::Right, 10.0);
        assert!((other.def_line_height.unwrap() - 53.5).abs() < 1e-9);
    }

    #[test]
    fn the_ends_change_at_half_time() {
        let snap = |ms: u64, one: f64, two: f64| Snapshot {
            ms,
            players: [team_at(1, &[one; 10]), team_at(2, &[two; 10])].concat(),
            ball: None,
        };
        let mut snaps = Vec::new();
        for minute in 0..45u64 {
            snaps.push(snap(minute * 60_000, 30.0, 70.0));
        }
        for minute in 45..90u64 {
            snaps.push(snap(minute * 60_000, 70.0, 30.0));
        }
        let sides = Sides::of(&snaps);
        assert_eq!(sides.goal(1, 5 * 60_000), Some(Goal::Left));
        assert_eq!(sides.goal(2, 5 * 60_000), Some(Goal::Right));
        assert_eq!(sides.goal(1, 80 * 60_000), Some(Goal::Right));
        assert_eq!(sides.goal(2, 80 * 60_000), Some(Goal::Left));
        assert_eq!(sides.goal(0, 0), None);
    }

    #[test]
    fn standing_by_the_lost_ball_is_not_winning_it_back() {
        // Team 1 loses the ball at 0 and its player stays beside it, but a
        // team 2 player is nearer it the whole window: no regain.
        let snaps: Vec<Snapshot> = (0..13u64)
            .map(|i| {
                let mut players = team_at(1, &[30.0; 10]);
                players[0].at = (50.0, 30.0);
                let mut theirs = team_at(2, &[60.0; 10]);
                theirs[0].at = (50.5, 30.0);
                players.extend(theirs);
                Snapshot {
                    ms: i * 500,
                    players,
                    ball: Some((50.6, 30.0)),
                }
            })
            .collect();
        let m = measure(Action::PressAfterLoss, 1, &snaps, Goal::Left);
        assert_eq!(m["regained"], Some(0.0));
        assert_eq!(m["regain_s"], Some(6.0));
    }

    /// A real analysis judged end to end: `CONCAT_EVAL_DIR` names a folder
    /// with a `tracks.json` and a `calibration.json`.
    #[test]
    #[ignore]
    fn judges_a_real_analysis() {
        let Ok(dir) = std::env::var("CONCAT_EVAL_DIR") else {
            return;
        };
        let dir = Path::new(&dir);
        let set = TrackSet::load(dir).expect("tracks");
        let calibration = Calibration::load(dir);
        let context = Context::new(&set, &calibration);
        println!(
            "{} calibrated instants of {}",
            context.snaps.len(),
            set.samples.len()
        );
        for snap in context.snaps.iter().step_by(10) {
            let seen = |t| snap.players.iter().filter(|p| p.team == t).count();
            println!(
                "  {} ms: {} + {} players, ball {:?}",
                snap.ms,
                seen(1),
                seen(2),
                snap.ball.map(|(x, y)| (x.round(), y.round()))
            );
        }
        println!(
            "ends: 1 {:?}, 2 {:?}",
            context.sides.goal(1, 0),
            context.sides.goal(2, 0)
        );
        let changes = changes(&context.possession);
        println!("possession changes: {changes:?}");
        let profile = &Profile::presets()[0];
        for team in [1u8, 2] {
            let occurrence = Occurrence {
                action: Action::DefensiveBlock,
                team,
                start_ms: 0,
                end_ms: set.samples.last().map_or(0, |s| s.ms),
            };
            if let Some((measures, judgement)) = context.assess(&occurrence, profile, &[]) {
                println!("team {team} block: {measures:?}");
                println!(
                    "  {} (seen {:.0}%): {:?}",
                    judgement.verdict.label(),
                    judgement.observability * 100.0,
                    judgement.reasons
                );
            }
        }
        for (o, m, j) in evaluate(&set, &calibration, profile, &[]) {
            println!(
                "{} team {} {}-{} ms: {} {:?} {:?}",
                o.action.name(),
                o.team,
                o.start_ms,
                o.end_ms,
                j.verdict.label(),
                m,
                j.reasons
            );
        }
    }
}
