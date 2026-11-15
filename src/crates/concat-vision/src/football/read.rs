// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Reading the game: what is happening at an instant and through a play,
//! in numbers the tracking and the calibration back, and in plain Spanish
//! for the assistant to reason from instead of guessing.
//!
//! An instant is read from the team's side: the phase (organised attack,
//! attacking transition, organised defence, defensive transition), where
//! the ball is (third and lane), each side's shape - length, width, the
//! back line's height, the gap between the lines - who is near the ball
//! and behind it, and the numbers round the ball. A play is its key
//! moments - the losses and regains, the entries into the last third and
//! into the box, for and against - and how the team stood with the ball
//! and without it.

use std::collections::{HashMap, VecDeque};

use super::eval::{Context, Placed, Shape, Snapshot, depth, shape};

/// How long after the ball changes hands the play is still a transition.
const TRANSITION_MS: u64 = 6000;
/// How long a ball nobody is on still counts as its last side's.
const LOOSE_MS: u64 = 3000;
/// Round the ball, for the numbers either side, in metres.
const NEAR_BALL_M: f64 = 15.0;
/// Within this of an instant a snapshot stands for it.
const REACH_MS: u64 = 700;
/// Round the ball, for whose ground it is, in metres.
const CONTROL_M: f64 = 25.0;
/// A player this near the ball is on it or pressing it, in metres.
const CARRY_M: f64 = 2.5;
const PRESS_M: f64 = 8.0;
/// A pass: the ball from one team-mate to another this far apart at
/// least, within this long.
const PASS_MIN_M: f64 = 5.0;
const PASS_MS: u64 = 4000;
/// A pass that gains this much ground is progressive, in metres.
const PROGRESSIVE_M: f64 = 10.0;
/// Running: high speed and sprint from these on, and nobody runs faster
/// than the last - past it the tracking slipped. In km/h, the women's
/// game's usual bands.
const HIGH_SPEED_KMH: f64 = 16.0;
const SPRINT_KMH: f64 = 20.0;
const MAX_KMH: f64 = 33.0;
/// The pitch, in metres.
const LENGTH: f64 = 105.0;
const WIDTH: f64 = 68.0;

/// The phase of play, from one team's side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// With the ball, settled.
    Attack,
    /// With the ball, just won.
    AttackingTransition,
    /// Without the ball, the rival settled on it.
    Defence,
    /// Without the ball, just lost.
    DefensiveTransition,
    /// Nobody has it, or it cannot be told.
    Unknown,
}

impl Phase {
    /// Its name, as a Spanish coach says it.
    pub fn label(self) -> &'static str {
        match self {
            Phase::Attack => "ataque organizado",
            Phase::AttackingTransition => "transición ofensiva",
            Phase::Defence => "defensa organizada",
            Phase::DefensiveTransition => "transición defensiva",
            Phase::Unknown => "balón dividido o sin datos",
        }
    }
}

/// One instant, read from `team`'s side.
#[derive(Clone, Debug)]
pub struct Moment {
    /// The instant read, in source milliseconds.
    pub ms: u64,
    /// The phase, from the team's side.
    pub phase: Phase,
    /// How long ago the ball last changed hands, in milliseconds.
    pub since_change_ms: Option<u64>,
    /// The ball's third and lane, from the team's side, when it is seen.
    pub zone: Option<(&'static str, &'static str)>,
    /// The team's shape.
    pub ours: Shape,
    /// The rival's shape.
    pub theirs: Shape,
    /// The team's players minus the rival's within [`NEAR_BALL_M`] of the
    /// ball.
    pub superiority: Option<i32>,
    /// The share of the ground within [`CONTROL_M`] of the ball the team
    /// reaches first, 0 to 1.
    pub control: Option<f64>,
    /// The team's nearest outfield player to the ball, in metres, and how
    /// many are within [`PRESS_M`] of it.
    pub pressure: Option<(f64, usize)>,
}

/// A pass between two of a team's players.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pass {
    /// When it was received, in source milliseconds.
    pub ms: u64,
    /// The ground it gained towards the rival goal, in metres.
    pub gain: f64,
    /// The rival outfield players it left behind.
    pub bypassed: usize,
}

/// How much a team ran over a stretch, as far as the camera saw it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Physical {
    /// Metres covered, all the seen players together.
    pub distance_m: f64,
    /// Of those, at high speed.
    pub high_speed_m: f64,
    /// Sprints begun.
    pub sprints: usize,
    /// The fastest anybody went, in km/h.
    pub top_kmh: f64,
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// The share of the ground round the ball `team` reaches first: each
/// point of a two metre grid is its nearest outfield player's.
pub fn control(snap: &Snapshot, team: u8) -> Option<f64> {
    let ball = snap.ball?;
    let outfield: Vec<&Placed> = snap
        .players
        .iter()
        .filter(|p| !p.keeper && (p.team == 1 || p.team == 2))
        .collect();
    if outfield.len() < 4 {
        return None;
    }
    let steps = (CONTROL_M / 2.0) as i32;
    let (mut ours, mut all) = (0usize, 0usize);
    for i in -steps..=steps {
        for j in -steps..=steps {
            let at = (ball.0 + f64::from(i) * 2.0, ball.1 + f64::from(j) * 2.0);
            if at.0 < 0.0 || at.0 > LENGTH || at.1 < 0.0 || at.1 > WIDTH || dist(at, ball) > CONTROL_M {
                continue;
            }
            let Some(nearest) = outfield
                .iter()
                .min_by(|a, b| dist(a.at, at).total_cmp(&dist(b.at, at)))
            else {
                continue;
            };
            all += 1;
            if nearest.team == team {
                ours += 1;
            }
        }
    }
    (all > 0).then(|| ours as f64 / all as f64)
}

/// The player of `team` on the ball, if one is.
fn carrier(snap: &Snapshot, team: u8) -> Option<&Placed> {
    let ball = snap.ball?;
    snap.players
        .iter()
        .filter(|p| p.team == team && dist(p.at, ball) <= CARRY_M)
        .min_by(|a, b| dist(a.at, ball).total_cmp(&dist(b.at, ball)))
}

/// The passes `team` put together in `[from_ms, to_ms]`: the ball on one
/// player and then on another, the rival not on it in between.
pub fn passes(ctx: &Context, team: u8, from_ms: u64, to_ms: u64) -> Vec<Pass> {
    let rival = 3 - team;
    let mut out = Vec::new();
    let mut last: Option<(u64, u32, (f64, f64))> = None;
    let start = ctx.snaps.partition_point(|s| s.ms < from_ms);
    for snap in ctx.snaps[start..].iter().take_while(|s| s.ms <= to_ms) {
        let Some(now) = carrier(snap, team) else {
            if carrier(snap, rival).is_some() {
                last = None;
            }
            continue;
        };
        if let Some((ms, id, at)) = last
            && id != now.id
            && snap.ms.saturating_sub(ms) <= PASS_MS
            && dist(at, now.at) >= PASS_MIN_M
            && let Some(goal) = ctx.sides.goal(team, snap.ms)
        {
            let (a, b) = (depth(goal, at.0), depth(goal, now.at.0));
            let bypassed = snap
                .players
                .iter()
                .filter(|p| p.team == rival && !p.keeper)
                .filter(|p| {
                    let d = depth(goal, p.at.0);
                    d > a && d < b
                })
                .count();
            out.push(Pass { ms: snap.ms, gain: b - a, bypassed });
        }
        last = Some((snap.ms, now.id, now.at));
    }
    out
}

/// How much `team` ran in `[from_ms, to_ms]`. Speed is taken over about a
/// second, so the tracking's jitter does not read as running.
pub fn physical(ctx: &Context, team: u8, from_ms: u64, to_ms: u64) -> Physical {
    let mut out = Physical::default();
    let mut seen: HashMap<u32, (VecDeque<(u64, (f64, f64))>, bool)> = HashMap::new();
    let start = ctx.snaps.partition_point(|s| s.ms < from_ms);
    for snap in ctx.snaps[start..].iter().take_while(|s| s.ms <= to_ms) {
        for player in snap.players.iter().filter(|p| p.team == team && !p.keeper) {
            let (trail, sprinting) = seen.entry(player.id).or_default();
            // Out of sight for a while: a new run of instants.
            if trail.back().is_some_and(|(ms, _)| snap.ms.saturating_sub(*ms) > 600) {
                trail.clear();
                *sprinting = false;
            }
            let step_ms = trail.back().map(|(ms, _)| snap.ms.saturating_sub(*ms));
            trail.push_back((snap.ms, player.at));
            while trail.front().is_some_and(|(ms, _)| snap.ms.saturating_sub(*ms) > 1200) {
                trail.pop_front();
            }
            let (Some(step_ms), Some(&(then, from))) = (step_ms, trail.front()) else {
                continue;
            };
            let over = snap.ms.saturating_sub(then);
            if over < 600 {
                continue;
            }
            let kmh = dist(from, player.at) / (over as f64 / 1000.0) * 3.6;
            if kmh > MAX_KMH {
                continue;
            }
            let covered = kmh / 3.6 * (step_ms as f64 / 1000.0);
            out.distance_m += covered;
            out.top_kmh = out.top_kmh.max(kmh);
            if kmh >= HIGH_SPEED_KMH {
                out.high_speed_m += covered;
            }
            let sprint = kmh >= SPRINT_KMH;
            if sprint && !*sprinting {
                out.sprints += 1;
            }
            *sprinting = sprint;
        }
    }
    out
}

/// A key moment of a play.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// When, in source milliseconds.
    pub ms: u64,
    /// What happened, in Spanish.
    pub what: String,
}

fn snap_at(ctx: &Context, ms: u64) -> Option<&Snapshot> {
    let index = ctx.snaps.partition_point(|s| s.ms < ms);
    [index.checked_sub(1), Some(index)]
        .into_iter()
        .flatten()
        .filter_map(|i| ctx.snaps.get(i))
        .min_by_key(|s| s.ms.abs_diff(ms))
        .filter(|s| s.ms.abs_diff(ms) <= REACH_MS)
}

/// Who had the ball at `ms`, and since when the holder had held it.
fn holder_at(ctx: &Context, ms: u64) -> (Option<u8>, Option<u64>) {
    let index = ctx.possession.partition_point(|(t, _)| *t <= ms);
    let Some(&(_, holder)) = index.checked_sub(1).and_then(|i| ctx.possession.get(i)) else {
        return (None, None);
    };
    // A ball in the air or hidden for a moment is still its last side's:
    // whoever had it within [`LOOSE_MS`] has it.
    let holder = holder.or_else(|| {
        ctx.possession[..index]
            .iter()
            .rev()
            .take_while(|(t, _)| ms.saturating_sub(*t) <= LOOSE_MS)
            .find_map(|(_, who)| *who)
    });
    // The holder's first instant after the other side last had it: when
    // the ball changed hands, not when the other side was last seen on it.
    let since = holder.and_then(|_| {
        let past = &ctx.possession[..index];
        let other = past
            .iter()
            .rposition(|(_, other)| *other != holder && other.is_some())?;
        past[other + 1..]
            .iter()
            .find(|(_, who)| *who == holder)
            .map(|(t, _)| *t)
    });
    (holder, since)
}

/// The ball's third and lane from the side of a team defending `goal`'s end.
fn zone(goal: super::eval::Goal, ball: (f64, f64)) -> (&'static str, &'static str) {
    let d = depth(goal, ball.0);
    let third = if d < LENGTH / 3.0 {
        "tercio defensivo"
    } else if d < LENGTH * 2.0 / 3.0 {
        "tercio medio"
    } else {
        "último tercio"
    };
    // Lanes as the camera sees the pitch: y = 0 is the far touchline.
    let lane = match (ball.1 / (WIDTH / 5.0)).floor() as i32 {
        i32::MIN..=0 => "banda lejana",
        1 => "carril interior lejano",
        2 => "carril central",
        3 => "carril interior cercano",
        _ => "banda cercana",
    };
    (third, lane)
}

/// The instant `ms` read from `team`'s side, or None when no calibrated
/// instant is near or the ends cannot be told.
pub fn moment(ctx: &Context, team: u8, ms: u64) -> Option<Moment> {
    let snap = snap_at(ctx, ms)?;
    let rival = 3 - team;
    let goal = ctx.sides.goal(team, snap.ms)?;
    let rival_goal = ctx.sides.goal(rival, snap.ms)?;
    let (holder, since) = holder_at(ctx, snap.ms);
    let since_change_ms = since.map(|t| snap.ms.saturating_sub(t));
    let fresh = since_change_ms.is_some_and(|d| d <= TRANSITION_MS);
    let phase = match holder {
        Some(h) if h == team && fresh => Phase::AttackingTransition,
        Some(h) if h == team => Phase::Attack,
        Some(_) if fresh => Phase::DefensiveTransition,
        Some(_) => Phase::Defence,
        None => Phase::Unknown,
    };
    let superiority = snap.ball.map(|b| {
        let near = |t: u8| {
            snap.players
                .iter()
                .filter(|p| p.team == t && !p.keeper)
                .filter(|p| (p.at.0 - b.0).hypot(p.at.1 - b.1) <= NEAR_BALL_M)
                .count() as i32
        };
        near(team) - near(rival)
    });
    let pressure = snap.ball.and_then(|b| {
        let mut gaps: Vec<f64> = snap
            .players
            .iter()
            .filter(|p| p.team == team && !p.keeper)
            .map(|p| dist(p.at, b))
            .collect();
        gaps.sort_by(f64::total_cmp);
        let nearest = *gaps.first()?;
        Some((nearest, gaps.iter().filter(|d| **d <= PRESS_M).count()))
    });
    Some(Moment {
        control: control(snap, team),
        pressure,
        ms: snap.ms,
        phase,
        since_change_ms,
        zone: snap.ball.map(|b| zone(goal, b)),
        ours: shape(snap, team, goal, 10.0),
        theirs: shape(snap, rival, rival_goal, 10.0),
        superiority,
    })
}

/// `ms` as a match clock, mm:ss.
pub fn clock(ms: u64) -> String {
    format!("{:02}:{:02}", ms / 60_000, (ms / 1000) % 60)
}

fn metres(value: Option<f64>) -> String {
    value.map_or_else(|| "?".to_owned(), |v| format!("{v:.0} m"))
}

fn side(shape: &Shape) -> String {
    format!(
        "{} jugadoras vistas; bloque de {} de largo y {} de ancho; línea defensiva a {} de su portería; {} entre líneas; {} a menos de 10 m del balón y {} por detrás del balón",
        shape.count,
        metres(shape.block_length),
        metres(shape.width),
        metres(shape.def_line_height),
        metres(shape.line_gap),
        shape.near_ball.map_or_else(|| "?".to_owned(), |n| n.to_string()),
        shape.behind_ball.map_or_else(|| "?".to_owned(), |n| n.to_string()),
    )
}

/// An instant described for the assistant.
pub fn describe(moment: &Moment) -> String {
    let mut out = format!("Minuto {} · fase: {}", clock(moment.ms), moment.phase.label());
    if let Some(since) = moment.since_change_ms.filter(|d| *d <= TRANSITION_MS * 2) {
        out.push_str(&format!(" (el balón cambió de equipo hace {:.1} s)", since as f64 / 1000.0));
    }
    out.push('.');
    match moment.zone {
        Some((third, lane)) => {
            let place = match third {
                "tercio defensivo" => "nuestro tercio defensivo (cerca de nuestra portería)",
                "tercio medio" => "el tercio medio",
                _ => "el último tercio (campo rival, zona de ataque)",
            };
            let lane = if lane.starts_with("banda") { format!("la {lane}") } else { format!("el {lane}") };
            out.push_str(&format!(" Balón en {place}, por {lane}."));
        }
        None => out.push_str(" Balón no visible."),
    }
    out.push_str(&format!("\nNosotras: {}.", side(&moment.ours)));
    out.push_str(&format!("\nRival: {}.", side(&moment.theirs)));
    if let Some(s) = moment.superiority {
        let read = match s {
            s if s > 0 => format!("superioridad nuestra de {s}"),
            s if s < 0 => format!("inferioridad de {}", -s),
            _ => "igualdad".to_owned(),
        };
        out.push_str(&format!("\nCerca del balón (15 m): {read}."));
    }
    if let Some(share) = moment.control {
        out.push_str(&format!(
            "\nControl del espacio a 25 m del balón: nosotras {:.0} %, rival {:.0} %.",
            share * 100.0,
            (1.0 - share) * 100.0
        ));
    }
    if matches!(moment.phase, Phase::Defence | Phase::DefensiveTransition)
        && let Some((nearest, close)) = moment.pressure
    {
        out.push_str(&format!(
            "\nPresión sobre el balón: la más cercana a {nearest:.0} m, {close} a menos de 8 m."
        ));
    }
    out.push_str(&format!(
        "\nVisibilidad del equipo: {:.0} %.",
        moment.ours.observability * 100.0
    ));
    out
}

/// The key moments of `[from_ms, to_ms]` from `team`'s side, in order.
pub fn events(ctx: &Context, team: u8, from_ms: u64, to_ms: u64) -> Vec<Event> {
    let mut out = Vec::new();
    // The last side known to have it: a spell nobody has it is no change.
    let mut held: Option<u8> = None;
    let (mut last_third, mut in_box) = (None::<&'static str>, false);
    let start = ctx.snaps.partition_point(|s| s.ms < from_ms);
    for snap in ctx.snaps[start..].iter().take_while(|s| s.ms <= to_ms) {
        let (holder, _) = holder_at(ctx, snap.ms);
        if let Some(before) = held
            && let Some(now) = holder
            && before != now
        {
            out.push(Event {
                ms: snap.ms,
                what: if now == team { "recuperación" } else { "pérdida" }.to_owned(),
            });
        }
        held = holder.or(held);
        let (Some(ball), Some(goal)) = (snap.ball, ctx.sides.goal(team, snap.ms)) else {
            continue;
        };
        let (third, _) = zone(goal, ball);
        let d = depth(goal, ball.0);
        let central = (ball.1 - WIDTH / 2.0).abs() <= 20.16;
        let box_now = central && (d >= LENGTH - 16.5 || d <= 16.5);
        if last_third.is_some() && last_third != Some(third) {
            if third == "último tercio" {
                out.push(Event { ms: snap.ms, what: "el balón entra en el último tercio rival".into() });
            } else if third == "tercio defensivo" {
                out.push(Event { ms: snap.ms, what: "el balón entra en nuestro tercio defensivo".into() });
            }
        }
        if box_now && !in_box {
            out.push(Event {
                ms: snap.ms,
                what: if d >= LENGTH - 16.5 {
                    "balón en el área rival"
                } else {
                    "balón en nuestra área"
                }
                .into(),
            });
        }
        last_third = Some(third);
        in_box = box_now;
    }
    out
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

/// A stretch described for the assistant: its key moments and how the
/// team stood with the ball and without it.
pub fn describe_span(ctx: &Context, team: u8, from_ms: u64, to_ms: u64) -> String {
    let moments: Vec<Moment> = {
        let start = ctx.snaps.partition_point(|s| s.ms < from_ms);
        // A long stretch is read once a second: enough for its averages.
        let stride = if to_ms.saturating_sub(from_ms) > 120_000 { 5 } else { 1 };
        ctx.snaps[start..]
            .iter()
            .take_while(|s| s.ms <= to_ms)
            .step_by(stride)
            .filter_map(|s| moment(ctx, team, s.ms))
            .collect()
    };
    if moments.is_empty() {
        return "Sin datos en ese tramo: hace falta el vídeo analizado y el campo calibrado.".into();
    }
    let mut out = format!(
        "Tramo {}–{} ({:.0} s).",
        clock(from_ms),
        clock(to_ms),
        to_ms.saturating_sub(from_ms) as f64 / 1000.0
    );
    let with = moments
        .iter()
        .filter(|m| matches!(m.phase, Phase::Attack | Phase::AttackingTransition))
        .count();
    let without = moments
        .iter()
        .filter(|m| matches!(m.phase, Phase::Defence | Phase::DefensiveTransition))
        .count();
    let total = (with + without).max(1) as f64;
    out.push_str(&format!(
        " Posesión: nuestra {:.0} %, rival {:.0} %.",
        with as f64 * 100.0 / total,
        without as f64 * 100.0 / total
    ));
    let stood = |pick: &dyn Fn(&Moment) -> bool, label: &str| {
        let chosen: Vec<&Moment> = moments.iter().filter(|m| pick(m)).collect();
        if chosen.is_empty() {
            return String::new();
        }
        format!(
            "\n{label}: bloque {} de largo y {} de ancho, línea defensiva a {}, {} entre líneas, de media {:.1} cerca del balón y {:.1} por detrás; superioridad media junto al balón {:+.1}.",
            metres(mean(chosen.iter().filter_map(|m| m.ours.block_length))),
            metres(mean(chosen.iter().filter_map(|m| m.ours.width))),
            metres(mean(chosen.iter().filter_map(|m| m.ours.def_line_height))),
            metres(mean(chosen.iter().filter_map(|m| m.ours.line_gap))),
            mean(chosen.iter().filter_map(|m| m.ours.near_ball.map(|n| n as f64))).unwrap_or(0.0),
            mean(chosen.iter().filter_map(|m| m.ours.behind_ball.map(|n| n as f64))).unwrap_or(0.0),
            mean(chosen.iter().filter_map(|m| m.superiority.map(f64::from))).unwrap_or(0.0),
        )
    };
    out.push_str(&stood(
        &|m| matches!(m.phase, Phase::Attack | Phase::AttackingTransition),
        "Con balón",
    ));
    out.push_str(&stood(
        &|m| matches!(m.phase, Phase::Defence | Phase::DefensiveTransition),
        "Sin balón",
    ));
    let with_ball = |m: &&Moment| matches!(m.phase, Phase::Attack | Phase::AttackingTransition);
    let without_ball = |m: &&Moment| matches!(m.phase, Phase::Defence | Phase::DefensiveTransition);
    if let Some(share) = mean(moments.iter().filter(with_ball).filter_map(|m| m.control)) {
        out.push_str(&format!(
            "\nControl del espacio junto al balón cuando atacamos: {:.0} %.",
            share * 100.0
        ));
    }
    if let Some(share) = mean(moments.iter().filter(without_ball).filter_map(|m| m.control)) {
        out.push_str(&format!(
            "\nControl del espacio junto al balón cuando defendemos: {:.0} %.",
            share * 100.0
        ));
    }
    if let (Some(nearest), Some(close)) = (
        mean(moments.iter().filter(without_ball).filter_map(|m| m.pressure.map(|p| p.0))),
        mean(moments.iter().filter(without_ball).filter_map(|m| m.pressure.map(|p| p.1 as f64))),
    ) {
        out.push_str(&format!(
            "\nPresión sin balón: la más cercana a {nearest:.1} m del balón de media, {close:.1} jugadoras a menos de 8 m."
        ));
    }
    let made = passes(ctx, team, from_ms, to_ms);
    if !made.is_empty() {
        let progressive = made.iter().filter(|p| p.gain >= PROGRESSIVE_M).count();
        let bypassed: usize = made.iter().map(|p| p.bypassed).sum();
        out.push_str(&format!(
            "\nPases nuestros detectados: {} ({progressive} progresivos de +10 m); rivales superadas con pase: {bypassed}.",
            made.len()
        ));
        if let Some(best) = made.iter().max_by_key(|p| p.bypassed).filter(|p| p.bypassed >= 2) {
            out.push_str(&format!(
                " El mejor, en {}, supera a {} y gana {:.0} m.",
                clock(best.ms),
                best.bypassed,
                best.gain
            ));
        }
    }
    let ran = physical(ctx, team, from_ms, to_ms);
    if ran.distance_m > 0.0 {
        out.push_str(&format!(
            "\nFísico (solo lo que ve la cámara): {:.0} m recorridos entre todas, {:.0} m a alta velocidad (+16 km/h), {} sprints (+20 km/h), punta de {:.1} km/h.",
            ran.distance_m, ran.high_speed_m, ran.sprints, ran.top_kmh
        ));
    }
    let key = events(ctx, team, from_ms, to_ms);
    if key.is_empty() {
        out.push_str("\nMomentos clave: ninguno (el balón no cambió de equipo ni de zona).");
    } else {
        out.push_str("\nMomentos clave:");
        for event in key.iter().take(20) {
            out.push_str(&format!("\n- {} {}", clock(event.ms), event.what));
        }
    }
    out
}

/// The reading's cards for a report, over everything analysed: the
/// number, what it counts, and how it reads - as the model's KPI cards.
pub fn cards(ctx: &Context, team: u8) -> Vec<(String, String, String)> {
    let Some(end) = ctx.snaps.last().map(|s| s.ms) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    // Once a second is enough for an average over a match.
    if let Some(share) = mean(ctx.snaps.iter().step_by(5).filter_map(|s| control(s, team))) {
        out.push((
            format!("{:.0} %", share * 100.0),
            "Control del espacio".to_owned(),
            "terreno junto al balón al que llegamos antes".to_owned(),
        ));
    }
    let made = passes(ctx, team, 0, end);
    if !made.is_empty() {
        out.push((
            made.iter().filter(|p| p.gain >= PROGRESSIVE_M).count().to_string(),
            "Pases progresivos".to_owned(),
            format!("de +10 m · de {} pases detectados", made.len()),
        ));
        out.push((
            made.iter().map(|p| p.bypassed).sum::<usize>().to_string(),
            "Rivales superadas con pase".to_owned(),
            format!("{} pases rompen líneas (3 o más)", made.iter().filter(|p| p.bypassed >= 3).count()),
        ));
    }
    let ran = physical(ctx, team, 0, end);
    if ran.distance_m > 0.0 {
        out.push((
            ran.sprints.to_string(),
            "Sprints".to_owned(),
            format!(
                "+20 km/h · {:.0} m a alta velocidad · punta {:.1} km/h (lo que ve la cámara)",
                ran.high_speed_m, ran.top_kmh
            ),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::football::eval::{Placed, Sides, possession};

    /// Team 1 defends the left end, team 2 the right; the ball with team 1
    /// for the first two seconds, then team 2's.
    fn context() -> Context {
        let mut snaps = Vec::new();
        for i in 0..40u64 {
            let ms = i * 200;
            let mut players = Vec::new();
            for k in 0..10u32 {
                players.push(Placed { id: k, team: 1, keeper: false, at: (20.0 + k as f64 * 3.0, 10.0 + k as f64 * 5.0), role: None });
                players.push(Placed { id: 100 + k, team: 2, keeper: false, at: (60.0 + k as f64 * 3.0, 10.0 + k as f64 * 5.0), role: None });
            }
            // The ball at the feet of a team 1 player, then of a team 2 one.
            let ball = if ms < 2000 { (20.5, 10.0) } else { (60.5, 10.0) };
            snaps.push(Snapshot { ms, players, ball: Some(ball) });
        }
        let possession = possession(&snaps);
        let sides = Sides::of(&snaps);
        Context { snaps, possession, sides }
    }

    #[test]
    fn an_instant_reads_its_phase_and_zone() {
        let ctx = context();
        let early = moment(&ctx, 1, 1000).expect("an instant");
        assert!(matches!(early.phase, Phase::Attack | Phase::AttackingTransition | Phase::Unknown));
        assert_eq!(early.zone.map(|z| z.0), Some("tercio defensivo"));
        let text = describe(&early);
        assert!(text.contains("Nosotras"), "{text}");
        let late = moment(&ctx, 1, 7000).expect("an instant");
        assert!(matches!(late.phase, Phase::Defence | Phase::DefensiveTransition | Phase::Unknown));
    }

    #[test]
    fn the_ground_round_the_ball_is_shared_and_running_is_measured() {
        let ctx = context();
        let share = control(&ctx.snaps[0], 1).expect("a share");
        assert!((0.0..=1.0).contains(&share));
        // Nobody moves in this match: no metres, no sprints, no passes.
        let ran = physical(&ctx, 1, 0, 7800);
        assert_eq!(ran.sprints, 0);
        assert!(ran.distance_m < 1.0, "{ran:?}");
        assert!(passes(&ctx, 1, 0, 7800).is_empty());
    }

    #[test]
    fn a_ball_moved_between_team_mates_is_a_pass() {
        let mut ctx = context();
        // The ball from team 1's deepest player to one fifteen metres on.
        for snap in ctx.snaps.iter_mut().filter(|s| s.ms >= 1000) {
            snap.ball = Some((35.0, 35.0));
        }
        let made = passes(&ctx, 1, 0, 1800);
        assert_eq!(made.len(), 1, "{made:?}");
        assert!(made[0].gain > 10.0, "{made:?}");
    }

    #[test]
    fn a_stretch_names_its_key_moments() {
        let ctx = context();
        let text = describe_span(&ctx, 1, 0, 7800);
        assert!(text.contains("Tramo"), "{text}");
        assert!(text.contains("Momentos clave"), "{text}");
    }
}
