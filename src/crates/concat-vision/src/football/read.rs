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

use super::eval::{Context, Shape, Snapshot, depth, shape};

/// How long after the ball changes hands the play is still a transition.
const TRANSITION_MS: u64 = 6000;
/// Round the ball, for the numbers either side, in metres.
const NEAR_BALL_M: f64 = 15.0;
/// Within this of an instant a snapshot stands for it.
const REACH_MS: u64 = 700;
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
    let since = ctx.possession[..index]
        .iter()
        .rev()
        .find(|(_, other)| *other != holder && other.is_some())
        .map(|(t, _)| *t);
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
    Some(Moment {
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
        Some((third, lane)) => out.push_str(&format!(" Balón en nuestro {third}, por el {lane}.")),
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
    out.push_str(&format!(
        "\nVisibilidad del equipo: {:.0} %.",
        moment.ours.observability * 100.0
    ));
    out
}

/// The key moments of `[from_ms, to_ms]` from `team`'s side, in order.
pub fn events(ctx: &Context, team: u8, from_ms: u64, to_ms: u64) -> Vec<Event> {
    let mut out = Vec::new();
    let mut held: Option<Option<u8>> = None;
    let (mut last_third, mut in_box) = (None::<&'static str>, false);
    let start = ctx.snaps.partition_point(|s| s.ms < from_ms);
    for snap in ctx.snaps[start..].iter().take_while(|s| s.ms <= to_ms) {
        let (holder, _) = holder_at(ctx, snap.ms);
        if let Some(before) = held
            && before != holder
            && let Some(now) = holder
        {
            out.push(Event {
                ms: snap.ms,
                what: if now == team { "recuperación" } else { "pérdida" }.to_owned(),
            });
        }
        held = Some(holder);
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
        ctx.snaps[start..]
            .iter()
            .take_while(|s| s.ms <= to_ms)
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
    fn a_stretch_names_its_key_moments() {
        let ctx = context();
        let text = describe_span(&ctx, 1, 0, 7800);
        assert!(text.contains("Tramo"), "{text}");
        assert!(text.contains("Momentos clave"), "{text}");
    }
}
