// SPDX-License-Identifier: AGPL-3.0-or-later
//! Informe pospartido para el cuerpo técnico.
use crate::football::eval::{self, Context, Goal, Measures, Profile, Verdict};
use std::collections::HashMap;
/// Acción evaluada en una jugada.
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    /// Identificador KPI de acción.
    pub action_key: String,
    /// Nombre visible de acción.
    pub action_name: String,
    /// Identificador de jugada.
    pub play_id: String,
    /// Nombre de jugada.
    pub play_name: String,
    /// Inicio en segundos de línea temporal.
    pub start: f64,
    /// Valoración.
    pub verdict: Verdict,
    /// Medidas registradas.
    pub measures: Measures,
}
/// Resumen de una fase.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhaseReport {
    /// Nombre de fase.
    pub phase: String,
    /// Clave de acción.
    pub action_key: String,
    /// Conteo bueno.
    pub good: usize,
    /// Conteo mejorable.
    pub improvable: usize,
    /// Conteo malo.
    pub bad: usize,
    /// Conteo sin datos.
    pub unknown: usize,
    /// Hallazgos positivos.
    pub worked: Vec<String>,
    /// Hallazgos a mejorar.
    pub failed: Vec<String>,
    /// Explicaciones causales.
    pub why: Vec<String>,
    /// Jugadas destacadas.
    pub best: Vec<String>,
    /// Jugadas más débiles.
    pub worst: Vec<String>,
}
/// Construye un resumen por acción según las reglas del perfil.
pub fn build(items: &[Item], profile: &Profile) -> Vec<PhaseReport> {
    let mut keys: Vec<String> = items.iter().map(|i| i.action_key.clone()).collect();
    keys.sort();
    keys.dedup();
    let order = [
        "build_up",
        "press_after_loss",
        "recovery_run",
        "defensive_block",
    ];
    keys.sort_by(|a, b| {
        let ia = order.iter().position(|x| x == a).unwrap_or(4);
        let ib = order.iter().position(|x| x == b).unwrap_or(4);
        ia.cmp(&ib).then_with(|| {
            if ia == 4 {
                items
                    .iter()
                    .find(|i| &i.action_key == a)
                    .map(|i| i.action_name.as_str())
                    .unwrap_or(a)
                    .cmp(
                        items
                            .iter()
                            .find(|i| &i.action_key == b)
                            .map(|i| i.action_name.as_str())
                            .unwrap_or(b),
                    )
            } else {
                std::cmp::Ordering::Equal
            }
        })
    });
    keys.into_iter()
        .map(|key| {
            let rows: Vec<&Item> = items.iter().filter(|i| i.action_key == key).collect();
            let name = rows
                .first()
                .map_or("Acciones".to_string(), |i| i.action_name.clone());
            let phase = match key.as_str() {
                "build_up" => "Salida de balón",
                "press_after_loss" => "Presión tras pérdida",
                "recovery_run" => "Transición defensiva",
                "defensive_block" => "Bloque defensivo (caja)",
                _ => &name,
            }
            .to_string();
            let mut r = PhaseReport {
                phase,
                action_key: key.clone(),
                ..Default::default()
            };
            for i in &rows {
                match i.verdict {
                    Verdict::Good => r.good += 1,
                    Verdict::Improvable => r.improvable += 1,
                    Verdict::Bad => r.bad += 1,
                    Verdict::Unknown => r.unknown += 1,
                }
            }
            let rules = profile.rules.get(&key).cloned().unwrap_or_default();
            let noun = match key.as_str() {
                "build_up" => "salidas",
                "press_after_loss" => "presiones",
                "recovery_run" => "transiciones",
                "defensive_block" => "acciones de bloque",
                _ => "acciones",
            };
            let mut fails = Vec::new();
            for rule in rules {
                let vals: Vec<f64> = rows
                    .iter()
                    .filter_map(|i| i.measures.get(&rule.kpi).copied().flatten())
                    .collect();
                if vals.is_empty() {
                    continue;
                }
                let pass = vals
                    .iter()
                    .filter(|v| {
                        rule.min.is_none_or(|x| **v >= x) && rule.max.is_none_or(|x| **v <= x)
                    })
                    .count();
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let bound = rule
                    .min
                    .map(|x| ("≥", x))
                    .or_else(|| rule.max.map(|x| ("≤", x)));
                let Some((op, b)) = bound else { continue };
                let text = format!(
                    "En {} de {} {}, {} {} {}{} (media {}{})",
                    pass,
                    vals.len(),
                    noun,
                    eval::kpi_label(&rule.kpi),
                    op,
                    eval::fmt(b),
                    eval::unit_suffix(&rule.kpi),
                    eval::fmt(mean),
                    eval::unit_suffix(&rule.kpi)
                );
                if vals.len() >= 2 && pass as f64 / vals.len() as f64 >= 0.7 {
                    r.worked.push(text)
                }
                let failed = vals.len() - pass;
                if failed as f64 / vals.len() as f64 >= 0.4 {
                    let side = if op == "≥" {
                        "por debajo de"
                    } else {
                        "por encima de"
                    };
                    let ft = format!(
                        "En {} de {} {}, {} {} {}{} (media {}{})",
                        failed,
                        vals.len(),
                        noun,
                        eval::kpi_label(&rule.kpi),
                        side,
                        eval::fmt(b),
                        eval::unit_suffix(&rule.kpi),
                        eval::fmt(mean),
                        eval::unit_suffix(&rule.kpi)
                    );
                    fails.push((failed, rule.kpi.clone(), rule.min.is_some(), ft));
                }
            }
            fails.sort_by(|a, b| b.0.cmp(&a.0));
            r.failed = fails.iter().map(|x| x.3.clone()).collect();
            for (_, k, min, _) in fails.iter().take(3) {
                let msg = match (k.as_str(), *min) {
                    ("block_length", false) => {
                        "El bloque se estira: las líneas no acompañan el salto de la presión"
                    }
                    ("pressers_3s", true) => {
                        "Saltan pocas jugadoras: la presión llega tarde o en solitario"
                    }
                    ("nearest_2s", false) => "La primera jugadora llega tarde al poseedor",
                    ("regain_s", false) => {
                        "No se recupera rápido: la presión no cierra líneas de pase"
                    }
                    ("width", true) if key == "build_up" => {
                        "Poca amplitud: las laterales no dan la máxima anchura"
                    }
                    ("superiority_zone", true) => {
                        "Sin superioridad cerca del balón: no aparecen las jugadoras libres"
                    }
                    ("support_15m", true) => "Pocos apoyos cercanos a la poseedora",
                    ("progress_m", true) => "La salida no progresa: se juega en horizontal",
                    ("width", false) if key == "defensive_block" => {
                        "Salen de la caja más jugadoras de las que permite el modelo"
                    }
                    ("outside_box", false) => {
                        "Salen de la caja más jugadoras de las que permite el modelo"
                    }
                    ("line_gap", false) => "Demasiado espacio entre líneas",
                    ("def_line_height", true) => "La línea defensiva se hunde",
                    ("shift_follow", true) => "El bloque no bascula con el balón",
                    ("behind_ball_5s", true) => "Pocas jugadoras por detrás del balón tras perder",
                    ("reorganise_s", false) => "El equipo tarda en reorganizarse",
                    ("interiors_press_wide", true) => {
                        "No son las interiores quienes salen a presionar fuera"
                    }
                    ("pivot_central", true) => "El pivote no equilibra por dentro",
                    _ => "",
                };
                r.why.push(if msg.is_empty() {
                    format!("{} fuera del objetivo", eval::kpi_label(k))
                } else {
                    msg.to_string()
                })
            }
            let mut ranked = rows.clone();
            ranked.sort_by(|a, b| {
                let rank = |v: Verdict| match v {
                    Verdict::Good => 0,
                    Verdict::Improvable => 1,
                    Verdict::Bad => 2,
                    Verdict::Unknown => 3,
                };
                rank(a.verdict)
                    .cmp(&rank(b.verdict))
                    .then(a.start.total_cmp(&b.start))
            });
            r.best = ranked.iter().take(2).map(|i| i.play_id.clone()).collect();
            ranked.sort_by(|a, b| {
                let rank = |v: Verdict| match v {
                    Verdict::Bad => 0,
                    Verdict::Improvable => 1,
                    Verdict::Good => 2,
                    Verdict::Unknown => 3,
                };
                rank(a.verdict).cmp(&rank(b.verdict))
            });
            r.worst = ranked.iter().take(2).map(|i| i.play_id.clone()).collect();
            r
        })
        .collect()
}
/// Renderiza un informe HTML autocontenido y apto para impresión.
pub fn to_html(
    title: &str,
    subtitle: &str,
    phases: &[PhaseReport],
    thumbs: &HashMap<String, String>,
    names: &HashMap<String, String>,
) -> String {
    to_html_with(title, subtitle, &[], phases, thumbs, names)
}

/// [`to_html`] with the model's KPI cards under the title.
pub fn to_html_with(
    title: &str,
    subtitle: &str,
    cards: &[(String, String, String)],
    phases: &[PhaseReport],
    thumbs: &HashMap<String, String>,
    names: &HashMap<String, String>,
) -> String {
    fn esc(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }
    let mut h = format!(
        "<!doctype html><html lang=\"es\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>{}</title><style>body{{margin:0;background:#0f1e33;color:white;font:16px Arial,sans-serif}}main{{max-width:1000px;margin:auto;padding:40px}}section{{min-height:90vh;padding:30px 0;page-break-after:always}}.pill{{background:#24528a;border-radius:20px;padding:8px 16px}}.chips span{{padding:8px;border-radius:8px;margin-right:8px}}.good{{background:#176b50}}.mid{{background:#946d13}}.bad{{background:#8a3540}}.blocks{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px}}article{{background:#172b47;padding:18px;border-radius:12px}}img{{max-width:240px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:28px 0 8px}}.card{{background:#172b47;border-radius:12px;padding:20px}}.card b{{display:block;font-size:52px;line-height:1}}.card strong{{display:block;margin-top:14px;font-size:18px}}.card small{{display:block;margin-top:6px;color:#9fb0c8}}figure{{display:inline-block;margin:18px 18px 0 0}}figcaption{{color:#9fb0c8;font-size:13px}}@media print{{body{{background:white;color:#111}}section{{break-after:page}}}}</style><main><h1>{}</h1><p>{}</p>",
        esc(title),
        esc(title),
        esc(subtitle)
    );
    if !cards.is_empty() {
        h.push_str("<div class=\"cards\">");
        for (number, label, note) in cards {
            h.push_str(&format!(
                "<div class=\"card\"><b>{}</b><strong>{}</strong><small>{}</small></div>",
                esc(number),
                esc(label),
                esc(note)
            ));
        }
        h.push_str("</div>");
    }
    for (i, p) in phases.iter().enumerate() {
        h.push_str(&format!("<section><span class=\"pill\">{:02} · {}</span><h1>{}</h1><div class=\"chips\"><span class=\"good\">Bien {}</span><span class=\"mid\">Mejorable {}</span><span class=\"bad\">Mal {}</span></div><div class=\"blocks\"><article><h2>Qué funcionó</h2>{}</article><article><h2>Qué no funcionó</h2>{}</article><article><h2>Por qué pasó</h2>{}</article></div>",i+1,esc(&p.phase.to_uppercase()),esc(&p.phase),p.good,p.improvable,p.bad,bullets(&p.worked),bullets(&p.failed),bullets(&p.why)));
        for id in p.best.iter().chain(&p.worst).take(4) {
            if let Some(src) = thumbs.get(id) {
                let label = names.get(id).map(String::as_str).unwrap_or(id);
                h.push_str(&format!(
                    "<figure><img src=\"{}\" alt=\"{}\"><figcaption>{}</figcaption></figure>",
                    esc(src),
                    esc(label),
                    esc(label)
                ))
            }
        }
        h.push_str("</section>")
    }
    h.push_str("</main></html>");
    h
}
fn bullets(v: &[String]) -> String {
    if v.is_empty() {
        return "<p>Sin datos suficientes.</p>".into();
    }
    format!(
        "<ul>{}</ul>",
        v.iter()
            .map(|s| format!(
                "<li>{}</li>",
                s.replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;")
                    .replace('\'', "&#39;")
            ))
            .collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::football::eval::Rule;
    use std::collections::BTreeMap;

    fn item(key: &str, name: &str, id: &str, verdict: Verdict, kpi: &str, v: f64) -> Item {
        Item {
            action_key: key.into(),
            action_name: name.into(),
            play_id: id.into(),
            play_name: id.into(),
            start: 0.0,
            verdict,
            measures: Measures::from([(kpi.to_string(), Some(v))]),
        }
    }

    fn profile() -> Profile {
        let rule = |kpi: &str, min: Option<f64>, max: Option<f64>| Rule {
            kpi: kpi.into(),
            min,
            max,
            weight: 1.0,
            major: false,
        };
        Profile {
            name: "Atlas".into(),
            rules: BTreeMap::from([
                (
                    "press_after_loss".to_string(),
                    vec![rule("block_length", None, Some(33.0))],
                ),
                (
                    "build_up".to_string(),
                    vec![rule("width", Some(45.0), None)],
                ),
            ]),
        }
    }

    #[test]
    fn phases_say_what_worked_failed_and_why() {
        let mut items = vec![
            item(
                "press_after_loss",
                "Presión tras pérdida",
                "p1",
                Verdict::Bad,
                "block_length",
                41.0,
            ),
            item(
                "press_after_loss",
                "Presión tras pérdida",
                "p2",
                Verdict::Bad,
                "block_length",
                44.0,
            ),
            item(
                "press_after_loss",
                "Presión tras pérdida",
                "p3",
                Verdict::Good,
                "block_length",
                30.0,
            ),
        ];
        for n in 0..4 {
            items.push(item(
                "build_up",
                "Salida de balón",
                &format!("s{n}"),
                Verdict::Good,
                "width",
                48.0,
            ));
        }
        let phases = build(&items, &profile());
        assert_eq!(phases[0].phase, "Salida de balón", "build-up comes first");
        assert_eq!(phases[0].good, 4);
        assert!(
            phases[0].worked[0].contains("En 4 de 4 salidas"),
            "{:?}",
            phases[0].worked
        );
        assert!(
            phases[0].worked[0].contains("≥ 45 m"),
            "{:?}",
            phases[0].worked
        );
        let press = &phases[1];
        assert_eq!((press.good, press.bad), (1, 2));
        assert!(
            press.failed[0].contains("2 de 3 presiones"),
            "{:?}",
            press.failed
        );
        assert!(
            press.failed[0].contains("por encima de 33 m"),
            "{:?}",
            press.failed
        );
        assert!(
            press.why[0].contains("El bloque se estira"),
            "{:?}",
            press.why
        );
    }

    #[test]
    fn the_model_kpis_count_losses_regains_and_arrivals() {
        use crate::football::eval::{Placed, Sides, Snapshot, possession};
        // Team 1 attacks to the right: it holds the ball deep in the rival
        // box with three players there, loses it, wins it back in 4 s.
        let snap = |ms: u64, holder: u8, ball: (f64, f64)| {
            let mut players: Vec<Placed> = (0..10)
                .map(|n| Placed {
                    id: n,
                    team: 1,
                    keeper: false,
                    at: (40.0 + n as f64 * 2.0, 20.0 + n as f64 * 3.0),
                    role: None,
                })
                .collect();
            players.extend((0..10).map(|n| Placed {
                id: 100 + n,
                team: 2,
                keeper: false,
                at: (70.0 + n as f64, 15.0 + n as f64 * 4.0),
                role: None,
            }));
            // Three of team 1 in the rival box.
            for (i, p) in players.iter_mut().take(3).enumerate() {
                p.at = (95.0, 30.0 + i as f64 * 3.0);
            }
            let carrier = players
                .iter_mut()
                .find(|p| p.team == holder && (holder != 1 || p.at.0 < 90.0))
                .expect("a carrier");
            carrier.at = ball;
            Snapshot {
                ms,
                players,
                ball: Some((ball.0 + 0.5, ball.1)),
            }
        };
        let mut snaps = Vec::new();
        for i in 0..6u64 {
            snaps.push(snap(i * 1000, 1, (80.0 + i as f64 * 3.0, 34.0)));
        }
        for i in 6..9u64 {
            snaps.push(snap(i * 1000, 2, (80.0, 40.0)));
        }
        for i in 9..14u64 {
            snaps.push(snap(i * 1000, 1, (60.0, 34.0)));
        }
        let ctx = Context {
            possession: possession(&snaps),
            sides: Sides::of(&snaps),
            snaps,
        };
        let k = match_kpis(&ctx, 1);
        assert_eq!(k.losses, 1, "{k:?}");
        assert_eq!(k.recoveries_6s, 1, "{k:?}");
        assert!(k.final_third >= 1, "{k:?}");
        assert!(k.box_arrivals_3 >= 1, "{k:?}");
        let cards = kpi_cards(&k);
        assert_eq!(cards.len(), 4);
        let html = to_html_with("I", "S", &cards, &[], &HashMap::new(), &HashMap::new());
        assert!(html.contains("Recuperaciones tras pérdida"));
    }

    #[test]
    fn html_is_escaped_and_has_the_three_blocks() {
        let items = vec![item(
            "press_after_loss",
            "Presión <tras> pérdida",
            "p1",
            Verdict::Bad,
            "block_length",
            41.0,
        )];
        let phases = build(&items, &profile());
        let html = to_html(
            "Informe <U23>",
            "Atlas",
            &phases,
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(html.contains("Informe &lt;U23&gt;"));
        assert!(!html.contains("<U23>"));
        for heading in ["Qué funcionó", "Qué no funcionó", "Por qué pasó"] {
            assert!(html.contains(heading), "{heading}");
        }
    }
}

/// The game model's match KPIs, counted over a whole analysis for one team,
/// the way the staff's report counts them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MatchKpis {
    /// Balls the team lost.
    pub losses: usize,
    /// Of those, won back within six seconds.
    pub recoveries_6s: usize,
    /// Possessions that took the ball into the final third.
    pub final_third: usize,
    /// Possessions that reached the rival box with three or more of the
    /// team's outfield players in it.
    pub box_arrivals_3: usize,
    /// Times the rival reached the team's box within fifteen seconds of the
    /// team losing the ball.
    pub conceded_after_loss: usize,
}

/// Whether a point is inside the box at the end `goal` names.
fn in_box(goal: Goal, (x, y): (f64, f64)) -> bool {
    eval::depth(goal, x) <= 16.5 && (y - 34.0).abs() <= 20.16
}

/// Counts the model's KPIs for `team` over the match `ctx` reads.
pub fn match_kpis(ctx: &Context, team: u8) -> MatchKpis {
    let mut out = MatchKpis::default();
    let changes = eval::changes(&ctx.possession);
    for (index, change) in changes.iter().enumerate() {
        if change.from != team {
            continue;
        }
        out.losses += 1;
        if changes[index + 1..]
            .iter()
            .take_while(|c| c.ms <= change.ms + 6000)
            .any(|c| c.to == team)
        {
            out.recoveries_6s += 1;
        }
        let Some(own) = ctx.sides.goal(team, change.ms) else {
            continue;
        };
        let reached = ctx
            .snaps
            .iter()
            .filter(|s| s.ms > change.ms && s.ms <= change.ms + 15_000)
            .any(|s| s.ball.is_some_and(|b| in_box(own, b)));
        if reached {
            out.conceded_after_loss += 1;
        }
    }
    // The team's possessions, as runs of instants it holds the ball.
    let mut spell: Option<(bool, bool)> = None;
    for (snap, (_, holder)) in ctx.snaps.iter().zip(&ctx.possession) {
        if *holder != Some(team) {
            if let Some((third, arrival)) = spell.take() {
                out.final_third += usize::from(third);
                out.box_arrivals_3 += usize::from(arrival);
            }
            continue;
        }
        let state = spell.get_or_insert((false, false));
        let (Some(own), Some(ball)) = (ctx.sides.goal(team, snap.ms), snap.ball) else {
            continue;
        };
        let theirs = if own == Goal::Left {
            Goal::Right
        } else {
            Goal::Left
        };
        if eval::depth(own, ball.0) >= 70.0 {
            state.0 = true;
        }
        if in_box(theirs, ball) {
            let inside = snap
                .players
                .iter()
                .filter(|p| p.team == team && !p.keeper && in_box(theirs, p.at))
                .count();
            if inside >= 3 {
                state.1 = true;
            }
        }
    }
    if let Some((third, arrival)) = spell {
        out.final_third += usize::from(third);
        out.box_arrivals_3 += usize::from(arrival);
    }
    out
}

/// The KPI cards of a report, in the staff's order: the number, what it
/// counts, and how it reads.
pub fn kpi_cards(k: &MatchKpis) -> Vec<(String, String, String)> {
    vec![
        (
            k.recoveries_6s.to_string(),
            "Recuperaciones tras pérdida".into(),
            format!("en menos de 6 s · de {} pérdidas", k.losses),
        ),
        (
            k.final_third.to_string(),
            "Progresiones al último tercio".into(),
            "con balón controlado".into(),
        ),
        (
            k.box_arrivals_3.to_string(),
            "Llegadas al área".into(),
            "con 3 o más jugadoras".into(),
        ),
        (
            k.conceded_after_loss.to_string(),
            "Llegadas rivales a nuestra área".into(),
            "en 15 s tras pérdida propia".into(),
        ),
    ]
}
