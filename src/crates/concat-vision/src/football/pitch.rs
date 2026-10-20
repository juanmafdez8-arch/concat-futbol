// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! The pitch: from the picture to metres, and the teams by their shirts.
//!
//! **Calibration** is a few points of the picture matched to the pitch
//! landmarks they show - a corner of the box, the centre spot - at one
//! instant. Four or more give a homography, by the normalised direct
//! linear transform, from source fractions to pitch metres (105 × 68, the
//! origin at a corner flag, `x` along the touchline).
//!
//! A broadcast camera moves, so a calibration holds at its own instant
//! only. **Motion** follows the camera from there: between two analysed
//! instants the players seen in both give the picture's change as a
//! similarity - a zoom and a shift, the medians of their heights' ratios
//! and of their displacements - and the changes chained carry a point at
//! any instant back to the calibrated one. A pan is not quite a shift, so
//! the further from its calibration the looser the map; several
//! calibrations along a clip each serve the instants nearest them.
//!
//! **Teams** are found from the colour of the shirts: each player's
//! torso, the grass left out, averaged over the instants he is seen, then
//! two clusters by k-means. Keepers and referees are left out of it.

use std::collections::HashMap;

use concat_core::frame::{BYTES_PER_PIXEL, Frame};
use serde::{Deserialize, Serialize};

use super::{Kind, Sample, TrackSet, centre};

/// The pitch's length and width in metres.
pub const LENGTH: f64 = 105.0;
/// See [`LENGTH`].
pub const WIDTH: f64 = 68.0;

/// The points a calibration click snaps to, in metres: the corners, the
/// halfway line's ends and the centre spot, the circle's top and bottom,
/// the boxes' and the six-yard boxes' corners, the penalty spots.
pub fn landmarks() -> Vec<(f64, f64)> {
    let (l, w) = (LENGTH, WIDTH);
    let mid = w / 2.0;
    let mut out = vec![
        (0.0, 0.0),
        (0.0, w),
        (l, 0.0),
        (l, w),
        (l / 2.0, 0.0),
        (l / 2.0, w),
        (l / 2.0, mid),
        (l / 2.0, mid - 9.15),
        (l / 2.0, mid + 9.15),
    ];
    for (goal_line, inward) in [(0.0, 1.0), (l, -1.0)] {
        for (depth, half) in [(16.5, 20.16), (5.5, 9.16)] {
            out.push((goal_line, mid - half));
            out.push((goal_line, mid + half));
            out.push((goal_line + inward * depth, mid - half));
            out.push((goal_line + inward * depth, mid + half));
        }
        out.push((goal_line + inward * 11.0, mid));
    }
    out
}

/// The landmark nearest `(x, y)` metres, if one is within `reach` metres.
pub fn snap(x: f64, y: f64, reach: f64) -> Option<(f64, f64)> {
    landmarks()
        .into_iter()
        .map(|point| ((point.0 - x).hypot(point.1 - y), point))
        .filter(|(distance, _)| *distance <= reach)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, point)| point)
}

/// A 3 × 3 matrix, row-major.
pub type Matrix = [f64; 9];

/// The homography taking each `from` point to its `to` point, by the
/// normalised DLT; least squares over more than four. `None` with fewer
/// than four, or when they are degenerate (three in a line).
pub fn homography(from: &[(f64, f64)], to: &[(f64, f64)]) -> Option<Matrix> {
    if from.len() < 4 || from.len() != to.len() {
        return None;
    }
    let (from_n, t_from) = normalise(from);
    let (to_n, t_to) = normalise(to);
    // h33 = 1: each pair gives two equations in the other eight.
    let mut ata = [[0.0f64; 8]; 8];
    let mut atb = [0.0f64; 8];
    for (&(x, y), &(u, v)) in from_n.iter().zip(&to_n) {
        let rows = [
            ([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u),
            ([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v),
        ];
        for (row, rhs) in rows {
            for i in 0..8 {
                for j in 0..8 {
                    ata[i][j] += row[i] * row[j];
                }
                atb[i] += row[i] * rhs;
            }
        }
    }
    let h = solve8(ata, atb)?;
    let normalised = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0];
    // Undo the normalisations: H = T_to⁻¹ · Hn · T_from.
    let full = multiply(&invert(&t_to)?, &multiply(&normalised, &t_from));
    (full[8].abs() > 1e-12).then(|| full.map(|value| value / full[8]))
}

/// `h` applied to a point.
pub fn apply(h: &Matrix, (x, y): (f64, f64)) -> Option<(f64, f64)> {
    let w = h[6] * x + h[7] * y + h[8];
    (w.abs() > 1e-12).then(|| {
        (
            (h[0] * x + h[1] * y + h[2]) / w,
            (h[3] * x + h[4] * y + h[5]) / w,
        )
    })
}

fn normalise(points: &[(f64, f64)]) -> (Vec<(f64, f64)>, Matrix) {
    let n = points.len() as f64;
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x / n, sy + y / n));
    let spread = points
        .iter()
        .map(|(x, y)| (x - mx).hypot(y - my))
        .sum::<f64>()
        / n;
    let s = if spread > 1e-12 {
        2f64.sqrt() / spread
    } else {
        1.0
    };
    let t = [s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0];
    (
        points
            .iter()
            .map(|(x, y)| ((x - mx) * s, (y - my) * s))
            .collect(),
        t,
    )
}

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    std::array::from_fn(|k| {
        let (i, j) = (k / 3, k % 3);
        (0..3).map(|m| a[i * 3 + m] * b[m * 3 + j]).sum()
    })
}

fn invert(m: &Matrix) -> Option<Matrix> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-15 {
        return None;
    }
    let c = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ];
    Some(c.map(|value| value / det))
}

/// Gaussian elimination with partial pivoting on an 8 × 8 system. Indexed
/// as the textbook writes it: rows and columns are the algorithm.
#[allow(clippy::needless_range_loop)]
fn solve8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..8 {
            let factor = a[row][col] / a[col][col];
            for k in col..8 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 8];
    for row in (0..8).rev() {
        let sum: f64 = (row + 1..8).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - sum) / a[row][row];
    }
    Some(x)
}

/// One calibration: points of the picture at one instant, in source
/// fractions, matched to pitch metres.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    /// The source instant, in milliseconds.
    pub ms: u64,
    /// `[u, v, x, y]`: a source fraction and the metres it shows.
    pub pairs: Vec<[f64; 4]>,
}

/// A media file's calibrations, kept beside its tracks.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// The calibrations, any order.
    pub anchors: Vec<Anchor>,
}

impl Calibration {
    /// Reads `calibration.json` from a media's football folder.
    pub fn load(dir: &std::path::Path) -> Calibration {
        std::fs::read(dir.join("calibration.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    /// Writes it beside the tracks.
    pub fn save(&self, dir: &std::path::Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        std::fs::write(dir.join("calibration.json"), bytes).map_err(|error| error.to_string())
    }
}

/// The picture's change from one analysed instant to the next, as a
/// similarity: a point at the first lands at `s · p + (dx, dy)` at the
/// second.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Step {
    s: f64,
    dx: f64,
    dy: f64,
}

/// The camera followed along a track set: for each analysed instant, the
/// similarity taking a point there to where it would be at the first.
pub struct Motion {
    times: Vec<u64>,
    /// `(s, dx, dy)` with `p_first = s · p_i + (dx, dy)`.
    to_first: Vec<(f64, f64, f64)>,
}

impl Motion {
    /// The camera's motion through `set`, measured from the players seen
    /// at consecutive instants.
    pub fn of(set: &TrackSet) -> Motion {
        let mut times = Vec::with_capacity(set.samples.len());
        let mut to_first = Vec::with_capacity(set.samples.len());
        let mut acc = (1.0, 0.0, 0.0);
        for (index, sample) in set.samples.iter().enumerate() {
            if index > 0 {
                // Step from sample i-1 to i; its inverse takes i back.
                let step = step_between(&set.samples[index - 1], sample);
                let (s, dx, dy) = (1.0 / step.s, -step.dx / step.s, -step.dy / step.s);
                // p_first = acc(p_{i-1}) and p_{i-1} = inv(p_i).
                acc = (acc.0 * s, acc.0 * dx + acc.1, acc.0 * dy + acc.2);
            }
            times.push(sample.ms);
            to_first.push(acc);
        }
        Motion { times, to_first }
    }

    fn at(&self, ms: u64) -> Option<(f64, f64, f64)> {
        let index = self.times.partition_point(|&t| t < ms);
        let near = [index.checked_sub(1), Some(index)]
            .into_iter()
            .flatten()
            .filter(|&i| i < self.times.len())
            .min_by_key(|&i| self.times[i].abs_diff(ms))?;
        Some(self.to_first[near])
    }

    /// A source point at `from_ms` carried to where it would be at `to_ms`.
    pub fn carry(&self, (u, v): (f64, f64), from_ms: u64, to_ms: u64) -> Option<(f64, f64)> {
        let a = self.at(from_ms)?;
        let b = self.at(to_ms)?;
        // To the first instant, then back out to `to_ms`.
        let (x, y) = (a.0 * u + a.1, a.0 * v + a.2);
        Some(((x - b.1) / b.0, (y - b.2) / b.0))
    }
}

fn step_between(a: &Sample, b: &Sample) -> Step {
    let mut ratios = Vec::new();
    let mut pairs = Vec::new();
    for first in a
        .boxes
        .iter()
        .filter(|t| t.kind.is_person() && t.score > 0.0)
    {
        if let Some(second) = b.boxes.iter().find(|t| t.id == first.id && t.score > 0.0) {
            let ha = f64::from(first.bbox[3] - first.bbox[1]);
            let hb = f64::from(second.bbox[3] - second.bbox[1]);
            if ha > 1e-4 {
                ratios.push(hb / ha);
            }
            let (ax, ay) = centre(&first.bbox);
            let (bx, by) = centre(&second.bbox);
            pairs.push((
                (f64::from(ax), f64::from(ay)),
                (f64::from(bx), f64::from(by)),
            ));
        }
    }
    if pairs.len() < 3 {
        return Step {
            s: 1.0,
            dx: 0.0,
            dy: 0.0,
        };
    }
    let s = median(&mut ratios).clamp(0.8, 1.25);
    let mut dxs: Vec<f64> = pairs.iter().map(|((ax, _), (bx, _))| bx - s * ax).collect();
    let mut dys: Vec<f64> = pairs.iter().map(|((_, ay), (_, by))| by - s * ay).collect();
    Step {
        s,
        dx: median(&mut dxs),
        dy: median(&mut dys),
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// The map from source fractions at `ms` to pitch metres: the nearest
/// calibration's homography, its points carried there by the camera's
/// motion. `None` with no calibration.
pub fn map_at(calibration: &Calibration, motion: &Motion, ms: u64) -> Option<Matrix> {
    let anchor = calibration
        .anchors
        .iter()
        .filter(|anchor| anchor.pairs.len() >= 4)
        .min_by_key(|anchor| anchor.ms.abs_diff(ms))?;
    let mut from = Vec::new();
    let mut to = Vec::new();
    for pair in &anchor.pairs {
        let carried = motion.carry((pair[0], pair[1]), anchor.ms, ms)?;
        from.push(carried);
        to.push((pair[2], pair[3]));
    }
    homography(&from, &to)
}

/// A player's feet in a frame: the middle of his box's foot.
pub fn feet(bbox: &[f32; 4]) -> (f64, f64) {
    (f64::from(bbox[0] + bbox[2]) * 0.5, f64::from(bbox[3]))
}

/// The average colour of a person's shirt: the upper middle of his box,
/// the grass and the very dark and light left out.
pub fn shirt_colour(frame: &Frame, bbox: &[f32; 4]) -> Option<[u8; 3]> {
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let [l, t, r, b] = *bbox;
    let (w, h) = (r - l, b - t);
    let x0 = ((l + w * 0.3) * fw).max(0.0) as u32;
    let x1 = (((l + w * 0.7) * fw) as u32).min(frame.width());
    let y0 = ((t + h * 0.18) * fh).max(0.0) as u32;
    let y1 = (((t + h * 0.48) * fh) as u32).min(frame.height());
    let pixels = frame.pixels();
    let width = frame.width() as usize;
    let (mut sum, mut n) = ([0u64; 3], 0u64);
    for y in y0..y1 {
        for x in x0..x1 {
            let at = (y as usize * width + x as usize) * BYTES_PER_PIXEL;
            let (red, green, blue) = (pixels[at], pixels[at + 1], pixels[at + 2]);
            let grass = green > red.saturating_add(15) && green > blue.saturating_add(15);
            if grass {
                continue;
            }
            sum[0] += u64::from(red);
            sum[1] += u64::from(green);
            sum[2] += u64::from(blue);
            n += 1;
        }
    }
    (n >= 4).then(|| sum.map(|channel| (channel / n) as u8))
}

/// Two teams from the players' shirt colours: each track's average
/// colour, then two clusters by k-means, seeded by the two colours furthest
/// apart. The track ids of each team, and each team's colour. Tracks seen
/// fewer than three times are left out.
pub fn teams(colours: &HashMap<u32, Vec<[u8; 3]>>) -> (HashMap<u32, u8>, [[u8; 3]; 2]) {
    let tracks: Vec<(u32, [f64; 3])> = colours
        .iter()
        .filter(|(_, seen)| seen.len() >= 3)
        .map(|(id, seen)| {
            let n = seen.len() as f64;
            let mean = seen.iter().fold([0.0; 3], |acc, c| {
                [
                    acc[0] + f64::from(c[0]) / n,
                    acc[1] + f64::from(c[1]) / n,
                    acc[2] + f64::from(c[2]) / n,
                ]
            });
            (*id, mean)
        })
        .collect();
    let mut out = HashMap::new();
    if tracks.len() < 2 {
        return (out, [[230, 230, 230], [40, 40, 40]]);
    }
    let distance = |a: &[f64; 3], b: &[f64; 3]| {
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
    };
    let (mut c0, mut c1) = (tracks[0].1, tracks[1].1);
    let mut best = -1.0;
    for a in &tracks {
        for b in &tracks {
            let d = distance(&a.1, &b.1);
            if d > best {
                best = d;
                c0 = a.1;
                c1 = b.1;
            }
        }
    }
    for _ in 0..12 {
        let (mut s0, mut s1, mut n0, mut n1) = ([0.0; 3], [0.0; 3], 0.0, 0.0);
        for (_, colour) in &tracks {
            if distance(colour, &c0) <= distance(colour, &c1) {
                (0..3).for_each(|i| s0[i] += colour[i]);
                n0 += 1.0;
            } else {
                (0..3).for_each(|i| s1[i] += colour[i]);
                n1 += 1.0;
            }
        }
        if n0 > 0.0 {
            c0 = s0.map(|v| v / n0);
        }
        if n1 > 0.0 {
            c1 = s1.map(|v| v / n1);
        }
    }
    for (id, colour) in &tracks {
        let team = if distance(colour, &c0) <= distance(colour, &c1) {
            1
        } else {
            2
        };
        out.insert(*id, team);
    }
    let to_u8 = |c: [f64; 3]| c.map(|v| v.round().clamp(0.0, 255.0) as u8);
    (out, [to_u8(c0), to_u8(c1)])
}

/// Whether a kind takes part in the team split: outfield players only.
pub fn in_teams(kind: Kind) -> bool {
    kind == Kind::Player
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::football::TrackedBox;

    #[test]
    fn a_homography_maps_its_points_and_those_between() {
        // A trapezoid in the picture, as a pitch seen from the stand.
        let from = [(0.2, 0.3), (0.8, 0.3), (0.95, 0.9), (0.05, 0.9)];
        let to = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)];
        let h = homography(&from, &to).expect("a homography");
        for (a, b) in from.iter().zip(&to) {
            let p = apply(&h, *a).unwrap();
            assert!(
                (p.0 - b.0).abs() < 1e-6 && (p.1 - b.1).abs() < 1e-6,
                "{p:?} {b:?}"
            );
        }
        // The middle of the far edge is the halfway line's far end.
        let p = apply(&h, (0.5, 0.3)).unwrap();
        assert!((p.0 - 52.5).abs() < 1e-6 && p.1.abs() < 1e-6);
    }

    #[test]
    fn three_points_in_a_line_are_no_homography() {
        let from = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)];
        let to = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(homography(&from, &to).is_none());
    }

    #[test]
    fn a_click_snaps_to_the_nearest_landmark() {
        assert_eq!(snap(51.0, 35.0, 3.0), Some((52.5, 34.0)));
        assert_eq!(snap(30.0, 30.0, 3.0), None);
        assert_eq!(landmarks().len(), 9 + 2 * 9);
    }

    fn person(id: u32, x: f32, y: f32, h: f32) -> TrackedBox {
        TrackedBox {
            id,
            kind: Kind::Player,
            bbox: [x - 0.01, y - h, x + 0.01, y],
            score: 0.9,
            team: 0,
        }
    }

    #[test]
    fn a_pan_is_followed_and_undone() {
        // Three players, the camera shifting 0.05 left an instant.
        let samples: Vec<Sample> = (0..4)
            .map(|i| Sample {
                ms: i * 200,
                boxes: (0..3)
                    .map(|p| person(p + 2, 0.3 + 0.2 * p as f32 - 0.05 * i as f32, 0.6, 0.1))
                    .collect(),
            })
            .collect();
        let set = TrackSet {
            version: 1,
            model: String::new(),
            aspect: 16.0 / 9.0,
            samples,
            team_colours: Vec::new(),
        };
        let motion = Motion::of(&set);
        // A point at the first instant is 0.15 further left at the fourth.
        let (u, v) = motion.carry((0.5, 0.5), 0, 600).unwrap();
        assert!((u - 0.35).abs() < 1e-4 && (v - 0.5).abs() < 1e-4, "{u} {v}");
    }

    #[test]
    fn two_kits_split_into_two_teams() {
        let mut colours = HashMap::new();
        for id in 0..5u32 {
            colours.insert(id, vec![[220, 30, 30]; 4]);
        }
        for id in 5..10u32 {
            colours.insert(id, vec![[240, 240, 240]; 4]);
        }
        let (teams, kits) = teams(&colours);
        assert_eq!(teams.len(), 10);
        assert_ne!(teams[&0], teams[&7]);
        assert!((0..5).all(|id| teams[&id] == teams[&0]));
        assert!(kits.iter().any(|kit| kit[0] > 200 && kit[1] < 60));
    }
}
