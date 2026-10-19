// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Linking detections into tracks: one id per player for as long as he
//! stays in view.
//!
//! A broadcast camera pans, so a player's box moves across the picture
//! even when he stands still, and a tracker that predicts each box from
//! its own velocity alone loses everyone at the start of a pan. Each step
//! therefore first measures the camera: every live track is paired with
//! its nearest detection, and the median of those displacements is how far
//! the picture moved. Predictions are shifted by it before the real
//! matching, which is by overlap, greedily, best pair first.
//!
//! The ball is not tracked like a person. There is one ball, so it has
//! one id, [`super::BALL`]; at each instant the detection nearest where
//! it was is the ball, or the surest one when it has been lost a while.
//!
//! Afterwards, tracks seen too briefly to be anyone are dropped, and short
//! gaps in the rest are filled in by moving the box in a straight line
//! across them, so a player hidden behind another for a moment keeps his
//! mark.

use std::collections::HashMap;

use super::{BALL, Detection, Kind, Sample, TrackedBox, centre, iou, lerp_box};

/// A track unseen for this many instants has left.
const LOST_AFTER: u32 = 30;
/// A detection this sure may start a track.
const START_SCORE: f32 = 0.35;
/// Below this overlap with a prediction, a detection is somebody else.
const MATCH_IOU: f32 = 0.1;
/// A track seen fewer times than this was noise.
const MIN_SEEN: usize = 3;
/// Gaps up to this many instants are filled in.
const FILL_GAP: u64 = 10;
/// The ball moves fast: a detection this far from where it was, in
/// fractions of the picture, can still be it.
const BALL_REACH: f32 = 0.2;

struct Live {
    id: u32,
    bbox: [f32; 4],
    velocity: [f32; 2],
    unseen: u32,
    kinds: HashMap<Kind, u32>,
}

impl Live {
    fn predicted(&self, camera: [f32; 2]) -> [f32; 4] {
        let steps = (self.unseen + 1) as f32;
        let dx = self.velocity[0] * steps + camera[0];
        let dy = self.velocity[1] * steps + camera[1];
        [
            self.bbox[0] + dx,
            self.bbox[1] + dy,
            self.bbox[2] + dx,
            self.bbox[3] + dy,
        ]
    }
}

/// Links detections, one list per analysed instant at `step_ms` apart
/// starting at `first_ms`, into the samples of a track set.
pub fn link(frames: &[(u64, Vec<Detection>)], step_ms: u64) -> Vec<Sample> {
    let mut live: Vec<Live> = Vec::new();
    let mut next_id = BALL + 1;
    let mut samples: Vec<Sample> = Vec::with_capacity(frames.len());
    let mut ball: Option<([f32; 4], u32)> = None;

    for (ms, detections) in frames {
        let people: Vec<&Detection> = detections.iter().filter(|d| d.kind.is_person()).collect();

        // The camera's move: the median displacement from each live track
        // to its nearest detection.
        let camera = camera_shift(&live, &people);

        // Every pair that overlaps enough, best first.
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (t, track) in live.iter().enumerate() {
            let predicted = track.predicted(camera);
            for (d, detection) in people.iter().enumerate() {
                let overlap = iou(&predicted, &detection.bbox);
                if overlap >= MATCH_IOU {
                    pairs.push((overlap, t, d));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut track_taken = vec![false; live.len()];
        let mut detection_taken = vec![false; people.len()];
        let mut boxes: Vec<TrackedBox> = Vec::new();
        for (_, t, d) in pairs {
            if track_taken[t] || detection_taken[d] {
                continue;
            }
            track_taken[t] = true;
            detection_taken[d] = true;
            let detection = people[d];
            let track = &mut live[t];
            let (ox, oy) = centre(&track.bbox);
            let (nx, ny) = centre(&detection.bbox);
            let steps = (track.unseen + 1) as f32;
            // The player's own motion, the camera's taken out, smoothed.
            let own = [
                (nx - ox - camera[0]) / steps,
                (ny - oy - camera[1]) / steps,
            ];
            track.velocity = [
                track.velocity[0] * 0.6 + own[0] * 0.4,
                track.velocity[1] * 0.6 + own[1] * 0.4,
            ];
            track.bbox = detection.bbox;
            track.unseen = 0;
            *track.kinds.entry(detection.kind).or_insert(0) += 1;
            boxes.push(TrackedBox {
                id: track.id,
                kind: detection.kind,
                bbox: detection.bbox,
                score: detection.score,
                team: 0,
            });
        }
        for (t, track) in live.iter_mut().enumerate() {
            if !track_taken[t] {
                track.unseen += 1;
                // An unseen player drifts with the camera, not on his own.
                let dx = camera[0];
                let dy = camera[1];
                track.bbox = [
                    track.bbox[0] + dx,
                    track.bbox[1] + dy,
                    track.bbox[2] + dx,
                    track.bbox[3] + dy,
                ];
            }
        }
        live.retain(|track| track.unseen <= LOST_AFTER);
        for (d, detection) in people.iter().enumerate() {
            if detection_taken[d] || detection.score < START_SCORE {
                continue;
            }
            let id = next_id;
            next_id += 1;
            let mut kinds = HashMap::new();
            kinds.insert(detection.kind, 1);
            live.push(Live {
                id,
                bbox: detection.bbox,
                velocity: [0.0, 0.0],
                unseen: 0,
                kinds,
            });
            boxes.push(TrackedBox {
                id,
                kind: detection.kind,
                bbox: detection.bbox,
                score: detection.score,
                team: 0,
            });
        }

        // The ball: nearest where it was, or the surest when lost.
        let balls: Vec<&Detection> = detections
            .iter()
            .filter(|d| d.kind == Kind::Ball)
            .collect();
        let chosen = match ball {
            Some((last, unseen)) if unseen < LOST_AFTER => {
                let (lx, ly) = centre(&last);
                let reach = BALL_REACH * (unseen + 1) as f32;
                balls
                    .iter()
                    .map(|d| {
                        let (x, y) = centre(&d.bbox);
                        ((x - lx - camera[0]).hypot(y - ly - camera[1]), *d)
                    })
                    .filter(|(distance, _)| *distance <= reach)
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, d)| d)
                    .or_else(|| sure_ball(&balls))
            }
            _ => sure_ball(&balls),
        };
        match chosen {
            Some(detection) => {
                ball = Some((detection.bbox, 0));
                boxes.push(TrackedBox {
                    id: BALL,
                    kind: Kind::Ball,
                    bbox: detection.bbox,
                    score: detection.score,
                    team: 0,
                });
            }
            None => {
                if let Some((_, unseen)) = ball.as_mut() {
                    *unseen += 1;
                }
            }
        }

        samples.push(Sample { ms: *ms, boxes });
    }

    // Each track wears the kind it was seen as most.
    let mut votes: HashMap<u32, HashMap<Kind, u32>> = HashMap::new();
    for sample in &samples {
        for tracked in &sample.boxes {
            *votes
                .entry(tracked.id)
                .or_default()
                .entry(tracked.kind)
                .or_insert(0) += 1;
        }
    }
    let kind_of: HashMap<u32, Kind> = votes
        .into_iter()
        .filter_map(|(id, counts)| {
            counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(kind, _)| (id, kind))
        })
        .collect();
    let seen: HashMap<u32, usize> = samples
        .iter()
        .flat_map(|sample| sample.boxes.iter().map(|tracked| tracked.id))
        .fold(HashMap::new(), |mut counts, id| {
            *counts.entry(id).or_insert(0) += 1;
            counts
        });
    for sample in &mut samples {
        sample
            .boxes
            .retain(|tracked| seen.get(&tracked.id).copied().unwrap_or(0) >= MIN_SEEN);
        for tracked in &mut sample.boxes {
            if let Some(kind) = kind_of.get(&tracked.id) {
                tracked.kind = *kind;
            }
        }
    }
    fill_gaps(&mut samples, step_ms);
    samples
}

/// Gives the tracks of a fresh analysis the ids they had in the analysis
/// before it, where they are the same player: a mark on a clip names its
/// player by id, and a re-analysis that numbered everyone afresh would
/// move it onto somebody else.
///
/// A new track is the old one it overlaps most across the instants both
/// analyses cover, if that overlap averages more than half; each old id
/// goes to one new track at most, the best match first. A track with no
/// match gets an id neither analysis used, so it can never take over an
/// old player's marks. The ball keeps [`BALL`] throughout.
pub fn keep_ids(old: &[Sample], new: &mut [Sample]) {
    /// Below this mean overlap, a new track is somebody else.
    const SAME: f32 = 0.5;
    let old_at: HashMap<u64, &Sample> = old.iter().map(|sample| (sample.ms, sample)).collect();
    // For each new track: how many instants it is in where the old
    // analysis looked, and the summed overlap with each old track there.
    let mut seen: HashMap<u32, u32> = HashMap::new();
    let mut overlap: HashMap<(u32, u32), f32> = HashMap::new();
    for sample in new.iter() {
        let Some(before) = old_at.get(&sample.ms) else {
            continue;
        };
        for tracked in sample.boxes.iter().filter(|tracked| tracked.id != BALL) {
            *seen.entry(tracked.id).or_insert(0) += 1;
            for was in before.boxes.iter().filter(|was| was.id != BALL) {
                let shared = iou(&tracked.bbox, &was.bbox);
                if shared > 0.0 {
                    *overlap.entry((tracked.id, was.id)).or_insert(0.0) += shared;
                }
            }
        }
    }
    let mut pairs: Vec<(f32, u32, u32)> = overlap
        .into_iter()
        .map(|((now, was), sum)| (sum / seen[&now].max(1) as f32, now, was))
        .filter(|(mean, _, _)| *mean > SAME)
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut renamed: HashMap<u32, u32> = HashMap::new();
    let mut taken: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (_, now, was) in pairs {
        if renamed.contains_key(&now) || taken.contains(&was) {
            continue;
        }
        renamed.insert(now, was);
        taken.insert(was);
    }
    let highest = old
        .iter()
        .chain(new.iter())
        .flat_map(|sample| sample.boxes.iter().map(|tracked| tracked.id))
        .max()
        .unwrap_or(BALL);
    let mut next = highest + 1;
    let mut fresh: HashMap<u32, u32> = HashMap::new();
    for sample in new.iter_mut() {
        for tracked in sample.boxes.iter_mut().filter(|tracked| tracked.id != BALL) {
            tracked.id = match renamed.get(&tracked.id) {
                Some(was) => *was,
                None => *fresh.entry(tracked.id).or_insert_with(|| {
                    next += 1;
                    next - 1
                }),
            };
        }
    }
}

fn sure_ball<'a>(balls: &[&'a Detection]) -> Option<&'a Detection> {
    balls
        .iter()
        .copied()
        .max_by(|a, b| a.score.total_cmp(&b.score))
}

/// How far the picture moved since the last instant: the median of each
/// live track's displacement to its nearest detection.
fn camera_shift(live: &[Live], people: &[&Detection]) -> [f32; 2] {
    if live.is_empty() || people.is_empty() {
        return [0.0, 0.0];
    }
    let mut dx: Vec<f32> = Vec::new();
    let mut dy: Vec<f32> = Vec::new();
    for track in live.iter().filter(|track| track.unseen == 0) {
        let (tx, ty) = centre(&track.bbox);
        let height = (track.bbox[3] - track.bbox[1]).max(0.01);
        if let Some((distance, (x, y))) = people
            .iter()
            .map(|d| {
                let (x, y) = centre(&d.bbox);
                ((x - tx).hypot(y - ty), (x, y))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            && distance < height * 2.0
        {
            dx.push(x - tx);
            dy.push(y - ty);
        }
    }
    if dx.len() < 3 {
        return [0.0, 0.0];
    }
    [median(&mut dx), median(&mut dy)]
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// Fills each track's short gaps with boxes moved in a straight line.
fn fill_gaps(samples: &mut [Sample], step_ms: u64) {
    let mut last: HashMap<u32, (usize, TrackedBox)> = HashMap::new();
    let mut fills: Vec<(usize, TrackedBox)> = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        for tracked in &sample.boxes {
            if let Some((was, before)) = last.get(&tracked.id) {
                let gap = index - was;
                let gap_ms = sample.ms.saturating_sub(samples[*was].ms);
                if gap > 1 && gap_ms <= FILL_GAP * step_ms.max(1) {
                    for between in was + 1..index {
                        let t = (between - was) as f32 / gap as f32;
                        fills.push((
                            between,
                            TrackedBox {
                                bbox: lerp_box(&before.bbox, &tracked.bbox, t),
                                score: 0.0,
                                ..*tracked
                            },
                        ));
                    }
                }
            }
            last.insert(tracked.id, (index, *tracked));
        }
    }
    for (index, tracked) in fills {
        samples[index].boxes.push(tracked);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn player(x: f32, score: f32) -> Detection {
        Detection {
            kind: Kind::Player,
            bbox: [x, 0.4, x + 0.03, 0.5],
            score,
        }
    }

    #[test]
    fn a_player_walking_keeps_his_id() {
        let frames: Vec<(u64, Vec<Detection>)> = (0..10)
            .map(|i| (i * 100, vec![player(0.1 + i as f32 * 0.005, 0.8)]))
            .collect();
        let samples = link(&frames, 100);
        let ids: std::collections::HashSet<u32> = samples
            .iter()
            .flat_map(|s| s.boxes.iter().map(|b| b.id))
            .collect();
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn a_pan_moves_everyone_and_nobody_is_lost() {
        // Four players, the camera panning fast enough that each box jumps
        // further than its own width between instants.
        let frames: Vec<(u64, Vec<Detection>)> = (0..8)
            .map(|i| {
                let pan = i as f32 * 0.04;
                (
                    i * 100,
                    (0..4)
                        .map(|p| player(0.1 + p as f32 * 0.15 - pan + 0.4, 0.8))
                        .collect(),
                )
            })
            .collect();
        let samples = link(&frames, 100);
        let ids: std::collections::HashSet<u32> = samples
            .iter()
            .flat_map(|s| s.boxes.iter().map(|b| b.id))
            .collect();
        assert_eq!(ids.len(), 4, "{ids:?}");
    }

    #[test]
    fn a_short_gap_is_filled_and_a_blip_is_dropped() {
        let mut frames: Vec<(u64, Vec<Detection>)> = (0..6)
            .map(|i| (i * 100, vec![player(0.2, 0.8)]))
            .collect();
        frames[3].1.clear();
        frames[1].1.push(player(0.8, 0.9));
        let samples = link(&frames, 100);
        assert_eq!(samples[3].boxes.len(), 1);
        assert!(samples.iter().all(|s| s.boxes.len() == 1));
    }

    #[test]
    fn a_reanalysis_keeps_each_players_id() {
        let frames = |shift: f32| -> Vec<(u64, Vec<Detection>)> {
            (0..6)
                .map(|i| (i * 100, vec![player(0.1 + shift, 0.8), player(0.6 + shift, 0.8)]))
                .collect()
        };
        let old = link(&frames(0.0), 100);
        let mut new = link(&frames(0.002), 100);
        // Numbered afresh, the new analysis may well swap them; keep_ids
        // must give each player back his old id.
        for sample in &mut new {
            for tracked in &mut sample.boxes {
                tracked.id = 100 + tracked.id;
            }
        }
        keep_ids(&old, &mut new);
        for (before, after) in old.iter().zip(&new) {
            for was in &before.boxes {
                let now = after
                    .boxes
                    .iter()
                    .find(|b| iou(&b.bbox, &was.bbox) > 0.5)
                    .expect("the same player");
                assert_eq!(now.id, was.id);
            }
        }
    }

    #[test]
    fn a_new_player_gets_an_unused_id() {
        let old = link(
            &(0..5).map(|i| (i * 100, vec![player(0.1, 0.8)])).collect::<Vec<_>>(),
            100,
        );
        let mut new = link(
            &(0..5).map(|i| (i * 100, vec![player(0.7, 0.8)])).collect::<Vec<_>>(),
            100,
        );
        keep_ids(&old, &mut new);
        let old_id = old[0].boxes[0].id;
        assert!(new.iter().all(|s| s.boxes.iter().all(|b| b.id != old_id)));
    }

    #[test]
    fn the_ball_has_one_id() {
        let frames: Vec<(u64, Vec<Detection>)> = (0..5)
            .map(|i| {
                (
                    i * 100,
                    vec![Detection {
                        kind: Kind::Ball,
                        bbox: [0.5 + i as f32 * 0.02, 0.5, 0.51 + i as f32 * 0.02, 0.51],
                        score: 0.3,
                    }],
                )
            })
            .collect();
        let samples = link(&frames, 100);
        assert!(samples.iter().all(|s| s.boxes.len() == 1 && s.boxes[0].id == BALL));
    }
}
